import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation import GenerationMixin, GenerationConfig
import datasets
from datasets import load_from_disk
from safetensors.torch import load_model, save_model
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_model
from datasets import load_dataset, load_from_disk
from accelerate.utils import DistributedDataParallelKwargs
import os
import re
from dotenv import load_dotenv
import shutil
from repeat_main import MLPMixer
from cached_inference import CachedMLPMixer
from recurrent_inference import RecurrentMLPMixer
from dual_srm import DualMLPMixer
from transformers import TrainerCallback

class DualMixer(DualMLPMixer, GenerationMixin):

	def __init__(
		self,
		vocab_size: int,
		hidden_dim: int,
		seq_len: int,
		num_blocks: int,
		heads=None,
		kernel=1,
		expanded_convs=False,
		copy=False,        
		mixed_heads=False,
		combined_heads=False,
		decay=False,
		parallel_heads=False,
		use_projections=True,
		dropout_layer=False,
		**kwargs
	):
		super().__init__(vocab_size, hidden_dim, seq_len, num_blocks, heads=heads, kernel=kernel, expanded_convs=expanded_convs,
			mixed_heads=mixed_heads, combined_heads=combined_heads, decay=decay, parallel_heads=parallel_heads, use_projections=use_projections)
		self._init_weights()
		self.generation_config = GenerationConfig()
		config  = {
				 'hidden_size':hidden_dim,
				 'intermediate_size': 4*hidden_dim,
				 'num_hidden_layers': num_blocks,
				 'num_attention_heads': 4,
				 'vocab_size': vocab_size
				}
		self.config = LlamaConfig(**config)
		self.hidden_dim = hidden_dim
		self.n_heads = heads
		self.seq_len = seq_len
		self.main_input_name = 'input_ids'
		self._supports_cache_class = False
		self.cache_built = False
		self.device = self.output_layer.weight.device
		self.warnings_issued={}
	
	def add_model_tags(self, tag):
		print (tag)

	def gradient_checkpointing_enable(self, *args, **kwargs):
		pass

	def can_generate(self):
		return True

	def count_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def _is_stateful(self):
		return False

	def build_cache(self, input_ids):
		for i in range(len(input_ids[0])-1):
			x = self.input_layer(input_ids[:, i])
			for block in self.mixer_blocks:
				x = block(x, i, True)
		self.cache_built = True
		return

	def clear_cache(self):
		for block in self.mixer_blocks:
			for h in range(len(block.token_mixing_layer.mixer_heads)):
				block.token_mixing_layer.mixer_heads[h].cache = torch.zeros(self.hidden_dim//self.n_heads).to('cuda') # only for mixed heads
		self.cache_built = False

	def forward(self, input_ids, labels=None, **kwargs):
		is_recurrent = input_ids.shape[1] < self.seq_len
		if not self.cache_built and is_recurrent:
			self.build_cache(input_ids)
		index = input_ids.shape[1] - 1
		if is_recurrent:
			input_ids = input_ids[:, -1] # last token only
		# model's forward pass
		x = self.input_layer(input_ids)
		for block in self.mixer_blocks:
			x = block(x, index, is_recurrent)
		logits = self.output_layer(x).unsqueeze(1)
		if labels is not None:
			shift_logits = logits.squeeze(1)[:, :-1].contiguous()
			shift_labels = labels[:, 1:].contiguous()
			shift_logits = shift_logits.view(-1, self.vocab_size)
			shift_labels = shift_labels.view(-1)
			loss = self.loss_fn(shift_logits, shift_labels)
			return CausalLMOutput(loss=loss, logits=logits)
		else:
			return CausalLMOutput(loss=0, logits=logits)

def answer_extract(answer):
	cleaned_output = answer.split('####')
	if len(cleaned_output) > 1:
		cleaned_output = cleaned_output[1].strip(' ,!@#$%^&*')
	return cleaned_output

def output_extract(predicted_output):
	if isinstance(predicted_output, list):
		predicted_output = predicted_output[0]
	output = re.findall("(-?[$0-9.,]{2,})|(-?[0-9]+)", predicted_output)
	if output:
		output = output[-1] # matches last pattern
	outs = []
	#print (output)
	if (isinstance(output, tuple) or isinstance(output, list)) and len(output) > 1: 
		outs.append((output[0] if output[0] else output[1]).strip(' %$@!*,.'))
	else:
		if not output:
			outs.append('')
		else:
			outs.append(output.strip(' %$@!*,.'))
	cleaned_output = outs[0]
	return cleaned_output


def test_correctness(completions, answers, **kwargs) -> list[float]:
	extracted_responses = [output_extract(c) for c in completions] 
	extracted_answers = [answer_extract(a) for a in answers]
	values = [1. if r == a else 0.0 for r, a in zip(extracted_responses, extracted_answers)]
	return values

def train_loop(policy_model,
			train_dataset,
			test_dataset,
			tokenizer,
			accelerator=None,
			generate_batch=2048,
			train_steps=200000,
			batch_size=128,
			learning_rate=1e-4,
			value_constant=100.,
			log_steps=1,
			eval_steps=100,
			save_steps=200,
			checkpoint_dir=''
			):
	 
	if accelerator is None:
		ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
		accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision='fp16')
	
	if accelerator.is_main_process:
		os.makedirs(checkpoint_dir, exist_ok=True)
	device = accelerator.device
	policy_model = policy_model.to(device)
	optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
	reward_model, optimizer = accelerator.prepare(policy_model, optimizer)
	
	if accelerator.num_processes > 1:
		process_index = accelerator.process_index
		num_processes = accelerator.num_processes
		dataset_size = len(train_dataset)
		indices_per_process = dataset_size // num_processes
		start_idx = process_index * indices_per_process
		end_idx = start_idx + indices_per_process if process_index < num_processes - 1 else dataset_size
		process_indices = list(range(start_idx, end_idx))
		accelerator.print(f"Process {process_index}: Training on indices {start_idx} to {end_idx}")
	else:
		process_indices = list(range(len(train_dataset)))

	total_loss = 0.0	
	for step in range(train_steps):
		local_indices = step % len(process_indices)
		# TODO: deal with more than one input query per batch
		data_idx = process_indices[local_indices]
		data_elements = train_dataset[local_indices]
		
		questions = data_elements['question']
		answers = data_elements['answer']
		tokens_to_generate = 256
		n_ctx = 1024
		# Tokenize the question
		input_ids = tokenizer.encode(questions, 
			return_tensors='pt', 
			max_length=n_ctx-tokens_to_generate, 
			padding='max_length', 
			padding_side='left').to(device)
		
		# Generate samples using policy model
		if accelerator.is_main_process:
			accelerator.print(f"Step {step}/{train_steps}: Generating {generate_batch} samples per question...")

		if hasattr(policy_model, 'module'):
			policy_model.module.clear_cache()
		else:
			policy_model.clear_cache()
		
		with torch.no_grad():
			generated_ids = policy_model.generate(
				input_ids.repeat(generate_batch, 1),
				max_new_tokens=tokens_to_generate,
				do_sample=True,
				temperature=0.7,
				top_p=0.9,
				pad_token_id=tokenizer.pad_token_id
			).to('cpu')
		accelerator.wait_for_everyone()	
		# Remove prompt and test correctness
		prompt_length = input_ids.shape[1]
		generated_tokens = generated_ids[:, prompt_length:]
		generated_strings = tokenizer.batch_decode(generated_tokens)
		expanded_answers = [answers for _ in range(generate_batch)]
		values = test_correctness(generated_strings, expanded_answers)
		# Acceptance training (online)
		good_indices = [i for i in range(len(values)) if values[i] == 1.]
		print (f'Number of good outputs: {len(good_indices)}')
		good_generations = generated_ids[good_indices, :]
		accelerator.wait_for_everyone()
		for batch_idx in range(0, len(good_generations), batch_size):
			optimizer.zero_grad()
			batch_end = min(batch_idx + batch_size, len(good_generations))
			batch_ids = generated_ids[batch_idx:batch_end]

			batch_input_ids = batch_ids.to(device)
			batch_labels = torch.clone(batch_input_ids)
			batch_labels[:, :n_ctx-tokens_to_generate] = torch.ones((batch_labels.shape[0], n_ctx-tokens_to_generate)) * -100 # mask loss on input
			
			output = policy_model(batch_input_ids, labels=batch_labels)

			loss = output.loss
			accelerator.backward(loss)
			total_loss += loss.item()
			optimizer.step()
			accelerator.wait_for_everyone()	

		if step % log_steps == 0:
			accelerator.print(f"Step {step}: Loss={total_loss/(log_steps * 256):.4f}")
			total_loss = 0.0

		# Save checkpoint (only on main process)
		if step % save_steps == 0 and accelerator.is_main_process:
			checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
			os.makedirs(checkpoint_path, exist_ok=True)
			if hasattr(reward_model, 'module'):
				unwrapped_model = reward_model.module
			else:
				unwrapped_model = reward_model
			if hasattr(unwrapped_model, '_orig_mod'):
				unwrapped_model = unwrapped_model._orig_mod
			
			save_model(unwrapped_model, os.path.join(checkpoint_path, "model.safetensors"))		
			accelerator.print(f"Saved checkpoint to {checkpoint_path}")
		accelerator.wait_for_everyone()
	
	# Wait for all processes to finish
	accelerator.wait_for_everyone()
	return reward_model

def prepare_nshot(example, n_shot=1):
	three_shot_prompt = '\n'.join([f"Question: {train_dataset[i]['question']} \nAnswer: {train_dataset[i]['answer']}" for i in range(n_shot)])
	example['prompt'] = f"{three_shot_prompt}\n Question: {example['question']} \nAnswer :"
	example['completion'] = example['answer']
	example['text'] = example['prompt'] + '|' + example['completion']
	return example

def formatting_prompts_func(example):
	output_text = example['prompt'] + example['completion']
	return output_text


if __name__ == '__main__':
	device = "cuda" if torch.cuda.is_available() else "cpu"
	load_dotenv()
	checkpoint_root = os.getenv('CHECKPOINT_ROOT')
	data_root = os.getenv('DATA_ROOT')
	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)
	print("Vocab size: ", n_vocab)

	tokenized_length = 1024
	dim = 1024
	layers = 16
	n_heads = 4
	kernel = 1

	model = DualMixer(
		n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
		mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).to(device)

	print (model)
	dataset = load_dataset("openai/gsm8k", "main")
	train_dataset, eval_dataset = dataset['train'], dataset['test'] # positive control
	train_dataset = train_dataset.map(prepare_nshot, num_proc=16)
	eval_dataset = eval_dataset.map(prepare_nshot, num_proc=16)
	
	print (len(train_dataset))
	#model_path=f'{checkpoint_root}/fineweb_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors'
	model_path=f'{checkpoint_root}/gsm8k_sft_srm_c1024/checkpoint-1100/model.safetensors'
	load_model(model, model_path)
	print ('pretrained model loaded')
	response_template = '|'

	output_dir = f'{checkpoint_root}/gsm8k_acceptance_sampling'

	model.train()
	train_loop(model, train_dataset, eval_dataset, tokenizer, checkpoint_dir=output_dir)
