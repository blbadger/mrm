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
from transformers import TextStreamer
from dual_srm import DualMLPMixer
import warnings
import time
import uuid
import pprint
warnings.simplefilter(action='ignore', category=UserWarning)


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
		is_reward_model=False,
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
		self.is_reward_model = is_reward_model
		if is_reward_model:
			self.loss_fn = nn.MSELoss()
			self.reward_head = nn.Linear(self.hidden_dim, 1)
	
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

	def get_cache(self):
		cache_store = {}
		for block in self.mixer_blocks:
			for h in range(len(block.token_mixing_layer.mixer_heads)):
				cache_store[f'{block},{h}'] = block.token_mixing_layer.mixer_heads[h].cache
		return cache_store

	def load_cache(self, cache_store):
		for block in self.mixer_blocks:
			for h in range(len(block.token_mixing_layer.mixer_heads)):
				block.token_mixing_layer.mixer_heads[h].cache = cache_store[f'{block},{h}']
		return

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
		print (labels)
		is_recurrent = input_ids.shape[1] < self.seq_len
		# mask pad tokens in labels for loss computation
		if labels is not None:
			labels = torch.where(labels==tokenizer.pad_token_id, -100., labels).to(self.input_layer.weight.dtype)
		if not self.cache_built and is_recurrent:
			self.build_cache(input_ids)
		index = input_ids.shape[1] - 1
		if is_recurrent:
			input_ids = input_ids[:, -1] # last token only
		# model's forward pass
		x = self.input_layer(input_ids)
		for block in self.mixer_blocks:
			x = block(x, index, is_recurrent)
		if not self.is_reward_model:
			logits = self.output_layer(x).unsqueeze(1)
		else:
			# reward model output
			output = self.reward_head(x).squeeze(-1)
			if labels is not None:
				loss = self.loss_fn(output, labels)
			else:
				loss= 0
			return CausalLMOutput(loss=loss, logits=output)

		# policy model output
		if labels is not None:
			shift_logits = logits[:, :-1].contiguous()
			shift_labels = labels[:, 1:].contiguous()
			shift_logits = shift_logits.view(-1, self.vocab_size)
			shift_labels = shift_labels.view(-1)
			loss = self.loss_fn(shift_logits, shift_labels)
			return CausalLMOutput(loss=loss, logits=logits)
		else:
			return CausalLMOutput(loss=0, logits=logits)


# can be asynchronously computed on CPU during GPU generations
def convert_generations_to_tree(tokens):
	tree = {}
	parent_map = {} # maps parent token index to uuid
	first_tokens = {}
	for j, token in enumerate(tokens[:, 0]):
		if token.item() not in first_tokens:
			new_id = uuid.uuid4()
			tree[new_id] = {'token': token.item(), 'value': [], 'parent': None, 'children': [], 'is_leaf': True, 'cache_store': None, 'index': j}
			parent_map[str(j)] = new_id
			first_tokens[token.item()] = new_id
		else:
			node_id = first_tokens[token.item()]
			parent_map[str(j)] = node_id

	for index in range(1, tokens.shape[1]):
		node_map = {} # maps (parent, current) tuple to uuid
		for j, token in enumerate(tokens[:, index]):
			if (parent_map[str(j)], token.item()) not in node_map:
				# add new node
				new_id = uuid.uuid4()
				tree[new_id] = {'token': token.item(), 'value': [], 'parent': parent_map[(str(j))], 'children': [], 'is_leaf': True, 'cache_store': None, 'index': j}
				parent = parent_map[str(j)]
				tree[parent]['children'].append(new_id)

				node_map[(parent_map[str(j)], token.item())] = new_id
			else:
				# not a unique node, assign to existing
				new_id = node_map[(parent_map[str(j)], token.item())]
			parent_map[str(j)] = new_id # replace parent map with current token
			tree[parent]['is_leaf'] = False

	return tree


# example node data structure
# {'token': 143, 'value': 0.23, 'parent': id_number, 
#  'children': [id_number...], 'is_leaf': True, 'cache_store': {['str']:torch.Tensor}}

# example cache_store data structure:
# {'mixer_blocks[0],1': torch.tensor([0.1, 0.2, 0.3]), 'mixer_blocks[0],2': torch.tensor([0.4, 0.5, 0.6])}

# example tree data structure
# {id_number: {'token': 143, 'value': 0.23, 'parent': id_number, ...}, 
# id_number_2: {'token': ...}}

def predict_next_nodes(policy_model, node, n_taken=2):
	policy_model.load_cache(node.cache_store)
	_, logits = policy_model.forward(node.token)
	sorted_logits, indices = torch.topk(logits, n_taken)
	top_indices = indices[:n_taken]
	for index in top_indices:
		# make new child node
		new_id = uuid.uuid4()
		node.children.append(new_id)
		new_node = {new_id: {'token': index, 'value': None, 'is_leaf': True, 'parent': [node.key], 'cache_store':policy_model.get_cache(), 'children': []}}
	return top_indices

def tree_backup(tree, values):
	# Algorithm: find leaves, back up each leaf value to root, and accumulate
	# NB this expects leaves to have values already
	for key, node in tree.items():
		if node['is_leaf']:
			leaf_value = values[node['index']] # assign value by index
			node['value'] = [leaf_value]
			# back up value
			while True:
				parent_id = tree[key]['parent']
				if not parent_id:
					break
				tree[parent_id]['value'].append(leaf_value)
				key = tree[key]['parent']

	for key, node in tree.items():
		node['value'] = sum(node['value']) / len(node['value'])
	return tree

def test_correctness(completions, answer, value_constant=1.0, **kwargs) -> list[float]:
	extracted_responses = [output_extract(c) for c in completions] 
	extracted_answers = [answer_extract(answer) for _ in completions] # duplicate answer
	#print('='*60, f"Question:\n{prompts[1]}", f"\nAnswer:\n{answer[1]}\n",'-'*50, f"\nResponse:\n{completions[1]}", f"\nExtracted:\n{extracted_responses[1]}\n")
	#print (extracted_responses, answer_extract(answer))
	values = [value_constant if r == a else 0.0 for r, a in zip(extracted_responses, extracted_answers)]
	return values

def get_token_sequences(tree):
	outputs = {} # maps node_id to output
	for key, node in tree.items():
		if node['is_leaf']:
			leaf_key = key
			output = []
			while True:
				parent_id = tree[key]['parent']
				token = tree[key]['token']
				output.append(token)
				if not parent_id:
					break
				key = parent_id
			output.reverse()
			outputs[leaf_key] = output
	return outputs

def get_token_values(tree):
	# values shape: [b]
	outputs = {} # maps node_id to output
	for key, node in tree.items():
		leaf_key = key
		if node['is_leaf']:
			leaf_key = key
			value = node['value']
			output = []
			while True:
				parent_id = tree[key]['parent']
				value = tree[key]['value']
				output.append(value)
				if not parent_id:
					break
				key = parent_id
			output.reverse()
			outputs[leaf_key] = output
	return outputs

def get_value_batch(tree, leaf_token_map):
	batch = []
	leaf_nodes = {key:node for key, node in tree.items() if node['is_leaf']}
	index_to_node = {str(node['index']): key for key, node in leaf_nodes.items()}
	for i in range(len(leaf_nodes)):
		if str(i) in index_to_node:
			batch.append(leaf_token_map[index_to_node[str(i)]])
	
	batch = torch.tensor(batch)
	return batch

def get_evaluations(outputs, answer):
	# expects a single answer, ie all outputs are for the same question
	# batch decode tokens
	output_tokens = [[key, value] for key, value in outputs.items()]
	output_keys = [o.keys for o in outputs]
	detokenized_outputs = tokenizer.decode(output_tokens)
	extracted_answer = answer_extract(answer)
	extracted_outputs = [output_extract(o) for o in detokenized_outputs]
	values = test_correctness(extracted_outputs, extracted_answer)
	# zip back up into outputs
	for key, tokens, string, extracted_outputs, value in zip(output_keys, output_tokens, detokenized_outputs, values):
		outputs[key] = {'tokens': tokens, 'string': string, 'value': value}
	return outputs

def assign_leaf_node_values(tree, answer):
	output_dict = get_token_sequences(tree) # maps node_id to output tokens
	evaluation_dict = get_evaluations(output_dict, answer)
	for key, val in evaluation_dict.items():
		tree[key]['value'] = val['value']
	return tree

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

def prepare_nshot(example, n_shot=3):
    # n shot append and rename fields for rl
    three_shot_prompt = '\n'.join([f"Question: {train_dataset[i]['question']} \nAnswer: {train_dataset[i]['answer']}" for i in range(n_shot)])
    example['prompt'] = f"{three_shot_prompt}\n Question: {example['question']} \n Answer:"
    example['cleaned_answer'] = answer_extract(example['answer'])
    return example

def train_loop(policy_model,
			reward_model,
			train_dataset,
			test_dataset,
			tokenizer,
			accelerator=None,
			generate_batch=512,
			train_steps=200000,
			batch_size=16,
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
	reward_model = reward_model.to(device)
	policy_model = policy_model.to(device)
	optimizer = torch.optim.AdamW(reward_model.parameters(), lr=learning_rate)
	
	reward_model, optimizer = accelerator.prepare(reward_model, optimizer)
	
	policy_model.eval()  # Freeze policy model
	reward_model.train()	
	
	# Split dataset across processes for DDP
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
		# Get a dataset element (cycle through process-specific indices)
		local_idx = step % len(process_indices)
		data_idx = process_indices[local_idx]
		data_element = train_dataset[data_idx]
		
		question = data_element['question']
		answer = data_element['answer']
		tokens_to_generate = 256

		input_ids = tokenizer.encode(question, 
			return_tensors='pt', 
			max_length=1024-tokens_to_generate, 
			padding='max_length', 
			padding_side='left').to(device)
		
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
		prompt_length = input_ids.shape[1]
		generated_tokens = generated_ids[:, prompt_length:]
		
		# Convert generations to tree structure, back up values, and process
		print ('building and processing tree')
		tree = convert_generations_to_tree(generated_ids)
		completions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
		values = test_correctness(completions, answer, value_constant)
		tree = tree_backup(tree, values)
		token_values = get_token_values(tree)
		token_sequences = get_token_sequences(tree)
		num_leaves = len([node['value'] for node in tree.values() if node['is_leaf']])
		correct_leaves = sum([node['value'] for node in tree.values() if node['is_leaf']]) / value_constant
		print (f'Correct paths: {correct_leaves}')	
		value_batch = get_value_batch(tree, token_values)
		
		# Online training batches
		num_leaves = len(token_sequences)
		pad_number = generate_batch - num_leaves
		token_sequences = torch.cat((generated_ids, generated_ids[:pad_number]), dim=0)
		num_leaves = len(generated_ids)
		value_batch = torch.cat((value_batch, value_batch[:pad_number]), dim=0)
		accumulation_steps = num_leaves // batch_size
		accelerator.wait_for_everyone()
		for batch_idx in range(0, len(generated_ids), batch_size):
			optimizer.zero_grad()
			batch_end = min(batch_idx + batch_size, num_leaves)
			batch_ids = generated_ids[batch_idx:batch_end]
			batch_values = value_batch[batch_idx:batch_end]
			batch_input_ids = torch.tensor(batch_ids, dtype=torch.long).to(device)
			batch_target_values = torch.tensor(batch_values, dtype=torch.float).to(device)
			output = reward_model(batch_input_ids, labels=batch_input_ids)
			loss = output.loss
			accelerator.backward(loss)
			total_loss += loss.item()
			optimizer.step()
			accelerator.wait_for_everyone()	
			
		# Logging
		if step % log_steps == 0:	
			accelerator.print(f"Step {step}: Loss={total_loss/(log_steps * accumulation_steps * num_leaves * 1024):.4f}, Correct Leaves={correct_leaves:.4f}, Num Leaves={num_leaves}")
			total_loss = 0.0
		# Periodic evaluation (only on main process)
		#if step % eval_steps == 0 and test_dataset is not None and accelerator.is_main_process:
		#	accelerator.print(f"Step {step}: Running evaluation...")
		#	evaluate_model(policy_model, reward_model, test_dataset, tokenizer, device)

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
	return reward_model


def evaluate_model(policy_model, reward_model, test_dataset, tokenizer, device, num_samples=10):
	"""Evaluate model on test dataset"""
	policy_model.eval()
	reward_model.eval()
	
	total_correct = 0
	total_samples = 0
	
	with torch.no_grad():
		for i in range(min(num_samples, len(test_dataset))):
			data_element = test_dataset[i]
			question = data_element['question']
			answer = data_element['answer']
			
			input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
			policy_model.clear_cache()
			
			generated_ids = policy_model.generate(
				input_ids,
				max_new_tokens=256,
				do_sample=False,
				pad_token_id=tokenizer.pad_token_id
			)
			
			completion = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
			values = test_correctness([completion], answer)
			total_correct += values[0]
			total_samples += 1
	
	accuracy = total_correct / total_samples if total_samples > 0 else 0
	print(f"Evaluation Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
	
	policy_model.train()
	reward_model.train()
	
	return accuracy


def tree_test():
	generations = torch.tensor([
	[0, 4, 5, 3],
	[0, 4, 4, 1],
	[0, 3, 5, 3],
	[0, 4, 5, 2]
	])
	values = [0, 1, 0, 0]
	tree = convert_generations_to_tree(generations)
	# print (get_token_sequences(tree))
	tree = tree_backup(tree, values)
	print (tree)
	print ('\n\n\n')
	print (get_token_values(tree))
	return

def throughput_test():
	text ='''Four score and seven years ago, our forefathers, for the purpose of a more perfect union, sought'''
	batch_size = 500
	input_ids = torch.tensor(tokenizer.encode(text)[1:]).repeat(batch_size, 1).to(device) # ignore bos token
	print (input_ids.shape)
	tokens_to_generate = 50
	streamer = TextStreamer(tokenizer, skip_prompt=False)
	start = time.time()
	print (f'Example: {tokenizer.decode(output_ids[0])}, elapsed time: {time.time() - start}, t/s: {(tokens_to_generate * batch_size)/(time.time() - start)}')
	return

if __name__ == "__main__":
	load_dotenv()
	checkpoint_root = os.getenv('CHECKPOINT_ROOT')
	data_root = os.getenv('DATA_ROOT')
	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = tokenizer.vocab_size
	print("Vocab size: ", n_vocab)

	dataset = load_dataset("openai/gsm8k", "main")
	train_dataset, eval_dataset = dataset['train'], dataset['test']
	train_dataset = train_dataset.map(prepare_nshot, num_proc=16)
	print (train_dataset[0])
	eval_dataset = eval_dataset.map(prepare_nshot, num_proc=16)

	tokenized_length = 1024
	dim = 1024
	layers = 16
	n_heads = 4
	kernel = 1

	policy_model = DualMixer(
		n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
		mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True, is_reward_model=False).float()

	reward_model = DualMixer(
		n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
		mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True, is_reward_model=True).float()

	model_path=f'{checkpoint_root}/gsm8k_SFT_srm_c1024/chkpt-300/model.safetensors'
	load_model(policy_model, model_path)
	policy_model = torch.compile(policy_model)
	reward_model = torch.compile(reward_model)

	checkpoint_dir = f"{checkpoint_root}/gsm8k_tree_reward_b512"
	train_loop(policy_model, reward_model, train_dataset,eval_dataset, tokenizer, checkpoint_dir=checkpoint_dir)
