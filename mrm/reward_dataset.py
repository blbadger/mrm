import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, get_linear_schedule_with_warmup
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
import random
from dotenv import load_dotenv
import shutil
from repeat_main import MLPMixer
from cached_inference import CachedMLPMixer
from recurrent_inference import RecurrentMLPMixer
from tree_trainer import DualMixer
from transformers import TextStreamer
from tree_trainer import DualMLPMixer, convert_generations_to_tree, tree_backup, test_correctness 
from tree_trainer import get_token_sequences, get_token_values, get_token_batch, get_value_batch, get_token_batch, get_evaluations, assign_leaf_node_values, answer_extract, output_extract
import warnings
import time
import uuid
import pprint
from tqdm import tqdm
from datasets import Dataset
warnings.simplefilter(action='ignore', category=UserWarning)


def prepare_nshot(example, n_shot=2, use_random=True):
    # n shot append and rename fields for rl
    if use_random:
    	offset = random.randint(0, len(train_dataset)-n_shot)
    three_shot_prompt = '\n'.join([f"Question: {train_dataset[i+offset]['question']} \nAnswer: {train_dataset[i+offset]['answer']}" for i in range(n_shot)])
    example['prompt'] = f"{three_shot_prompt}\n Question: {example['question']} \n Answer:"
    example['cleaned_answer'] = answer_extract(example['answer'])
    return example


def generate_values(policy_model,
		train_dataset,
		tokenizer,
		accelerator=None,
		generate_batch=512,
		generate_steps=2000,
		value_constant=10.,
		save_every=200,
		output_path=''
		):

	if accelerator is None:
		ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
		accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision='fp16')

	device = accelerator.device
	policy_model = policy_model.to(device)
	
	policy_model.eval()  # Freeze policy model
	process_index = accelerator.process_index
	num_processes = accelerator.num_processes

	dataset_size = len(train_dataset)
	indices_per_process = dataset_size // num_processes
	start_idx = process_index * indices_per_process
	end_idx = start_idx + indices_per_process if process_index < num_processes - 1 else dataset_size
	process_indices = list(range(start_idx, end_idx))
	print(f"Process {process_index}: Training on indices {start_idx} to {end_idx}")

	total_tokens = []
	total_values = []
	for step in tqdm(range(generate_steps)):
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
			truncation=True,
			padding='max_length', 
			padding_side='left').to(device)
		
		if hasattr(policy_model, 'module'):
			policy_model.module.clear_cache()
		else:
			policy_model.clear_cache()
		
		with torch.no_grad():
			generated_ids = policy_model.generate(
				input_ids.repeat(generate_batch, 1),
				max_new_tokens=tokens_to_generate,
				tokenizer=tokenizer,
				do_sample=True,
				temperature=0.7,
				top_p=0.9,
				pad_token_id=tokenizer.pad_token_id
			).to('cpu')
		accelerator.wait_for_everyone()	
		prompt_length = input_ids.shape[1]
		generated_tokens = generated_ids[:, prompt_length:]

		# clean generation, remove extra questions
		generated_strings = tokenizer.decode(generated_tokens)
		truncated_strings = [i.split('\nQuestion')[0] for i in generated_strings]
		generated_tokens = tokenizer.encode(truncated_strings)
		expanded_question = [question for _ in range(generate_batch)]
		generated_ids = torch.cat([tokenizer.encode(i+j, return_tensors='pt', pad_token_id=tokenizer.eos_token_id, padding_side='right', padding='max_length', max_length=1024, truncation=True) for i, j in zip(expanded_question, truncated_strings)], dim=0)

		# Convert generations to tree structure, back up values, and process
		tree = convert_generations_to_tree(generated_ids) # all ids, not just generated
		completions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True) # generated only
		values = test_correctness(completions, answer, value_constant)
		tree = tree_backup(tree, values)
		token_values = get_token_values(tree)
		token_sequences = get_token_sequences(tree)
		correct_leaves = sum([node['value'] for node in tree.values() if node['is_leaf']]) / value_constant
		value_batch = get_value_batch(tree, token_values)
		token_batch = get_token_batch(tree, token_sequences)
		total_tokens += token_batch.tolist()
		total_values += value_batch.tolist()
		accelerator.wait_for_everyone()

		if step % save_every==0 and step > 0:
			print (len(total_tokens), len(total_values))
			dataset_dict = {'input_ids': total_tokens, 'values': total_values}
			dataset = Dataset.from_dict(dataset_dict)
			dataset.save_to_disk(output_path + f'/{process_index}_{step}')
			total_tokens = []
			total_values= []
	return


if __name__ == "__main__":
	load_dotenv()
	checkpoint_root = os.getenv('CHECKPOINT_ROOT')
	data_root = os.getenv('DATA_ROOT')
	tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k", truncation_side='left')
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = tokenizer.vocab_size
	print("Vocab size: ", n_vocab)

	dataset = load_from_disk(f"{data_root}/gsm8k", "main")
	train_dataset, eval_dataset = dataset['train'], dataset['test']
	print (len(train_dataset))
	train_dataset = train_dataset.map(prepare_nshot, num_proc=16)
	train_dataset = train_dataset.remove_columns('question')
	train_dataset = train_dataset.rename_column('prompt', 'question') # assign n shot
	print (train_dataset[0])
	tokenized_length = 1024
	dim = 1024
	layers = 16
	n_heads = 4
	kernel = 1

	policy_model = DualMixer(
		n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
		mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True, is_reward_model=False).float()

	model_path=f'{checkpoint_root}/gsq_4_mixed_decay_nonparallel_projs_k1_1024_n16_c1024_b16x4/checkpoint-600000/model.safetensors'
	load_model(policy_model, model_path)
	print ('model loaded')
	policy_model = torch.compile(policy_model).to('cuda')

	input_ids = tokenizer.encode('Question: Sally has three crayons and Andy has twice as many crayons as Sally and Harry has twice as many as Andy. How many crayons do they have in total? \nAnswer: ', return_tensors='pt', add_special_tokens=False).repeat(5, 1).to('cuda')
	output = torch.tensor(policy_model.generate(input_ids, max_new_tokens=128, temperature=0.7, top_p=0.9, tokenizer=tokenizer, do_sample=True))
	input_length = len(input_ids[1])
	model_generation = output[:, input_length:]
	input_strings = tokenizer.decode(input_ids)
	print ('\n\n', output, tokenizer.decode(model_generation), '\n\n')
	model_strings = tokenizer.decode(model_generation)
	truncated_generations = [i.split('\nQuestion')[0] for i in model_strings]
	tokens = tokenizer.encode(truncated_generations)
	cleaned_generations = [i+j for i, j in zip(input_strings, truncated_generations)]
	print (cleaned_generations)

	policy_model.clear_cache()
	print (policy_model.index)
	output_path = f'{data_root}/gsm8k_rewards_t512'
	generate_values(policy_model, train_dataset, tokenizer, output_path=output_path)

