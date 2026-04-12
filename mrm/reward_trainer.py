import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import datasets
from datasets import load_from_disk
from safetensors.torch import load_model, save_model
import mlflow
from prettytable import PrettyTable
import os
from dotenv import load_dotenv
import shutil
from tree_trainer import DualMLPMixer

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
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

	reward_model = DualMixer(
		n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
		mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True, is_reward_model=True).float()

	# initialize with policy model weights (aside from reward head)
	reward_model_path=f'{checkpoint_root}/gsm8k_SFT_srm_c1024/meta-chkpt-300/model.safetensors'
	load_model(reward_model, reward_model_path, strict=False)

	n_gpus = torch.cuda.device_count()
	total_batch_size = 64 #  16x4 for nctx=1024
	batch_size = total_batch_size // n_gpus
	train_path = f"{data_root}/gsm8k_rewards_t512"

	output_dir = f"{checkpoint_root}/gsm8k_reward_model_t512/full_dataset"
  
	datasets.config.IN_MEMORY_MAX_SIZE = 1e9
	train_dataset = load_from_disk(train_path, keep_in_memory=None).skip(10000)
	test_dataset = load_from_disk(test_path, keep_in_memory=None).take(10000)
	print(len(train_dataset), len(test_dataset))
	mlflow.end_run()
	print("training begun")
	print(model)
	print (output_dir)
	training_arguments = transformers.TrainingArguments(
		num_train_epochs=2,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		gradient_accumulation_steps=1,
		warmup_steps=4000,
		eval_steps=4000,
		save_steps=8000,
		learning_rate=2e-4,
		fp16=True,
		eval_strategy="steps",
		output_dir=output_dir,
		optim="adamw_torch",
		overwrite_output_dir=True,
		save_safetensors=True,
		max_steps=200000,
		torch_compile=True,
		label_names=['rewards']
	)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments,
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	)

	# save driver code snapshot in checkpoint dir 
	code_path = os.path.abspath(__file__) 
	if not os.path.isdir(output_dir): 
		os.mkdir(output_dir) 
	shutil.copy(code_path, output_dir) 
	print (trainer.evaluate())
	model.train()
	trainer.train()
