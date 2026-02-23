import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation import GenerationMixin, GenerationConfig
import datasets
from datasets import load_from_disk
from safetensors.torch import load_model
import os
from dotenv import load_dotenv
import shutil
from repeat_main import MLPMixer
from cached_inference import CachedMLPMixer
from recurrent_inference import RecurrentMLPMixer
from transformers import TextStreamer
import warnings
import time
warnings.simplefilter(action='ignore', category=UserWarning)

class InferenceMLPMixer(CachedMLPMixer, GenerationMixin):

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
		dropout_layer=False
	):
		super().__init__(vocab_size, hidden_dim, seq_len, num_blocks, heads=heads, kernel=kernel, expanded_convs=expanded_convs, copy=copy, 
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
		self.main_input_name = 'input_ids'
		self._supports_cache_class = False
		self.cache_built = False
		self.device = self.output_layer.weight.device
		if dropout_layer:
			# overwrite original dropout layer with dropout included
			for i in range(len(self.mixer_blocks)):
				self.mixer_blocks[i].channel_mixing_layer = nn.Sequential(
				nn.Linear(hidden_dim, hidden_dim * self.mixer_blocks[i].expansion_factor),
				nn.SiLU(),
				nn.Dropout(0.),
				nn.Linear(hidden_dim * self.mixer_blocks[i].expansion_factor, hidden_dim),
			)

	def can_generate(self):
		return True

	def count_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def build_cache(self, input_ids):
		for i in range(1, len(input_ids[0])):
			x = self.input_layer(input_ids[:, :i])
			for block in self.mixer_blocks:
				x = block(x)
		self.cache_built = True
		return	

	def forward(self, input_ids, labels=None, **kwargs):
		if not self.cache_built:
			self.build_cache(input_ids)
		if labels is not None:
			labels = labels[:, 1:].contiguous()
		# model's forward pass
		x = self.input_layer(input_ids)
		for block in self.mixer_blocks:
			x = block(x)
		logits = self.output_layer(x)
		logits = logits[:, -1].unsqueeze(1).contiguous()
		if labels is not None:
			return CausalLMOutput(loss=0, logits=logits)
		else:
			return CausalLMOutput(loss=0, logits=logits)


class RecurrentInference(RecurrentMLPMixer, GenerationMixin):

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
		dropout_layer=False
	):
		super().__init__(vocab_size, hidden_dim, seq_len, num_blocks, heads=heads, kernel=kernel, expanded_convs=expanded_convs, copy=copy, 
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
		self.main_input_name = 'input_ids'
		self._supports_cache_class = False
		self.cache_built = False
		self.device = self.output_layer.weight.device
		if dropout_layer:
			# overwrite original dropout layer with dropout included
			for i in range(len(self.mixer_blocks)):
				self.mixer_blocks[i].channel_mixing_layer = nn.Sequential(
				nn.Linear(hidden_dim, hidden_dim * self.mixer_blocks[i].expansion_factor),
				nn.SiLU(),
				nn.Dropout(0.),
				nn.Linear(hidden_dim * self.mixer_blocks[i].expansion_factor, hidden_dim),
			)

	def can_generate(self):
		return True

	def count_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def build_cache(self, input_ids):
		for i in range(len(input_ids[0])-1):
			x = self.input_layer(input_ids[:, i])
			for block in self.mixer_blocks:
				x = block(x, i)
		self.cache_built = True
		return

	def forward(self, input_ids, labels=None, **kwargs):
		if not self.cache_built:
			self.build_cache(input_ids)
		index = input_ids.shape[1] - 1
		input_ids = input_ids[:, -1] # last token only
		
		# model's forward pass
		x = self.input_layer(input_ids)
		for block in self.mixer_blocks:
			x = block(x, index)
		logits = self.output_layer(x).unsqueeze(1)
		if labels is not None:
			return CausalLMOutput(loss=0, logits=logits)
		else:
			return CausalLMOutput(loss=0, logits=logits)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = tokenizer.vocab_size
    print("Vocab size: ", n_vocab)

    tokenized_length = 1024
    dim = 1024
    layers = 16
    n_heads = 4
    kernel = 1

    model = RecurrentInference(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)
    generation_config = GenerationConfig()
    print (model)
    load_model(model, f"{checkpoint_root}/fineweb_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors")
    model = torch.compile(model)
    text ='''Four score and seven years ago, our'''
    batch_size = 64000
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).repeat(batch_size, 1).to(device) # ignore bos token
    print (input_ids.shape)
    tokens_to_generate = 1000
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    start = time.time()
    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + tokens_to_generate, generation_config=generation_config) #, streamer=streamer)
    print (f'Example: {tokenizer.decode(output_ids[0])}, elapsed time: {time.time() - start}, t/s: {(tokens_to_generate * batch_size)/(time.time() - start)}')
