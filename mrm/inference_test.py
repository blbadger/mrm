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
from compat_repeat_test import RepeatCausalLinear, RepeatHeads, MixerBlock

class MLPMixer(nn.Module, GenerationMixin):

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
	):

		super(MLPMixer, self).__init__()

		self.vocab_size = vocab_size
		self.hidden_dim = hidden_dim
		self.seq_len = seq_len
		self.num_blocks = num_blocks
		self.input_layer = nn.Embedding(vocab_size, hidden_dim)

		self.mixer_blocks = nn.ModuleList(
			[
				MixerBlock(
					hidden_dim, seq_len, heads=heads, expanded_convs=expanded_convs, kernel=kernel
				)
				for _ in range(num_blocks)
			]
		)
		self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False, device='cuda')

		self._init_weights()
		self.loss_fn = nn.CrossEntropyLoss()

		self.generation_config = GenerationConfig()
		config  = {
				 'hidden_size':hidden_dim,
				 'intermediate_size': 4*hidden_dim,
				 'num_hidden_layers': layers,
				 'num_attention_heads': 4,
				 'vocab_size': vocab_size
			 }
		self.config = LlamaConfig(**config)
		self.main_input_name = 'input_ids'
		self._supports_cache_class = False
		self.device = self.output_layer.weight.device

	def can_generate(self):
		return True

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, RepeatCausalLinear): 
				# Kaiming He initialization for Swish activation
				nn.init.kaiming_normal_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	def count_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def forward(self, input_ids, labels=None, **kwargs):
		# pad input_ids
		pad_sizes = [self.seq_len - len(input_ids[i]) for i in range(len(input_ids))]
		input_lengths = [len(input_ids[i]) for i in range(len(input_ids))]

		padded_inputs = []
		for i in range(len(input_ids)):
			input_id_arr = input_ids[i].unsqueeze(0)
			pad = torch.ones(pad_sizes[i], dtype=torch.long).to(input_ids.device).unsqueeze(0)
			padded_input = torch.cat((input_id_arr, pad), dim=1)
			padded_inputs.append(padded_input)
		input_ids = torch.cat(padded_inputs, dim=0)
		labels = torch.where(input_ids==1, -100, input_ids) #mask pad token loss

		if labels is not None:
			labels = labels[:, 1:].contiguous()

		# model's forward pass
		x = self.input_layer(input_ids)
		for block in self.mixer_blocks:
			x = block(x)
		logits = self.output_layer(x)
		logits = logits[:, :-1].contiguous()

		truncated_logits = []
		for i in range(logits.shape[0]):
			truncated_logits.append(logits[i, :input_lengths[i]])
		truncated_logits = torch.stack(truncated_logits, dim=0)

		if labels is not None:
			logits = logits.view(-1, self.vocab_size)
			labels = labels.view(-1)

			loss = self.loss_fn(logits, labels)
			print (loss)
			return CausalLMOutput(loss=loss, logits=truncated_logits)
		else:
			return CausalLMOutput(loss=0, logits=truncated_logits)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = tokenizer.vocab_size
    print("Vocab size: ", n_vocab)

    tokenized_length = 512
    dim = 1024
    layers = 16
    n_heads = None
    kernel = 1

    model = MLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False
    ).float().to(device)

    generation_config = GenerationConfig(greedy=True)

    print (model)
    load_model(model, f"{checkpoint_root}/fineweb_hNone_colrepeat_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    text = '''The color of the clear sky during daytime is usually'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token
    print (input_ids)
    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    print (tokenizer.decode(output_ids[0]))
