import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from transformers import MambaConfig, Mamba2Config, MambaForCausalLM, Mamba2ForCausalLM, Mamba2Model
import time

dim = 256
context_length = 512
n_layers = 8
state_size = dim//2
num_heads = 8
head_dim = dim//4
tokenizer = AutoTokenizer.from_pretrained(f"/home/azureuser/tokenizer_fineweb_8k")
vocab_size = len(tokenizer)

config_kwargs = { 
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': num_heads,
    'vocab_size': vocab_size,
    'state_size': state_size,
    'hidden_dropout_prob': 0,
    'pad_token_id': tokenizer.pad_token_id,
    'eos_token_id': tokenizer.eos_token_id,
    'chunk_size': context_length,
    'num_heads': num_heads,
    'head_dim': head_dim
}

config = Mamba2Config(**config_kwargs)
model = Mamba2ForCausalLM(config).to('cuda')

batch_size = 1000
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer("Let's count to one thousand! One, two, three, four, five, six, seven, eight, nine, ten, eleven,", return_tensors="pt").input_ids.repeat(batch_size, 1).to(model.device)
max_tokens = 500
start = time.time()
output = model.generate(input_ids, max_new_tokens=max_tokens)
end = time.time()
total_tokens = max_tokens * batch_size
print (f"Throughput (t/s): {total_tokens / (end - start)}")

