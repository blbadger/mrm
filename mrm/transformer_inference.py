import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import  LlamaModel, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.generation import GenerationMixin, GenerationConfig
import datasets
from datasets import load_from_disk
from safetensors.torch import load_model
import os
from dotenv import load_dotenv
import shutil
from repeat_main import MLPMixer
import time
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = tokenizer.vocab_size

    tokenized_length = 2048
    dim = 512
    layers = 16
    n_heads = 8

    llama_config_kwargs = {
        'hidden_size': dim,
        'intermediate_size': 4*dim,
        'num_hidden_layers': layers,
        'num_attention_heads': n_heads,
        'vocab_size': n_vocab
    }

    config = LlamaConfig(**llama_config_kwargs)
    model = LlamaForCausalLM(config).float().to(device)
    print (model)
    #load_model(model, f"{checkpoint_root}/fineweb_training/fineweb_llama_512_c1024/checkpoint-196000/model.safetensors")
    generation_config = GenerationConfig()
    text ='''Four score and seven years ago, our'''
    tokens_to_generate = 2000
    batch_size = 50
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).repeat(batch_size, 1 ).to(device) # ignore bos token
    print (input_ids.shape)
    start = time.time()
    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + tokens_to_generate, generation_config=generation_config) #, streamer=streamer)
    print (f'Output example: {tokenizer.decode(output_ids[0])}, Elapsed time: {time.time() - start}, t/s: {(batch_size * tokens_to_generate)/(time.time() - start)}')
