import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
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

if __name__ == "__main__":
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = tokenizer.vocab_size

    tokenized_length = 512
    dim = 512
    layers = 16
    n_heads = 8

    llama_config_kwargs = {
        'hidden_size': dim,
        'intermediate_size': 4*dim,
        'num_hidden_layers': depth,
        'num_attention_heads': n_heads,
        'vocab_size': n_vocab
    }

    config = {**config_kwargs}
    model = LlamaForCausalLM(config)
    load_model(model, f"{checkpoint_root}/fineweb_training/fineweb_llama_512_n16_h8_c512/checkpoint-200000/model.safetensors")
    generation_config = GenerationConfig()
    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).repeat(10, 1 ).to(device) # ignore bos token
    print (input_ids.shape)
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    start = time.time()
    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config) #, streamer=streamer)
    print (tokenizer.decode(output_ids[0]), time.time() - start)
