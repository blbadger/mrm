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
from transformers import TextStreamer
from inference import InferenceMLPMixer as CachedInferenceMLPMixer
from naive_inference import InferenceMLPMixer
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import pytest

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_row_repeat_equivalence(trained_model=True):
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = tokenizer.vocab_size
    print("Vocab size: ", n_vocab)

    tokenized_length = 512
    dim = 512
    layers = 16
    n_heads = None
    kernel = 1

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=False, combined_heads=False, decay=True, parallel_heads=False, use_projections=True, dropout_layer=True).float().to(device)

    model = InferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=False, combined_heads=False, decay=True, parallel_heads=False, use_projections=True, dropout_layer=True).float().to(device)

    generation_config = GenerationConfig()
    print (model)
    load_model(model, f"{checkpoint_root}/fineweb_hNone_repeat_k1_512_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    load_model(cached_model, f"{checkpoint_root}/fineweb_hNone_repeat_k1_512_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token
    print (input_ids)

    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    assert torch.equal(output_ids, cached_output_ids)
    return

def test_col_repeat_equivalence(trained_model=True):
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
    n_heads = 4
    kernel = 1

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=False, combined_heads=False, decay=False, parallel_heads=False, use_projections=True, dropout_layer=True).float().to(device)

    model = InferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=False, combined_heads=False, decay=False, parallel_heads=False, use_projections=True, dropout_layer=True).float().to(device)

    generation_config = GenerationConfig()
    print (model)
    load_model(model, f"{checkpoint_root}/fineweb_h4_colrepeat_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    load_model(cached_model, f"{checkpoint_root}/fineweb_h4_colrepeat_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token
    print (input_ids)

    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    assert torch.equal(output_ids, cached_output_ids)
    return

def test_mixed_row_col_equivalence(trained_model=False):
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
    n_heads = 4
    kernel = 1

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=False, parallel_heads=False, use_projections=True).float().to(device)

    model = InferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=False, parallel_heads=False, use_projections=True).float().to(device)

    model.load_state_dict(cached_model.state_dict())
    generation_config = GenerationConfig()
    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token

    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    assert torch.equal(output_ids, cached_output_ids)
    return


def test_col_decay_equivalence(trained_model=False):
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
    n_heads = 4
    kernel = 1

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=False, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    model = InferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=False, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    model.load_state_dict(cached_model.state_dict())
    generation_config = GenerationConfig()
    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token

    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    assert torch.equal(output_ids, cached_output_ids)
    return

def test_row_decay_equivalence(trained_model=False):
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

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=False, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    model = InferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=False, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    model.load_state_dict(cached_model.state_dict())
    generation_config = GenerationConfig()
    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token

    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    assert torch.equal(output_ids, cached_output_ids)
    return

def test_mixed_row_col_decay_equivalence(trained_model=False):
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
    n_heads = 4
    kernel = 1

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    model = InferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    #model.load_state_dict(cached_model.state_dict())
    generation_config = GenerationConfig()
    print (model)
    load_model(model, f"{checkpoint_root}/fineweb_h4_decay_mixedrepeat_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    load_model(cached_model, f"{checkpoint_root}/fineweb_h4_decay_mixedrepeat_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors")

    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token

    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    assert torch.equal(output_ids, cached_output_ids)
    return

def test_mixed_row_col_decay_scaling_equivalence():
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

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    model = InferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    generation_config = GenerationConfig()
    print (model)
    load_model(model, f"{checkpoint_root}/fineweb_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors")
    load_model(cached_model, f"{checkpoint_root}/fineweb_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors")
    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token
    print (input_ids)

    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    assert torch.equal(output_ids, cached_output_ids)
    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    return

def test_parallel_mixed_row_col_decay_equivalence():
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
    n_heads = 4
    kernel = 1

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=True, use_projections=True).float().to(device)

    model = InferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=True, use_projections=True).float().to(device)

    generation_config = GenerationConfig()
    print (model)
    load_model(model, f"{checkpoint_root}/fineweb_h4_decay_parallel_mixed_projs_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    load_model(cached_model, f"{checkpoint_root}/fineweb_h4_decay_parallel_mixed_projs_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    assert model.state_dict() == cached_model.state_dict()

    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token
    print (input_ids)

    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    assert torch.equal(output_ids, cached_output_ids)

    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    return
    

if __name__ == '__main__':
	row_repeat_equivalence()
