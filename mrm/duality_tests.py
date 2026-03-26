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
from inference import InferenceMLPMixer as CachedInferenceMLPMixer

from transformers import TextStreamer
from grpo_trainer import DualMixer
from inference import RecurrentInference
from naive_inference import InferenceMLPMixer
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import pytest

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_recurrent_dual(trained_model=True):
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

    dual_model = DualMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    generation_config = GenerationConfig()
   
    load_model(dual_model, f"{checkpoint_root}/gsm8k_SFT_srm_c1024/checkpoint-300/model.safetensors")
    load_model(cached_model, f"{checkpoint_root}/gsm8k_SFT_srm_c1024/checkpoint-300/model.safetensors")
    text ='''Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? 
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
 Question: Ahmed and Emily are having a contest to see who can get the best grade in the class. There have been 9 assignments and Ahmed has a 91 in the class. Emily has a 92. The final assignment is worth the same amount as all the other assignments. Emily got a 90 on the final assignment. What is the minimum grade Ahmed needs to get to beat Emily if all grades are whole numbers? 
 Answer:'''
    text = 'Four score and seven years ago, '
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token
    print (input_ids)

    output_ids = dual_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    assert torch.equal(output_ids, cached_output_ids)
    return

def test_parallel_dual(trained_model=True):
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

    dual_model = DualMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    model = MLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    generation_config = GenerationConfig()
   
    load_model(dual_model, f"{checkpoint_root}/gsm8k_SFT_srm_c1024/checkpoint-300/model.safetensors")
    load_model(model, f"{checkpoint_root}/gsm8k_SFT_srm_c1024/checkpoint-300/model.safetensors")
    text ='''Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? 
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
 Question: Ahmed and Emily are having a contest to see who can get the best grade in the class. There have been 9 assignments and Ahmed has a 91 in the class. Emily has a 92. The final assignment is worth the same amount as all the other assignments. Emily got a 90 on the final assignment. What is the minimum grade Ahmed needs to get to beat Emily if all grades are whole numbers? 
 Answer:'''
    input_ids = torch.tensor(tokenizer.encode(text, padding='max_length', max_length=tokenized_length)).unsqueeze(0).to(device) # ignore bos token
    print (input_ids.shape)

    dual_output_logits = dual_model.forward(input_ids, labels=input_ids)[1]
    output_logits = model.forward(input_ids, labels=input_ids).logits
    assert torch.allclose(dual_output_logits, output_logits)
    return

def test_recurrent_mixed_row_col_decay_scaling_equivalence():
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

    cached_model = DualMixer(
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

