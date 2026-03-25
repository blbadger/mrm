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
from grpo_trainer import DualMixer
from inference import RecurrentInference
from naive_inference import InferenceMLPMixer
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import pytest

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_duality_equivalence(trained_model=True):
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

    dual_model = DualMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    cached_model = CachedInferenceMLPMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    generation_config = GenerationConfig()
    print (model)
    path = 
    load_model(model, f"{checkpoint_root}/fineweb_h4_decay_mixedrepeat_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    load_model(cached_model, f"{checkpoint_root}/fineweb_h4_decay_mixedrepeat_k1_1024_n16_c512_b32x4/checkpoint-200000/model.safetensors")
    text ='''Four score and seven years ago, our'''
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).unsqueeze(0).to(device) # ignore bos token
    print (input_ids)

    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    cached_output_ids = cached_model.generate(input_ids, max_length=len(input_ids[0]) + 50, generation_config=generation_config)
    print (f"Reference output: {tokenizer.decode(output_ids[0])} \n Cached Output: {tokenizer.decode(cached_output_ids[0])}")
    assert torch.equal(output_ids, cached_output_ids)
    return
