import os
from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets


output_dir = ''
parent_dir = ''

def concatente(parent_dir):
    data_dirs = os.listdir(parent_dir)
    ds = concatenate_datasets([load_from_disk(parent_dir + '/' + dir) for dir in data_dirs])
    return ds

def tokenization(example):
    tokenizer = AutoTokenizer.from_pretrained(f'{checkpoint_root}/fineweb_tokenizer_8k')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    padding_side='left'
    context_length=1024
    # global padding_side, context_length
    n_ctx = context_length
    tokens = tokenizer.encode(
                    example['samples'],
                    add_special_tokens=False,
                    return_tensors='pt',
                    truncation=True,
                    padding='max_length',
                    padding_side=padding_side,
                    max_length=n_ctx
            )
    example['input_ids'] = tokens
    return example

ds = concatenate_dataset(parent_dir)
ds = ds.map(tokenization,  num_proc=64)
ds.save_to_disk(output_dir)