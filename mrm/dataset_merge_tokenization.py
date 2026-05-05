import os
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, concatenate_datasets
from dotenv import load_dotenv
load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')

parent_dir = '/home/bbadger/Desktop/gsm8k_rewards_t512'

def load_and_concatente(parent_dir):
    data_dirs = os.listdir(parent_dir)
    ds = concatenate_datasets([load_from_disk(parent_dir + '/' + dir) for dir in data_dirs])
    return ds

def tokenization(example):
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

tokenizer = AutoTokenizer.from_pretrained(f'{checkpoint_root}/tokenizer_fineweb_8k')
tokenizer.pad_token_id = tokenizer.eos_token_id

# concatenate dataset
ds = load_and_concatenate_dataset(parent_dir)

# tokenize if necessary
# ds = load_dataset('blbadger/gsq')
# ds = ds.map(tokenization,  num_proc=48)

# save dataset to disk
output_dir = '/home/bbadger/Desktop/gsm8k_rewards_t512_dataset'
ds.save_to_disk(output_dir)
