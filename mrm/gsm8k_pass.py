from inference import RecurrentInference
from datasets import load_dataset
from safetensors.torch import load_model
from transformers import AutoTokenizer, GenerationConfig, TextStreamer
from dotenv import load_dotenv
import shutil
import torch
import os
import time

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
    generation_config = GenerationConfig(do_sample=True, top_p=10)
    print (model)
    load_model(model, f"{checkpoint_root}/fineweb_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors")
    model = torch.compile(model)

    dataset = load_dataset("openai/gsm8k", "main")

    n_fewshot = 3
    fewshot_examples = '\n'.join([f"Question: {dataset['train'][i]['question']}\nAnswer: {dataset['train'][i]['answer']}" for i in range(n_fewshot)])
    example = dataset['test'][0]['question']
    full_input = f'{fewshot_examples}\nQuestion: {example}\nAnswer: '
    print (example)
    batch_size = 1000
    input_ids = torch.tensor(tokenizer.encode(full_input)).repeat(batch_size, 1).to(device) # ignore bos token
    print (input_ids.shape)
    tokens_to_generate = 100
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    start = time.time()
    output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + tokens_to_generate, generation_config=generation_config) #, streamer=streamer)
    print (f'Example: {tokenizer.decode(output_ids[0])}, elapsed time: {time.time() - start}, t/s: {(tokens_to_generate * batch_size)/(time.time() - start)}')
