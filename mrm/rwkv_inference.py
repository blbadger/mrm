import torch
import torch.nn as nn
import transformers
from transformers import RwkvConfig, RwkvModel, RwkvForCausalLM, AutoTokenizer
from transformers.generation import GenerationMixin, GenerationConfig
from prettytable import PrettyTable
from safetensors.torch import save_file, load_model
from safetensors import safe_open
import safetensors
import datasets
from datasets import load_from_disk
import warnings
import shutil
from dotenv import load_dotenv
import os
import mlflow
#warnings.filterwarnings(action='ignore')
import time
import logging
logging.basicConfig(level=logging.INFO)

class RWKVCLM(nn.Module):
    def __init__(self, model, vocab_size=8000, dim=512):
        super().__init__()
        self.model = model # a LlamaModel object
        self.lm_head = nn.Linear(dim, vocab_size)
        self.vocab_size = vocab_size
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None, attention_mask=None):
        #print (trainer.accelerator.scaler._scale)
        #trainer.accelerator.scaler._scale = torch.tensor(100.)
        output = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        labels = labels[:, 1:].contiguous()
        logits = self.lm_head(output)
        shift_logits = logits[:, :-1, :].contiguous() # b t e 
        if labels is not None:
            logits = shift_logits.view(-1, self.vocab_size)
            labels = labels.view(-1)
            loss = self.loss_fn(logits, labels)
            return loss, logits

        else:
            return logits

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

device = 'cuda' if torch.cuda.is_available else 'cpu'

dim = 256
context_length = 512
compression = 1
n_layers = 16
n_heads = 4

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
vocab_size = len(tokenizer)
config_kwargs = {
    'hidden_size':dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': n_heads,
    'vocab_size': vocab_size
}
# Initializing a LLaMA model
configuration = RwkvConfig(**config_kwargs)
model = RwkvModel(configuration)
#model = RWKVCLM(model, dim=dim, vocab_size=vocab_size)
model = RwkvForCausalLM(configuration).to(device)

generation_config = GenerationConfig()
text ='''Four score and seven years ago, our'''
tokens_to_generate = 500
batch_size = 40
input_ids = torch.tensor(tokenizer.encode(text)[1:]).repeat(batch_size, 1 ).to(device) # ignore bos token
print (input_ids.shape)
start = time.time()
output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + tokens_to_generate, generation_config=generation_config) #, streamer=streamer)
print (f'Output example: {tokenizer.decode(output_ids[0])}, Elapsed time: {time.time() - start}, t/s: {(batch_size * tokens_to_generate)/(time.time() - start)}')

