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
import re
from dotenv import load_dotenv
import shutil
from repeat_main import MLPMixer
from cached_inference import CachedMLPMixer
from recurrent_inference import RecurrentMLPMixer
from transformers import TextStreamer
from dual_srm import DualMLPMixer
import warnings
import time
import uuid
warnings.simplefilter(action='ignore', category=UserWarning)


class DualMixer(DualMLPMixer, GenerationMixin):

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        seq_len: int,
        num_blocks: int,
        heads=None,
        kernel=1,
        expanded_convs=False,
        copy=False,        
        mixed_heads=False,
        combined_heads=False,
        decay=False,
        parallel_heads=False,
        use_projections=True,
        dropout_layer=False,
        **kwargs
    ):
        super().__init__(vocab_size, hidden_dim, seq_len, num_blocks, heads=heads, kernel=kernel, expanded_convs=expanded_convs,
            mixed_heads=mixed_heads, combined_heads=combined_heads, decay=decay, parallel_heads=parallel_heads, use_projections=use_projections)
        self._init_weights()
        self.generation_config = GenerationConfig()
        config  = {
                 'hidden_size':hidden_dim,
                 'intermediate_size': 4*hidden_dim,
                 'num_hidden_layers': num_blocks,
                 'num_attention_heads': 4,
                 'vocab_size': vocab_size
                }
        self.config = LlamaConfig(**config)
        self.hidden_dim = hidden_dim
        self.n_heads = heads
        self.seq_len = seq_len
        self.main_input_name = 'input_ids'
        self._supports_cache_class = False
        self.cache_built = False
        self.device = self.output_layer.weight.device
        self.warnings_issued={}
    
    def add_model_tags(self, tag):
        print (tag)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        pass

    def can_generate(self):
        return True

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _is_stateful(self):
        return False

    def get_cache(self):
        cache_store = {}
    	for block in self.mixer_blocks:
            for h in range(len(block.token_mixing_layer.mixer_heads)):
                cache_store[f'{block},{h}'] = block.token_mixing_layer.mixer_heads[h].cache
        return cache_store

    def load_cache(self, cache_store):
    	for block in self.mixer_blocks:
            for h in range(len(block.token_mixing_layer.mixer_heads)):
                block.token_mixing_layer.mixer_heads[h].cache = cache_store[f'{block},{h}']
        return

    def build_cache(self, input_ids):
        for i in range(len(input_ids[0])-1):
            x = self.input_layer(input_ids[:, i])
            for block in self.mixer_blocks:
                x = block(x, i, True)
        self.cache_built = True
        return

    def clear_cache(self):
        for block in self.mixer_blocks:
            for h in range(len(block.token_mixing_layer.mixer_heads)):
                block.token_mixing_layer.mixer_heads[h].cache = torch.zeros(self.hidden_dim//self.n_heads).to('cuda') # only for mixed heads
        self.cache_built = False

    def forward(self, input_ids, labels=None, **kwargs):
        is_recurrent = input_ids.shape[1] < self.seq_len
        if not self.cache_built and is_recurrent:
            self.build_cache(input_ids)
        index = input_ids.shape[1] - 1
        if is_recurrent:
            input_ids = input_ids[:, -1] # last token only
        # model's forward pass
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x, index, is_recurrent)
        logits = self.output_layer(x).unsqueeze(1)
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = self.loss_fn(shift_logits, shift_labels)
            return CausalLMOutput(loss=loss, logits=logits)
        else:
            return CausalLMOutput(loss=0, logits=logits)

# example node data structure
# {'token': 143, 'value': 0.23, 'parent': id_number, 
#  'children': [id_number...], 'is_leaf': True, 'cache_store': {['str']:torch.Tensor}}

# example cache_store data structure:
# {'mixer_blocks[0],1': torch.tensor([0.1, 0.2, 0.3]), 'mixer_blocks[0],2': torch.tensor([0.4, 0.5, 0.6])}

# example tree data structure
# {id_number: {'token': 143, 'value': 0.23, 'parent': id_number, ...}, 
# id_number_2: {'token': ...}}

def predict_next_nodes(policy_model, node, n_taken=2):
    policy_model.load_cache(node.cache_store)
    _, logits = policy_model.forward(node.token)
    sorted_logits, indices = torch.topk(logits, n_taken)
    top_indices = indices[:n_taken]
    for index in top_indices:
        # make new child node
        new_id = uuid.uuid4()
        node.children.append(new_id)
        new_node = {new_id: {'token': index, 'value': None, 'is_leaf': True, 'parent': node.key, 'cache_store':policy_model.get_cache(), 'children': []}}
    return top_indices

def tree_backup(tree):
    # Algorithm: find leaves, back up each leaf value to root, accumulate
    # expects leaves to have values
    for node in tree:
        if node.is_leaf:
            leaf_value = node.value
            # back up value
            while node.parent:
                tree[node.parent].append(leaf_value)
                node = node.parent
    for node in tree:
        if node.value = sum(node.value) / len(node.value)
    
    return tree

def test_correctness(completions, answer, **kwargs) -> list[float]:
    extracted_responses = [output_extract(c) for c in completions] 
    # expects answers to be pre-extracted to save time
    #print('='*60, f"Question:\n{prompts[1]}", f"\nAnswer:\n{answer[1]}\n",'-'*50, f"\nResponse:\n{completions[1]}", f"\nExtracted:\n{extracted_responses[1]}\n")
    values = [1.0 if r == answer else 0.0 for r in extracted_responses]
    return values
    
def get_token_sequences(tree):
    outputs = {} # maps node_id to output
    for key, node in tree.items():
        if node['is_leaf']:
            output = []
            while node.parent:
                token = node['token']
                output.append(token)
                node = node.parent
            in_order_tokens = output.reverse()
            outputs[key] = in_order_tokens
    return outputs

def get_token_values(tree):
    outputs = {} # maps node_id to output
    for key, node in tree.items():
        if node['is_leaf']:
            output = []
            while node.parent:
                value = node['value'] # assumes that values have already been mean accumulated
                output.append(value)
                node = node.parent
            in_order_values = output.reverse()
            outputs[key] = in_order_values
    return outputs

def get_evaluations(outputs, answer):
    # expects a single answer, ie all outputs are for the same question
    # batch decode tokens
    output_tokens = [key, value for key, value in outputs.items()]
    output_keys = [o.keys for o in outputs]
    detokenized_outputs = tokenizer.decode(output_tokens)
    extracted_answer = answer_extract(answer)
    extracted_outputs = [output_extract(o) for o in detokenized_outputs]
    values = test_correctness(extracted_outputs, extracted_answer)
    # zip back up into outputs
    for key, tokens, string, extracted_outputs, value in zip(output_keys, output_tokens, detokenized_outputs, values):
        outputs[key] = {'tokens': tokens, 'string': string, 'value': value}
    return outputs

def assign_leaf_node_values(tree, answer):
    output_dict = get_token_sequences(tree) # maps node_id to output tokens
    evaluation_dict = get_evaluations(output_dict, answer)
    for key, val in evaluation_dict.items():
        tree[key]['value'] = val['value']
    return tree

def answer_extract(answer):
    cleaned_output = answer.split('####')
    if len(cleaned_output) > 1:
        cleaned_output = cleaned_output[1].strip(' ,!@#$%^&*')
    return cleaned_output

def output_extract(predicted_output):
    output = re.findall("(-?[$0-9.,]{2,})|(-?[0-9]+)", predicted_output)
    if output:
        output = output[-1]
    outs = []
    for i, out in enumerate(output):
        if isinstance(out, tuple) and len(out) > 1: 
           outs.append((out[0] if out[0] else out[1]).strip(' %$@!*,.'))
        else:
           outs.append(out)
    return outs

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    policy_model = DualMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    load_model(policy_model, f"{checkpoint_root}/finemath_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors")
    reward_model = DualMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).float().to(device)

    policy_model = torch.compile(policy_model)
    reward_model = torch.compile(reward_model)
    text ='''Four score and seven years ago, our forefathers, for the purpose of a more perfect union, sought'''
    batch_size = 500
    input_ids = torch.tensor(tokenizer.encode(text)[1:]).repeat(batch_size, 1).to(device) # ignore bos token
    print (input_ids.shape)
    tokens_to_generate = 50
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    start = time.time()
    print (f'Example: {tokenizer.decode(output_ids[0])}, elapsed time: {time.time() - start}, t/s: {(tokens_to_generate * batch_size)/(time.time() - start)}')
