import torch
import re
from trl import GRPOConfig, GRPOTrainer
from safetensors.torch import load_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset, load_from_disk, Dataset
from dual_srm import DualMLPMixer
from dotenv import load_dotenv
import os
from transformers.generation import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutput
import shutil

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
        self.index = 0
    
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
    
    def is_remote_code(self):
        return True

    def build_cache(self, input_ids):
        for i in range(len(input_ids[0])-1):
            x = self.input_layer(input_ids[:, i])
            for block in self.mixer_blocks:
                x = block(x, i, True)
        self.cache_built = True
        self.index = i+1
        return

    def clear_cache(self):
        for block in self.mixer_blocks:
            for h in range(len(block.token_mixing_layer.mixer_heads)):
                block.token_mixing_layer.mixer_heads[h].cache = torch.zeros(self.hidden_dim//self.n_heads).to('cuda') # only for mixed heads
        self.cache_built = False
        self.index = 0

    def forward(self, input_ids, labels=None, **kwargs):
        is_recurrent = input_ids.shape[1] < self.seq_len
        if not self.cache_built and is_recurrent:
            self.build_cache(input_ids)
        index = self.index
        if is_recurrent:
            input_ids = input_ids[:, -1] # last token only
        # model's forward pass
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x, index, is_recurrent)
        logits = self.output_layer(x).unsqueeze(1)
        self.index += 1
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = self.loss_fn(shift_logits, shift_labels)

            return CausalLMOutput(loss=loss, logits=logits)
        else:
            return CausalLMOutput(loss=0, logits=logits)

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

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    extracted_responses = [output_extract(c) for c in completions]
    extracted_answers = [output_extract(a) for a in answer]
    #print (completions[0], answer[0], extracted_responses[0], extracted_answers[0])
    rewards = [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, extracted_answers)]

#    print('='*60, f"Question:\n{prompts[index]}", f"\nAnswer:\n{answer[index]}\n",'-'*50, f"\nResponse:\n{completions[index]}", f"\nExtracted:\n{extracted_responses[index]}\n")
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, extracted_answers)]

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"####"
    matches = [pattern in c for c in completions]
    return [0.5 if match else 0.0 for match in matches]

def length_reward(completions, **kwargs):
    """For testing a trivial reward maximization"""
    responses = completions 
    return [0.001*len(response) for response in responses]

def prepare_nshot(example, n_shot=3):
    # n shot append and rename fields for rl
    three_shot_prompt = '\n'.join([f"Question: {train_dataset[i]['question']} \nAnswer: {train_dataset[i]['answer']}" for i in range(n_shot)])
    example['prompt'] = f"{three_shot_prompt}\n Question: {example['question']} \n Answer:"
    example['cleaned_answer'] = output_extract(example['answer'])
    return example

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    print("Vocab size: ", n_vocab)
	
    tokenized_length = 1024
    dim = 1024
    layers = 16
    n_heads = 4
    kernel = 1

    model = DualMixer(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True)
    model.is_gradient_checkpointing = False
    #print (model)

    model = LlamaForCausalLM.from_pretrained(f'{checkpoint_root}/gsm8k_SFT_transformer_c1024/checkpoint-300')
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset, eval_dataset = dataset['train'], dataset['test']
    print (train_dataset[0])
    train_dataset = train_dataset.map(prepare_nshot, num_proc=16)
    print (train_dataset[0])
    eval_dataset = eval_dataset.map(prepare_nshot, num_proc=16)
    print (len(train_dataset))
    #model_path=f'{checkpoint_root}/fineweb_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors'
    #model_path=f'{checkpoint_root}/gsm8k_SFT_srm_c1024/meta-chkpt-300/model.safetensors'
    #load_model(model, model_path)
    model = model.to('cuda')
    input_ids = tokenizer.encode('Q: What is two plus two? A: Four. Q: What is one plus four? A:', return_tensors='pt', add_special_tokens=False).to('cuda')
    output = torch.tensor(model.generate(input_ids, max_new_tokens=16, temperature=0., do_sample=False))
    print ('\n\n', output, tokenizer.decode(output[0]), '\n\n')
    max_prompt_length = tokenized_length - 256

    output_dir = f'{checkpoint_root}/gsm8k_transformer_s10_b15x4'
    training_args = GRPOConfig(
        learning_rate = 2e-5,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_torch",
        logging_steps = 1,
        per_device_train_batch_size=15,
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        num_generations = 10, 
        max_completion_length = tokenized_length - max_prompt_length,
        num_train_epochs = 3,
        save_steps = 100,
        max_grad_norm = 0.1,
        report_to = "none",
        output_dir = output_dir,
        fp16=True,
        beta=0.04,
        #torch_compile=True, 
        temperature = 0.7, 
)
	
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )
    # save driver code snapshot in checkpoint dir 
    code_path = os.path.abspath(__file__) 
    if not os.path.isdir(output_dir): 
        os.mkdir(output_dir) 
    shutil.copy(code_path, output_dir) 
    #training_args.save_json(output_dir + '/checkpoint-1250')
    trainer.train()
