import torch
import re
from trl import GRPOConfig, GRPOTrainer
from safetensors.torch import load_model
from transformers import AutoTokenizer, LlamaConfig
from datasets import load_dataset, load_from_disk, Dataset
from dual_srm import DualMLPMixer
from dotenv import load_dotenv
import os
from transformers.generation import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutput

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
def correctness_filter(prompts, completions, answer, max_chosen=64, **kwargs) -> list[float]:
    extracted_responses = [output_extract(c) for c in completions]
    extracted_answers = [answer_extract(a) for a in answer]
    good_outputs = []
    for r, a in zip(extracted_responses, extracted_answers):
        if r == a:
            good_outputs.append([r, a])
    return good_outputs[:max_chosen]

def gen_and_filter_inputs(prompts, answer, **kwargs):
    generation_config = {'do_sample': True, 'top_p':0.9, 'temperature':0.7}
    completions = model.generate(inputs, generation_config=generation_config)
    good_pairs = correctness_filter(prompts, completions, answer)
    goot_prompts, good_completions = zip(*good_pairs)[0], zip(*good_pairs)[1]
    return good_prompts, good_completions

def prepare_nshot(example, n_shot=3):
    three_shot_prompt = '\n'.join([f"Question: {train_dataset[i]['question']} \nAnswer: {train_dataset[i]['answer']}" for i in range(n_shot)])
    example['prompt'] = f"{three_shot_prompt}\n Question: {example['question']} \nAnswer :"
    example['completion'] = example['answer']
    example['text'] = example['prompt'] +'|' + example['completion']
    return example

def formatting_prompts_func(example):
    output_text = example['prompt'] + example['completion']
    return output_text

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

    model = SFTModel(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True).to(device)

    print (model)
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset, eval_dataset = dataset['train'], dataset['test'] # positive control
    train_dataset = train_dataset.map(prepare_nshot, num_proc=16).remove_columns(['prompt', 'completion'])
    eval_dataset = eval_dataset.map(prepare_nshot, num_proc=16).remove_columns(['prompt', 'completion'])
    
    print (len(train_dataset))
    model_path=f'{checkpoint_root}/fineweb_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors'
    #model_path=f'{checkpoint_root}/finemath_srm_h4_mixed_decay_nonparallel_projs_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors'
    load_model(model, model_path)
    print ('pretrained model loaded')
    response_template = '|'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, pad_to_multiple_of=1024)

    output_dir = f'{checkpoint_root}/gsm8k_SFT_srm_c1024'
    training_args = SFTConfig(
        learning_rate = 1e-4,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_torch",
        logging_steps = 25,
        per_device_train_batch_size=16,
        max_seq_length = tokenized_length,
        num_train_epochs = 500,
        save_steps = 100,
        eval_steps = 50,
        eval_strategy = 'steps',
        max_grad_norm = 0.1,
        report_to = "none",
        output_dir = output_dir,
        fp16=True,
        torch_compile=True
    )

    config =training_args
    trainer = SFTTrainer(
            model=model,
            args=config,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            formatting_func=gen_and_filter_inputs
        )
    
    model.train()
    trainer.train()