import torch
import re
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from safetensors.torch import load_model
from transformers import AutoTokenizer, LlamaConfig
from datasets import load_dataset, load_from_disk, Dataset
from repeat_main import MLPMixer
from dotenv import load_dotenv
import os
from transformers.generation import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutput


class SFTModel(MLPMixer):

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
        **kwargs):

        super().__init__(vocab_size, hidden_dim, seq_len, num_blocks, heads=heads, kernel=kernel, expanded_convs=expanded_convs,
            mixed_heads=mixed_heads, combined_heads=combined_heads, decay=decay, parallel_heads=parallel_heads, use_projections=use_projections)

    def forward(self, input_ids, labels=None, **kwargs):
        # forward pass
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x)
        logits = self.output_layer(x)

        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            shift_logits = logits[:, :-1].contiguous().view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = self.loss_fn(shift_logits, shift_labels)
            return CausalLMOutput(loss=loss, logits=logits)
        else:
            return CausalLMOutput(loss=0., logits=logits)


def prepare_nshot(example, n_shot=3):
    three_shot_prompt = '\n'.join([f"Question: {train_dataset[i]['question']} \nAnswer: {train_dataset[i]['answer']}" for i in range(n_shot)])
    example['prompt'] = f"{three_shot_prompt}\n Question: {example['question']} \nAnswer :"
    example['completion'] = example['answer']
    example['text'] = example['prompt'] +'||' + example['completion']
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
    train_dataset, eval_dataset = dataset['train'], dataset['test']
    train_dataset = train_dataset.map(prepare_nshot, num_proc=16).remove_columns(['prompt', 'completion'])
    eval_dataset = eval_dataset.map(prepare_nshot, num_proc=16).remove_columns(['prompt', 'completion'])
    
    print (len(train_dataset))
    #model_path=f'{checkpoint_root}/fineweb_h4_decay_nonparallel_mixed_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors'
    model_path=f'{checkpoint_root}/finemath_h4_mixed_decay_nonparallel_projs_k1_1024_n16_c1024_b16x4/checkpoint-200000/model.safetensors'
    load_model(model, model_path)
    print ('pretrained model loaded')
    response_template = '||'
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
        num_train_epochs = 13,
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
       )
    
    model.train()
    trainer.train()
