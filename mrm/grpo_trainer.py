import torch
import re
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, load_from_disk, Dataset
from recurrent_srm_model import RecurrentMLPMixer
from dual_srm import DualMLPMixer
from transformers.generation import GenerationMixin, GenerationConfig

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
        self.main_input_name = 'input_ids'
        self._supports_cache_class = False
        self.cache_built = False
        self.device = self.output_layer.weight.device
        self.counter = 0

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
                x = block(x, i)
        self.cache_built = True
        return

    def clear_cache(self):
        for block in self.mixer_blocks:
            for h in range(len(block.token_mixing_layer.mixer_heads)):
                block.token_mixing_layer.mixer_heads[h].cache = torch.zeros(self.hidden_dim//self.n_heads).to('cuda') # only for mixed heads

    def forward(self, input_ids, labels=None, **kwargs):
        if not self.cache_built and not self.training:
            self.build_cache(input_ids)
        index = input_ids.shape[1] - 1
        input_ids = input_ids[:, -1] # last token only
        
        # model's forward pass
        x = self.input_layer(input_ids)
        for block in self.mixer_blocks:
            x = block(x, index)
        logits = self.output_layer(x).unsqueeze(1)
        if labels is not None:
            return CausalLMOutput(loss=0, logits=logits)
        else:
            return CausalLMOutput(loss=0, logits=logits)


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train") -> Dataset:
    # 3 shot
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer']),
        'database': 'yes'
    }) 
    return data 


def output_check(predicted_output, ground_truth):
    """
    Check the output for execution accuracy.
    Args:
        output (str): The generated SQL query.
        gold_sql (str): The ground truth SQL query.
    Returns:
        bool: True if the output is correct, False otherwise.
    """
    cleaned_output = predicted_output.split('####')[1].strip(' ,!@#$%^&*')
    return res

# Reward functions
def correctness_reward_func(prompts, completions, answer, database, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def length_reward(completions, **kwargs):
    """For testing a trivial reward maximization"""
    responses = [completion[0]["content"] for completion in completions]
    return [0.001*len(response) for response in responses]

def downsample_rewards(completions, num_samples = 16, **kwargs):
    for response in responses:
        pass


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dotenv()
    checkpoint_root = os.getenv('CHECKPOINT_ROOT')
    data_root = os.getenv('DATA_ROOT')
    tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    print("Vocab size: ", n_vocab)
    dataset = get_gsm8k_questions()
    print (dataset[0])

    tokenized_length = 1024
    dim = 1024
    layers = 16
    n_heads = 4
    kernel = 1

    model = RouterModel(
        n_vocab, dim, tokenized_length, layers, heads=n_heads, kernel=kernel, expanded_convs=False, copy=False, 
        mixed_heads=True, combined_heads=False, decay=True, parallel_heads=False, use_projections=True)

    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset, eval_dataset = dataset['train'], dataset['test']
    print (train_dataset[0])
    print (len(train_dataset))
    model_path=''

    max_prompt_length = 720

    output_dir = f'{checkpoint_root}/srm_grpo_128'
    training_args = GRPOConfig(
        learning_rate = 1e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_torch",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        num_generations = 128, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        num_train_epochs = 10, # Set to 1 for a full training run
        save_steps = 500,
        max_grad_norm = 0.9,
        report_to = "none",
        output_dir = output_dir,
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            execution_reward_func,
            format_reward_func
        ],
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )

    trainer.train()