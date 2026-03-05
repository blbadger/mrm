import transformers
from transformers import RwkvConfig, RwkvModel
from prettytable import PrettyTable
from safetensors.torch import save_file, load_model
from safetensors import safe_open
import safetensors
import datasets
import warnings
import shutil
from dotenv import load_dotenv

from transformer
warnings.filterwarnings(action='ignore')


class RWKVCLM(nn.Module):
    def __init__(self, model, vocab_size=8000, dim=512):
        super().__init__()
        self.model = model # a LlamaModel object
        self.lm_head = nn.Linear(dim, vocab_size)
        self.vocab_size = vocab_size
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None, attention_mask=None):
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

dim = 512
context_length = 512
compression = 1
n_layers = 16
n_heads = 4

tokenizer = AutoTokenizer.from_pretrained(f"{data_root}/tokenizer_fineweb_8k")
vocab_size = len(tokenizer)
config_kwargs = {
    'hidden_size':encoder_dim,
    'intermediate_size': 4*encoder_dim,
    'num_hidden_layers': n_layers,
    'num_attention_heads': n_heads,
    'vocab_size': vocab_size
}
print (llama_config_kwargs)
# Initializing a LLaMA model
configuration = RwkvConfig(**config_kwargs)
model = RwkvModel(configuration)
model = RWKVCLM(model)

train_path = f"{data_root}/fineweb-tokenized-train-c512-8k"
test_path = f"{data_root}/fineweb-tokenized-test-c512-8k"

datasets.config.IN_MEMORY_MAX_SIZE = 1e9
train_dataset = load_from_disk(train_path) 
test_dataset = load_from_disk(test_path) 
print (len(train_dataset[0]['input_ids']))

batch_size = 32
n_devices = 1

# get number of devices (assumes that all visible devices are used for training)
if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()

# descriptive name for output
output_dir = f'{checkpoint_root}/fineweb_rwkv\
_{encoder_dim}\
c{compression}\
_d{decoder_dim}\
_n{n_layers}\
_c{context_length}_b{batch_size}x{n_devices}'

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=500,
        eval_steps=4000,
        save_steps=20000,
        learning_rate=2e-4,
        fp16=True,
        eval_strategy='steps',
        output_dir=output_dir,
        optim='adamw_torch',
        overwrite_output_dir=True,
        max_steps=200000,
        torch_compile=True
)

trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
#       compute_metrics=compute_hamming_metric,
#       preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

print (f"training model, saving to {output_dir}")
# save driver code snapshot in checkpoint dir
code_path = os.path.abspath(__file__)
if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
shutil.copy(code_path, output_dir)

print (f"training begun: saving results in {output_dir}")
model.train()
trainer.train()
