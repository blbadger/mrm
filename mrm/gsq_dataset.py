from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk, Dataset

def prepare_nshot(example, n_shot=3):
    three_shot_prompt = '\n'.join([f"Question: {train_dataset[i]['question']} \nAnswer: {train_dataset[i]['answer']}" for i in range(n_shot)])
    example['prompt'] = f"{three_shot_prompt}"
    return example

dataset = load_dataset("openai/gsm8k", "main")
train_dataset = dataset['train']
train_dataset = train_dataset.map(prepare_nshot, num_proc=16)

## VLLM output
model_name = "Qwen2.5-Math-7B-Instruct"
llm = LLM(model_name)
save_every = 1000

for i in range(len(train_dataset)):
    text=f'''<|im_start|>system\nPlease give correct answers with questions, matching the exact format of the examples given.<|im_end|>
<|im_start|>user\nGiven the following examples, write a new grade-school mathematical Question and Answer in the same format:\n{train_dataset['prompt'][i]}<|im_end|>
<|im_start|>assistant\n'''
    output = llm.generate(
        text, 
        sampling_params=SamplingParams(
            n=100, temperature=0.7, 
            top_p=0.9, 
            max_tokens=2048,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"],
            stop_token_ids=[151645, 151643],
            tensor_parallel_size=4)
        )
    all_outputs.append(output)

    print(output[0].outputs[0].text)