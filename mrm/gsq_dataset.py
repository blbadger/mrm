from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import random

def prepare_nshot(example, n_shot=5):
    start_index = random.randint(0, len(train_dataset) -n_shot)
    three_shot_prompt = '\n'.join([f"Question: {train_dataset[i+start_index]['question']} \nAnswer: {train_dataset[i+start_index]['answer']}" for i in range(n_shot)])
    example['prompt'] = f"{three_shot_prompt}"
    return example

dataset = load_dataset("openai/gsm8k", "main")
train_dataset = dataset['train']
train_dataset = train_dataset.map(prepare_nshot, num_proc=16)

if __name__ == '__main__':
    ## VLLM output
    model_name = "Qwen/Qwen2.5-Math-7B"
    llm = LLM(model_name, tensor_parallel_size=4)
    save_every = 200
    output_path = '/home/badger/gsq_dataset'

    all_outputs = []
    for i in tqdm(range(len(train_dataset))):
        text=f'''<|im_start|>system\nPlease write a single question with a correct answer, matching the exact format of the examples given (answer follows ####, computation in <<>>).<|im_end|>
    <|im_start|>user\nGiven the following examples, write a new grade-school mathematical question distinct from the ones provided and provide a correct answer in the same format as shown in the examples given. Do not simply answer existing questions, write new a one and answer that. Do not box the answer, use the format provided. Examples: \n\n{train_dataset['prompt'][i]}<|im_end|>
    <|im_start|>assistant\n Question: '''
        text = f"{train_dataset['prompt'][i]}\nQuestion:"
        #print (train_dataset['prompt'][i])
        output = llm.generate(
            text, 
            use_tqdm=False,
            sampling_params=SamplingParams(
                n=200, temperature=0.7, 
                top_p=0.9,
                max_tokens=512,
                stop=["</s>", "<|im_end|>", "<|endoftext|>"],
                stop_token_ids=[151645, 151643],
                )
            )
        all_outputs += [output[0].outputs[i].text for i in range(len(output))]
        if i % save_every == 0 and i > 0:
                dataset_dict = {'samples': all_outputs}
                dataset = Dataset.from_dict(dataset_dict)
                dataset.save_to_disk(output_path + f'/_{i}')
                all_outputs = []

                print(output[0].outputs[0].text)
                print (output[0].outputs[1].text)
