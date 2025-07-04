from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import datasets
import jsonlines
from tqdm import tqdm
import os
from promptor import Promptor

def post_process(in_str):
    patterns = [
        'Here is a passage to answer the question:',
        'Here is a passage that answers the question:',
        'Here is a passage answering the question:',
        "Here's a passage that attempts to answer the question:",
        "Here's a passage that answers the question:",
        "Here's a passage that answers your question:",
        "Here's a possible passage:",
        "Here is a possible passage:",
        "Here is a potential passage:",
        "Here's a potential passage:",
        "Here's a passage:",
        "Here is the passage:",
        "Here's the passage:"
    ]
    for pattern in patterns:
        if pattern in in_str:
            in_str = in_str.split(pattern)[1]
            break
    return in_str.strip()

def generate_doc_for_dpo(args):
    
    max_batch_size = 256
    model_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.9)
    
    query_path = args.query_path
    queries = datasets.load_dataset('json', data_files=[query_path], split="train")
    query_texts = [query['query'] for query in queries]
    length = len(query_texts) 
    
    temperature_list = [0.8, 0.9, 1.0, 1.1]
    top_p = 0.9
    outfile_dir = args.outfile_dir
    os.makedirs(outfile_dir, exist_ok=True)
    
    for temperature in temperature_list:
        sampling_params = SamplingParams(n=8, temperature=temperature, top_p=top_p, max_tokens=512)
        print(f"temperature is {temperature}")
        save_file_name = f"q2d_doc_gen_{temperature}.jsonl"
        save_file = os.path.join(outfile_dir, save_file_name)       
        prompter = Promptor(task=args.task_type)
        
        with jsonlines.open(save_file, 'w') as writer:
            for i in tqdm(range(0, length, max_batch_size), leave=False, desc="Generating documents"):
                j = min(i + max_batch_size, length)
                queries = query_texts[i: j]
                prompts_list = []
                for query in queries:
                    prompt = prompter.build_prompt(query)
                    user_input = [ {"role": "user", "content": prompt},]
                    user_input = tokenizer.apply_chat_template(user_input, add_generation_prompt=True, tokenize=False)
                    prompts_list.append(user_input)
                outputs_list = llm.generate(prompts_list, sampling_params)
                
                
                for outputs in outputs_list:
                    outputs = outputs.outputs
                    texts = [output.text for output in outputs]
                    for text in texts:
                        text = post_process(text)
                        output_data = {
                            "passage": text
                        }
                        writer.write(output_data)
                
        
        passage_file_name = f"q2d_d8_{temperature}.jsonl"
        passage_file = os.path.join(outfile_dir, passage_file_name)
        with jsonlines.open(save_file, "r") as reader, jsonlines.open(passage_file, "w") as writer:
            buffer = []
            for in_data in reader:
                buffer.append(in_data["passage"])
                if len(buffer) == 8:
                    writer.write({"passages": buffer})
                    buffer = []
            
    # Merge documents generated at all temperatures
    combined_file_path = os.path.join(outfile_dir, 'gen_doc_combined.jsonl')
    gen1 = os.path.join(outfile_dir, 'q2d_d8_0.8.jsonl')
    gen2 = os.path.join(outfile_dir, 'q2d_d8_0.9.jsonl')
    gen3 = os.path.join(outfile_dir, 'q2d_d8_1.0.jsonl')
    gen4 = os.path.join(outfile_dir, 'q2d_d8_1.1.jsonl')
    
    with jsonlines.open(gen1, 'r') as reader1, \
        jsonlines.open(gen2, 'r') as reader2, \
        jsonlines.open(gen3, 'r') as reader3, \
        jsonlines.open(gen4, 'r') as reader4, \
        jsonlines.open(combined_file_path, 'w') as writer:
        for indata1, indata2, indata3, indata4 in zip(reader1, reader2, reader3, reader4):
            combined_list = []
            combined_list.extend(indata1["passages"])
            combined_list.extend(indata2["passages"])
            combined_list.extend(indata3["passages"])
            combined_list.extend(indata4["passages"])
            outdata = {"passages": combined_list}
            writer.write(outdata)

    print(f"All data combined into {combined_file_path}")


    # out_file_name = f"query_posi_doc32_10w.jsonl"
    # out_file = os.path.join(outfile_dir, out_file_name)
    # with jsonlines.open(query_path, 'r') as reader1, jsonlines.open(combined_file_path, 'r') as reader2, jsonlines.open(out_file, 'w') as writer:
    #     for data1, data2 in zip(reader1, reader2):
    #         query = data1['query']
    #         posi = data1['positive']
    #         cot = data2['passages']
    #         writer.write({"query": query, "positives": posi, "doc": cot})



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
                        default=None,
                        help="The path of model file."
                        )
    parser.add_argument("--query_path", type=str, 
                        default=None,
                        help="The path of queries where the model generates corresponding documents based on the query."
                        )
    parser.add_argument("--task_type", type=str, 
                        default=None,
                        help="prompt type."
                        )
    
    parser.add_argument("--outfile_dir", type=str,
                        help="DPO data generated by llm."
                        )

    args = parser.parse_args()
    generate_doc_for_dpo(args)

if __name__ == "__main__":
    main()


