from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List, Optional
import argparse
import os
import datasets
import jsonlines
from tqdm import tqdm
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

def generate_docs(args, data_dir, dataname_list, llm, tokenizer, sampling_params, prompter):
    max_batch_size = 128

    for dataname in dataname_list:
        print(f"Dataname is {dataname}")
        query_file_name = 'queries.jsonl'
        query_path = os.path.join(data_dir, dataname, query_file_name)
        queries = datasets.load_dataset('json', data_files=[query_path], split="train")
        query_texts = [query['text'] for query in queries]
        length = len(query_texts)

        save_file_name = 'doc_gen.jsonl'
        save_file = os.path.join(data_dir, dataname, save_file_name)

        with jsonlines.open(save_file, 'w') as writer:
            for i in tqdm(range(0, length, max_batch_size), leave=False, desc="Generating documents"):
                j = min(i + max_batch_size, length)
                queries_batch = query_texts[i: j]
                prompts_list = []
                for query in queries_batch:
                    prompt = prompter.build_prompt(query, '')
                    user_input = [{"role": "user", "content": prompt}]
                    user_input = tokenizer.apply_chat_template(user_input, add_generation_prompt=True, tokenize=False)
                    prompts_list.append(user_input)

                outputs = llm.generate(prompts_list, sampling_params)
                for output in outputs:
                    passage = post_process(output.outputs[0].text)
                    writer.write({"passage": passage})

        out_file_name = 'query_d.jsonl'
        outfile = os.path.join(data_dir, dataname, out_file_name)
        with jsonlines.open(save_file, "r") as reader, jsonlines.open(outfile, "w") as writer:
            for index, indata in enumerate(reader):
                writer.write({"query": query_texts[index], "passage": indata['passage']})

        print(f'Query_d file saved to {outfile}')

        queries_path = os.path.join(data_dir, dataname, args.queries_file_name)
        with jsonlines.open(outfile, "r") as reader1, jsonlines.open(query_path, "r") as reader2, jsonlines.open(queries_path, "w") as writer:
            for in_data1, in_data2 in zip(reader1, reader2):
                in_data2["text"] = [in_data1['query'], in_data1['passage']]
                writer.write(in_data2)

        print(f'New queries file saved to {queries_path}')

def generate_doc_for_eval_beir(args):
    data_dir = args.beir_dir
    model_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.95)
    sampling_params = SamplingParams(temperature=1, top_p=0.9, max_tokens=512)
    prompter = Promptor(task=args.task_type)

    dataname_list = ['arguana', 'climate-fever', 'dbpedia-entity', 'fever', 'fiqa', 'hotpotqa',
                     'nfcorpus', 'msmarco', 'nq', 'quora', 'scifact', 'scidocs', 'trec-covid', 'webis-touche2020']
    generate_docs(args, data_dir, dataname_list, llm, tokenizer, sampling_params, prompter)

    cqadata_dir = os.path.join(data_dir, 'cqadupstack')
    cqadataname_list = ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics',
                        'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    generate_docs(args, cqadata_dir, cqadataname_list, llm, tokenizer, sampling_params, prompter)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None, help="The path of model file.")
    parser.add_argument("--queries_file_name", type=str, default=None, help="Path of save inference doc.")
    parser.add_argument("--beir_dir", type=str, default=None, help="Directory to save and load beir datasets.")
    parser.add_argument("--task_type", type=str, default=None, help="prompt type.")
    args = parser.parse_args()
    generate_doc_for_eval_beir(args)

if __name__ == "__main__":
    main()
