import json
import random
import torch
import numpy as np
from tqdm import tqdm
from flagmodel import FlagModel
from generator.promptor import Promptor
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, 
                    default=None)
parser.add_argument("--outdir", action='store_true',
                        help="The path of dpo data."
                        )
parser.add_argument("--task_type", type=str, 
                        default=None,
                        help="prompt type."
                        )

args = parser.parse_args()

random.seed(42)  


model = FlagModel(args.model_name_or_path, pooling_method='mean', use_fp16=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_file_name = 'query_posi_doc32_ans.jsonl'
input_file = os.path.join(args.outdir, input_file_name)

raw_dpo_file_name = 'dpo-data-raw.jsonl'
raw_dpo_file = os.path.join(args.outdir, raw_dpo_file_name)

output_file_name = 'dpo-data.jsonl'
output_file = os.path.join(args.outdir, output_file_name)


with open(input_file, 'r', encoding='utf-8') as f_in, open(raw_dpo_file_name, 'w', encoding='utf-8') as f_out:
    for line in tqdm(f_in, desc="Processing Passages"):
        data = json.loads(line)
        query = data['query']
        positive = data['positive']
        docs = data['passages']  
        answer = data['answer']

        if answer == '':
            continue

        if len(docs) != 32:
            print(f"Warning: Passage contains {len(docs)} docs, expected 32.")
            continue

        
        with torch.no_grad():
            positive_embedding = model.encode_corpus([positive], max_length=512)  
            answer_embedding = model.encode_corpus([answer], max_length=512)  
            docs_embeddings = model.encode_queries(docs, batch_size=32, max_length=512)  
            
             
            if isinstance(positive_embedding, np.ndarray):
                positive_embedding = torch.tensor(positive_embedding).to(device)
            if isinstance(answer_embedding, np.ndarray):
                answer_embedding = torch.tensor(answer_embedding).to(device)
            if isinstance(docs_embeddings, np.ndarray):
                docs_embeddings = torch.tensor(docs_embeddings).to(device)
            
            positive_embedding = positive_embedding.expand(docs_embeddings.size(0), -1)
            answer_embedding = answer_embedding.expand(docs_embeddings.size(0), -1)
            
            # sim(.)
            posi_dot_product_scores = torch.sum(positive_embedding * docs_embeddings, dim=1)
            answ_dot_product_scores = torch.sum(answer_embedding * docs_embeddings, dim=1)
            
            # filter
            dot_product_scores = posi_dot_product_scores + answ_dot_product_scores
            max_score, max_index = torch.max(dot_product_scores, dim=0) 
            if max_score < 0.6:
                continue
            
            # rank
            posi_sorted_indices = torch.argsort(posi_dot_product_scores, descending=True)
            answ_sorted_indices = torch.argsort(answ_dot_product_scores, descending=True)

            posi_ranks = torch.zeros_like(posi_sorted_indices)
            answ_ranks = torch.zeros_like(answ_sorted_indices)

            posi_ranks[posi_sorted_indices] = torch.arange(1, len(posi_dot_product_scores) + 1)
            answ_ranks[answ_sorted_indices] = torch.arange(1, len(answ_dot_product_scores) + 1)

            # reward score
            r_rank = 1 / posi_ranks
            r_ans = 1 / answ_ranks            
            r_final = r_rank + r_ans         
            
            r_final = r_final.cpu().numpy()

            most_relevant_idx = np.argmax(r_final)
            least_relevant_idx = np.argmin(r_final)
            
            most_relevant_doc = docs[most_relevant_idx]
            least_relevant_doc = docs[least_relevant_idx]
            
            
            most_relevant_score = float(r_final[most_relevant_idx])
            least_relevant_score = float(r_final[least_relevant_idx])


            prompter = Promptor(task=args.task_type)
            prompt = prompter.build_prompt(query)
            out_data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": most_relevant_doc
                },
                "rejected": {
                    "from": "gpt",
                    "value": least_relevant_doc
                }
            }

            
            f_out.write(json.dumps(out_data, ensure_ascii=False) + '\n')
            

# shuf 3w data
sample_size = 30000
with open(raw_dpo_file_name, 'r', encoding='utf-8') as f:
    lines = f.readlines()


if len(lines) < sample_size:
    print(f"Warning: There are only {len(lines)} records in the data file, which is less than 30000. All will be used.")
    sampled_lines = lines  
else:
    sampled_lines = random.sample(lines, sample_size)

with open(output_file, 'w', encoding='utf-8') as f_out:
    for line in sampled_lines:
        f_out.write(line)
