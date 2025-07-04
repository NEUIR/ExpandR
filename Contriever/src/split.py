import os
import random

# raw data file path
input_file = 'LLM-QE/data/process_data/merge_data_80w.jsonl'
# for dpo training (10w)
output_file_1 = 'LLM-QE/data/process_data/dpo-data/split_10w.jsonl'
# for supervised training (708,740)
output_file_2 = 'LLM-QE/data/process_data/supervised-data/split_70w.jsonl'

sample_size = 100000

def random_split_data(input_path, out_path_1, out_path_2, n):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)

    with open(out_path_1, 'w', encoding='utf-8') as fout1, \
         open(out_path_2, 'w', encoding='utf-8') as fout2:
        for i, line in enumerate(lines):
            if i < n:
                fout1.write(line)
            else:
                fout2.write(line)


if __name__ == "__main__":
    random_split_data(input_file, output_file_1, output_file_2, sample_size)