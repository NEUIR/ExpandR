cd ../src/generator

# inference docs
CUDA_VISIBLE_DEVICES=1 python vllm_inference_dpo.py  \
    --model_name_or_path LLM-QE/model/base_model/Llama-3-8B-Instruct \
    --query_path LLM-QE/data/process_data/dpo-data/split_10w.jsonl \
    --task_type q2d \
    --outfile_dir  LLM-QE/data/process_data/dpo-data/gen-data


# inference answer
CUDA_VISIBLE_DEVICES=1 python vllm_inference_answer.py  \
    --model_name_or_path LLM-QE/model/base_model/Llama-3-8B-Instruct \
    --query_path LLM-QE/data/process_data/dpo-data/split_10w.jsonl \
    --task_type q2a \
    --outfile_dir  LLM-QE/data/process_data/dpo-data/gen-data


cd ../src

# combined datas
python get_query_posi_doc32_ans.py \
    --query_path LLM-QE/data/process_data/dpo-data/split_10w.jsonl \
    --doc_path LLM-QE/data/process_data/dpo-data/gen-data/gen_doc_combined.jsonl \
    --answer_path LLM-QE/data/process_data/dpo-data/gen-data/gen_answer.jsonl \
    --outfile LLM-QE/data/process_data/dpo-data/gen-data/query_posi_doc32_ans.jsonl


# construct dpo data
CUDA_VISIBLE_DEVICES=1 python construct_dpo_data.py \
    --model_name_or_path LLM-QE/model/base_model/Llama-3-8B-Instruct \
    --outdir LLM-QE/data/process_data/dpo-data/gen-data \
    --task_type q2d



# split train & dev
python shuf_data_train_dev.py  \
    --input_file LLM-QE/data/process_data/dpo-data/gen-data/dpo_data.jsonl \
    --train_file LLM-QE/data/process_data/dpo-data/gen-data/train_data.jsonl \
    --dev_file LLM-QE/data/process_data/dpo-data/gen-data/dev_data.jsonl \
    --dev_num 3000