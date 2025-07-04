cd ../src/generator

# inference & construct supervised data
CUDA_VISIBLE_DEVICES=1 python vllm_inference_supervised.py  \
    --model_name_or_path ExpandR/model/base_model/Llama-3-8B-Instruct \
    --query_path ExpandR/data/process_data/supervised-data/split_70w.jsonl \
    --task_type q2d \
    --outfile_dir  ExpandR/data/process_data/supervised-data/gen-data


# split train & dev
cd ../src
python shuf_data_train_dev.py  \
    --input_file ExpandR/data/process_data/supervised-data/gen-data/supervised_data.jsonl \
    --train_file ExpandR/data/process_data/supervised-data/gen-data/train_data.jsonl \
    --dev_file ExpandR/data/process_data/supervised-data/gen-data/dev_data.jsonl \
    --dev_num 70874
