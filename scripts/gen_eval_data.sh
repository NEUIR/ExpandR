cd ../src/generator

# inference
CUDA_VISIBLE_DEVICES=1 python vllm_inference_eval.py  \
    --model_path # the path to dpo-llama \
    --queries_file_name queries_q2d.jsonl \
    --beir_dir ExpandR/data/beir \
    --task_type q2d 