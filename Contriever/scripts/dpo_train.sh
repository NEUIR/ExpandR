# shuf data 3w
gpu_vis=1
MASTER_PORT=2351
formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time
deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT LLM-QE/src/train/dpo/train.py \
    --model_name_or_path LLM-QE/model/base_model/Meta-Llama-3-8B-Instruct \
    --train_data_path LLM-QE/data/process_data/dpo-data/gen-data/train_data.jsonl \
    --eval_data_path LLM-QE/data/process_data/dpo-data/gen-data/dev_data.jsonl \
    --max_length 612 \
    --max_prompt_length 100 \
    --output_dir LLM-QE/model/dpo_ckpts/$formatted_time/ \
    --save_steps 300 \
    --eval_steps 300 \
    --per_device_train_batch_size 2 \
    --max_grad_norm 10.0 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 5 \
    --logging_dir LLM-QE/model/dpo_ckpts/$formatted_time/logdir \
    --bf16 True \
    --use_lora True \
    --num_train_epochs 2 \
    --deepspeed LLM-QE/src/train/configs/ds_config_zero2.json


