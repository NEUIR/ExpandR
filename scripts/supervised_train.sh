
formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time
CUDA_VISIBLE_DEVICES=1,2 nohup torchrun --nproc_per_node 2 --master_port=29501 \
    LLM-QE/src/train/supervised/run.py \
    --output_dir LLM-QE/model/supervised_ckpts/$formatted_time/ \
    --model_name_or_path LLM-QE/model/base_model/contriever \
    --train_data LLM-QE/data/process_data/supervised-data/gen-data/train_data.jsonl \
    --learning_rate 3e-5 \
    --train_group_size 2 \
    --num_train_epochs 3 \
    --sentence_pooling_method 'mean' \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 32 \
    --normlized True \
    --temperature 0.02 \
    --use_inbatch_neg True \
    --query_max_len 512 \
    --passage_max_len 512 \
    --logging_steps 3 \
    --save_steps 200 \
    --eval_steps 200 \
    --report_to tensorboard \
    --logging_dir LLM-QE/model/supervised_ckpts/$formatted_time//logdir/ 