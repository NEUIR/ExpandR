export PYTHONPATH=ExpandR/src/train
formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time
opts='--use_t5_decoder --use_converted'
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2 --master_port=29501 \
    ExpandR/src/train/supervised/run.py \
    --output_dir ExpandR/model/supervised_ckpts/$formatted_time/ \
    --model_name_or_path ExpandR/model/base_model/AnchorDR \
    --train_data ExpandR/data/process_data/supervised-data/gen-data/train_data.jsonl \
    --learning_rate 1e-5 \
    --train_group_size 2 \
    --num_train_epochs 3 \
    --sentence_pooling_method 'mean' \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 64 \
    --normlized True \
    --temperature 0.02 \
    --use_inbatch_neg True \
    --query_max_len 512 \
    --passage_max_len 512 \
    --logging_steps 3 \
    --save_steps 200 \
    --eval_steps 200 \
    --report_to tensorboard \
    --logging_dir ExpandR/model/supervised_ckpts/$formatted_time/logdir/ 


