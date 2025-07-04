# # inference
# CUDA_VISIBLE_DEVICES=1 python ExpandR/src/generator/vllm_inference_eval.py \
#     --model_name_or_path # the path to dpo-llama \
#     --queries_file_name queries_q2d.jsonl \
#     --beir_dir ExpandR/data/beir \
#     --task_type q2d

export PYTHONPATH=ExpandR/src
# eval
BEIR_DIR=ExpandR/data/beir

EMBEDDINGS_FILE_DIR=ExpandR/data/corpus_embeddings

MODEL_PATH='' # The path of the trained anchordr model

OUTPUT_DIR_BASE='' # the path to save eval results

cur_query_file_name='queries_q2d.jsonl'

caqd_path_dir='ExpandR/data/beir/cqadupstack'

# msmarco
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/msmarco_embeddings.npy \
    --dataset msmarco \
    --output_dir $OUTPUT_DIR_BASE/msmarco \
    --per_gpu_batch_size 256 \
    --query_file_name $cur_query_file_name 


# trec-covid
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/trec-covid_embeddings.npy \
    --dataset trec-covid \
    --output_dir $OUTPUT_DIR_BASE/trec-covid \
    --per_gpu_batch_size 1024 \
    --query_file_name $cur_query_file_name


# nfcorpus
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/nfcorpus_embeddings.npy \
    --dataset nfcorpus \
    --output_dir $OUTPUT_DIR_BASE/nfcorpus \
    --per_gpu_batch_size 1024 \
    --query_file_name $cur_query_file_name

# nq
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/nq_embeddings.npy \
    --dataset nq \
    --output_dir $OUTPUT_DIR_BASE/nq \
    --per_gpu_batch_size 512 \
    --query_file_name $cur_query_file_name


# hotpotqa
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/hotpotqa_embeddings.npy \
    --dataset hotpotqa \
    --output_dir $OUTPUT_DIR_BASE/hotpotqa \
    --per_gpu_batch_size 512 \
    --query_file_name $cur_query_file_name


# fiqa
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/fiqa_embeddings.npy \
    --dataset fiqa \
    --output_dir $OUTPUT_DIR_BASE/fiqa \
    --per_gpu_batch_size 1024 \
    --query_file_name $cur_query_file_name


# arguana
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/arguana_embeddings.npy \
    --dataset arguana \
    --output_dir $OUTPUT_DIR_BASE/arguana \
    --per_gpu_batch_size 1024 \
    --query_file_name $cur_query_file_name


# webis-touche2020
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/webis-touche2020_embeddings.npy \
    --dataset webis-touche2020 \
    --output_dir $OUTPUT_DIR_BASE/webis-touche2020 \
    --per_gpu_batch_size 1024 \
    --query_file_name $cur_query_file_name


# cqadupstack
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/cqadupstack_embeddings.npy \
    --dataset cqadupstack \
    --caqd_path $caqd_path_dir \
    --output_dir $OUTPUT_DIR_BASE/cqadupstack \
    --per_gpu_batch_size 1024 \
    --query_file_name $cur_query_file_name


# quora
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/quora_embeddings.npy \
    --dataset quora \
    --output_dir $OUTPUT_DIR_BASE/quora \
    --per_gpu_batch_size 512 \
    --query_file_name $cur_query_file_name


# dbpedia-entity
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/dbpedia-entity_embeddings.npy \
    --dataset dbpedia-entity \
    --output_dir $OUTPUT_DIR_BASE/dbpeida \
    --per_gpu_batch_size 1024 \
    --query_file_name $cur_query_file_name


# scidocs
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/scidocs_embeddings.npy \
    --dataset scidocs \
    --output_dir $OUTPUT_DIR_BASE/scidocs \
    --per_gpu_batch_size 1024 \
    --query_file_name $cur_query_file_name


# fever
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/fever_embeddings.npy \
    --dataset fever \
    --output_dir $OUTPUT_DIR_BASE/fever \
    --per_gpu_batch_size 256 \
    --query_file_name $cur_query_file_name


# climate-fever
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/climate-fever_embeddings.npy \
    --dataset climate-fever \
    --output_dir $OUTPUT_DIR_BASE/climate-fever \
    --per_gpu_batch_size 256 \
    --query_file_name $cur_query_file_name


# scifact
python ExpandR/src/evaluate/eval_beir.py \
    --beir_dir $BEIR_DIR \
    --model_name_or_path $MODEL_PATH \
    --embeddings_file $EMBEDDINGS_FILE_DIR/scifact_embeddings.npy \
    --dataset scifact \
    --output_dir $OUTPUT_DIR_BASE/scifact \
    --per_gpu_batch_size 1024 \
    --query_file_name $cur_query_file_name




