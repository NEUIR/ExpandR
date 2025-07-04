
import argparse
import torch
import logging
import os

import slurm
import contriever
import beir_utils
import utils
import dist_utils

from transformers import RobertaTokenizer, RobertaConfig

logger = logging.getLogger(__name__)


def main(args):

    if args.local_rank != -1:
        slurm.init_distributed_mode(args)
        slurm.init_signal_handler()
    else:
        args.local_rank = 0
        args.global_rank = 0
        args.world_size = 1

    os.makedirs(args.output_dir, exist_ok=True)

    logger = utils.init_logger(args)
    
    model, tokenizer, _ = contriever.load_retriever(args.model_name_or_path)
    model = model.cuda()
    model.eval()
    query_encoder = model
    doc_encoder = model   

    logger.info("Start indexing")
    
    logger.info("Evaluating {}".format(args.dataset))


    metrics = beir_utils.evaluate_model(
        max_length=args.text_maxlength,
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        tokenizer=tokenizer,
        dataset=args.dataset,
        cqad_path=args.caqd_path,
        batch_size=args.per_gpu_batch_size,
        norm_query=args.norm_query,
        norm_doc=args.norm_doc,
        is_main=dist_utils.is_main(),
        split="dev" if args.dataset == "msmarco" else "test",
        score_function=args.score_function,
        beir_dir=args.beir_dir,
        save_results_path=args.save_results_path,
        lower_case=args.lower_case,
        normalize_text=args.normalize_text,
        embeddings_file=args.embeddings_file,
        query_file_name=args.query_file_name,
    )

    if dist_utils.is_main():
        for key, value in metrics.items():
            logger.info(f"{args.dataset} : {key}: {value:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_dir", type=str, default="./", help="Directory to save and load beir datasets")
    parser.add_argument("--caqd_path", type=str, default="./", help="Directory to save and load beir datasets")
    parser.add_argument("--results_path", type=str, default="./", help="Path to save results")
    
    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--embeddings_file", type=str, default="beir/corpus_embeddings.npy",help="Save corpus embeddings path")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    
    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")
    parser.add_argument("--lower_case", action="store_true", help="lowercase query and document text")
    parser.add_argument(
        "--normalize_text", action="store_true", help="Apply function to normalize some common characters"
    )
    parser.add_argument("--save_results_path", type=str, default=None, help="Path to save result object")
    parser.add_argument(
        "--score_function", type=str, default="dot", help="Metric used to compute similarity between two embeddings"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")
    parser.add_argument("--query_file_name", type=str, default='queries.jsonl', help="query file name")

    args, _ = parser.parse_known_args()
    main(args)
