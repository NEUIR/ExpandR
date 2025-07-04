import os
from collections import defaultdict
import numpy as np
import torch
import glob
import logging


import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

import dist_utils as dist_utils

from evaluate.denseencodermodel import DenseEncoderModel

logger = logging.getLogger(__name__)






def evaluate_model(
    max_length,
    query_encoder,
    doc_encoder,
    tokenizer,
    dataset,
    cqad_path,
    batch_size=128,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    is_main=True,
    split="test",
    score_function="dot",
    beir_dir="BEIR/datasets",
    save_results_path=None,
    lower_case=False,
    normalize_text=False,
    embeddings_file=None,
    results_path=None,
    every_score_path=None,
    query_file_name=None,
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder
    
    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            norm_query=norm_query,
            norm_doc=norm_doc,
            lower_case=lower_case,
            normalize_text=normalize_text,
        ),
        batch_size=batch_size,
        dataname=dataset,
    )
    
        
    retriever = EvaluateRetrieval(dmodel, score_function=score_function, embeddings_file=embeddings_file)
    data_path = os.path.join(beir_dir, dataset)

    if not os.path.isdir(data_path) and is_main:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)
    dist_utils.barrier()

    if not dataset == "cqadupstack":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path, query_file=query_file_name).load(split=split)
        results = retriever.retrieve(corpus, queries)      
        
        if is_main:
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                if isinstance(metric, str):
                    metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                for key, value in metric.items():
                    metrics[key].append(value)
            if save_results_path is not None:
                torch.save(results, f"{save_results_path}")
    elif dataset == "cqadupstack":  # compute average over datasets
        data_path = os.path.join(cqad_path, "*/")
        print(data_path)
        paths = glob.glob(data_path)
        paths = [d for d in paths if os.path.isdir(d)]
        # print(paths)
        # exit()
        for path in paths:
            print(path)
            corpus, queries, qrels = GenericDataLoader(data_folder=path, query_file=query_file_name).load(split=split)
            results = retriever.retrieve(corpus, queries)
            sub_dataset = os.path.basename(os.path.normpath(path))
            logger.info("Evaluating {}".format(sub_dataset))
            if is_main:
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
                for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                    if isinstance(metric, str):
                        metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                    for key, value in metric.items():
                        metrics[key].append(value)
        for key, value in metrics.items():
            assert (
                len(value) == 12
            ), f"cqadupstack includes 12 datasets, only {len(value)} values were compute for the {key} metric"

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    return metrics
