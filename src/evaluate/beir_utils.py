import os
from collections import defaultdict
from typing import List, Dict, Union, cast
import numpy as np
import torch
import torch.distributed as dist
import glob
import logging


import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

import dist_utils as dist_utils
import normalize_text
logger = logging.getLogger(__name__)


class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text

    # 
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        return self.encode_queries_doc1_query_faster(queries, batch_size=batch_size)
    
    
    def encode_queries_doc1_query_faster(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
            logger.info("embedding 1docs and 1query.......")
            # batch_size = 2
            
            orig_queries = queries[:]
            
            if dist.is_initialized():
                idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
            else:
                idx = range(len(queries))

            assert type(orig_queries[0])==list, '需要传入列表'
            queries = [orig_queries[i][0] for i in idx]
            pseudo_docs = [orig_queries[i][1] for i in idx]
            
            # import itertools

            # def flatten(list_of_lists):
            #     return list(itertools.chain.from_iterable(list_of_lists))

            # queries = flatten(queries) 
            
            if self.normalize_text:
                queries = [normalize_text.normalize(q) for q in queries]
                pseudo_docs = [normalize_text.normalize(q) for q in pseudo_docs]
            if self.lower_case:
                queries = [q.lower() for q in queries]
                pseudo_docs = [q.lower() for q in pseudo_docs]

            allemb = []
            nbatch = (len(queries) - 1) // batch_size + 1
            with torch.no_grad():
                for k in range(nbatch):
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, len(queries))
                    # import pdb
                    # pdb.set_trace()
                    qencode = self.tokenizer.batch_encode_plus(
                        queries[start_idx:end_idx],
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        add_special_tokens=self.add_special_tokens,
                        return_tensors="pt",
                    )
                    qencode = {key: value.cuda() for key, value in qencode.items()}
                    qemb = self.query_encoder(**qencode, normalize=self.norm_query)
                    
                    dencode = self.tokenizer.batch_encode_plus(
                        pseudo_docs[start_idx:end_idx],
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        add_special_tokens=self.add_special_tokens,
                        return_tensors="pt",
                    )
                    dencode = {key: value.cuda() for key, value in dencode.items()}
                    demb = self.query_encoder(**dencode, normalize=self.norm_query)
                    
                    
                    emb = qemb * 0.5 + demb * 0.5
                    # logger.info("emb shape after is: {}".format(emb.shape))
                    allemb.append(emb.cpu())
                    # logger.info("allemb before shape is: {}".format(len(allemb)))
                    # import pdb
                    # pdb.set_trace()

            allemb = torch.cat(allemb, dim=0)
            logger.info("allemb after shape is: {}".format(allemb.shape))
            allemb = allemb.cuda()
            if dist.is_initialized():
                allemb = dist_utils.varsize_gather_nograd(allemb)
            allemb = allemb.cpu().numpy()
            # pdb.set_trace()
            return allemb
        
       
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
            
            if dist.is_initialized():
                idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
            else:
                idx = range(len(corpus))
            corpus = [corpus[i] for i in idx]
            corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
            if self.normalize_text:
                corpus = [normalize_text.normalize(c) for c in corpus]
            if self.lower_case:
                corpus = [c.lower() for c in corpus]

            allemb = []
            nbatch = (len(corpus) - 1) // batch_size + 1
            with torch.no_grad():
                for k in range(nbatch):
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, len(corpus))

                    cencode = self.tokenizer.batch_encode_plus(
                        corpus[start_idx:end_idx],
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        add_special_tokens=self.add_special_tokens,
                        return_tensors="pt",
                    )
                    cencode = {key: value.cuda() for key, value in cencode.items()}
                    emb = self.doc_encoder(**cencode, normalize=self.norm_doc)
                    allemb.append(emb.cpu())

            allemb = torch.cat(allemb, dim=0)
            allemb = allemb.cuda()
            if dist.is_initialized():
                allemb = dist_utils.varsize_gather_nograd(allemb)
            allemb = allemb.cpu().numpy()
            return allemb

  
   


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
