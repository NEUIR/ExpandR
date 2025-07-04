import torch
import torch.distributed as dist
import numpy as np
from typing import List, Dict

import dist_utils as dist_utils
import normalize_text

class DenseEncoderModel:
    def __init__(self, query_encoder, doc_encoder=None, tokenizer=None,
                 max_length=512, add_special_tokens=True,
                 norm_query=False, norm_doc=False,
                 lower_case=False, normalize_text=False):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder or query_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text

    def _preprocess(self, texts: List[str]) -> List[str]:
        if self.normalize_text:
            texts = [normalize_text.normalize(t) for t in texts]
        if self.lower_case:
            texts = [t.lower() for t in texts]
        return texts


    def _encode_batches(self, texts: List[str], encoder_fn, batch_size: int) -> np.ndarray:
        allemb = []

        if dist.is_initialized():
            idx = np.array_split(range(len(texts)), dist.get_world_size())[dist.get_rank()]
            texts = [texts[i] for i in idx]

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokenized = self.tokenizer.batch_encode_plus(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                add_special_tokens=self.add_special_tokens,
                return_tensors="pt",
            )
            tokenized = {k: v.cuda() for k, v in tokenized.items()}

            with torch.no_grad():
                output = encoder_fn(tokenized)
                if isinstance(output, tuple):
                    output = output[1]  # 只取 q_reps 或 p_reps
                output = torch.nn.functional.normalize(output, dim=-1)  # ✅ 归一化
            allemb.append(output.cpu())

        embs = torch.cat(allemb, dim=0).cuda()
        if dist.is_initialized():
            embs = dist_utils.varsize_gather_nograd(embs)

        return embs.cpu().numpy()




    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        queries = self._preprocess(queries)
        # return self._encode_batches(queries, self.query_encoder.encode_query, batch_size)
        return self._encode_queries_doc1_query_faster(queries, batch_size)

    def _encode_queries_doc1_query_faster(self, queries: List[List[str]], batch_size: int, **kwargs) -> np.ndarray:
        """
        queries: List of [query, pseudo_doc]
        Returns: averaged embeddings
        """

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
            queries = [queries[i] for i in idx]

        query_texts = self._preprocess([q[0] for q in queries])
        pseudo_docs = self._preprocess([q[1] for q in queries])
        allemb = []

        with torch.no_grad():
            for i in range(0, len(query_texts), batch_size):
                q_batch = query_texts[i:i+batch_size]
                d_batch = pseudo_docs[i:i+batch_size]

                q_tok = self.tokenizer.batch_encode_plus(
                    q_batch, max_length=self.max_length, padding=True, truncation=True,
                    add_special_tokens=self.add_special_tokens, return_tensors="pt"
                )
                d_tok = self.tokenizer.batch_encode_plus(
                    d_batch, max_length=self.max_length, padding=True, truncation=True,
                    add_special_tokens=self.add_special_tokens, return_tensors="pt"
                )

                q_tok = {k: v.cuda() for k, v in q_tok.items()}
                d_tok = {k: v.cuda() for k, v in d_tok.items()}

                _, q_emb = self.query_encoder.encode_query(q_tok)
                _, d_emb = self.doc_encoder.encode_passage(d_tok)
                
                q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
                d_emb = torch.nn.functional.normalize(d_emb, dim=-1)

                emb = (q_emb + d_emb) * 0.5
                allemb.append(emb.cpu())

        embs = torch.cat(allemb, dim=0).cuda()
        if dist.is_initialized():
            embs = dist_utils.varsize_gather_nograd(embs)
        return embs.cpu().numpy()

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        docs = [c["title"] + " " + c["text"] if c["title"] else c["text"] for c in corpus]
        docs = self._preprocess(docs)
        return self._encode_batches(docs, self.doc_encoder.encode_passage, batch_size)
