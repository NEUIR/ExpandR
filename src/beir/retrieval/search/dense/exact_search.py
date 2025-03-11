from .util import cos_sim, dot_score
import logging
import sys
import torch
from typing import Dict, List
import numpy as np
import faiss
import os
import torch
import pickle
import faiss
import jsonlines

logger = logging.getLogger(__name__)

# use faiss-gpu for retrieval
#Parent class for any dense model
class DenseRetrievalExactSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, dataname: str = None, **kwargs):
        # model is a class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.dataname = dataname
        self.score_functions = {'cos_sim': self.cosine_similarity, 'dot': self.dot_product_faiss}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True # TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.results = {}
        # Initialize faiss GPU resources
        self.res = faiss.StandardGpuResources()
    
    def cosine_similarity(self, query_embeddings, corpus_embeddings):
        faiss.normalize_L2(query_embeddings)
        faiss.normalize_L2(corpus_embeddings)
        index = faiss.IndexFlatIP(query_embeddings.shape[1])
        index = faiss.index_cpu_to_gpu(self.res, 0, index)  
        index.add(corpus_embeddings)
        distances, indices = index.search(query_embeddings, self.top_k)
        return distances, indices
    
    def dot_product_faiss(self, query_embeddings, corpus_embeddings):
        index = faiss.IndexFlatIP(query_embeddings.shape[1])  
        index = faiss.index_cpu_to_gpu(self.res, 0, index)  
        
        
        corpus_embeddings = corpus_embeddings.cpu().numpy().astype(np.float32) if isinstance(corpus_embeddings, torch.Tensor) else corpus_embeddings
        query_embeddings = query_embeddings.cpu().numpy().astype(np.float32) if isinstance(query_embeddings, torch.Tensor) else query_embeddings
        
        logger.info("Query embeddings shape:{}".format(query_embeddings.shape))
        logger.info("Corpus embeddings shape:{}".format(corpus_embeddings.shape))
        
        
        index.add(corpus_embeddings)
        distances, indices = index.search(query_embeddings, self.top_k)
        return distances, indices

    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               embeddings_file: str = "corpus_embeddings.npy",
               return_sorted: bool = False,
               **kwargs) -> Dict[str, Dict[str, float]]:
        
        self.top_k = top_k  # Store top_k for later use in similarity functions
        
        logger.info("Dataname is {}".format(self.dataname))
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("Score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
        
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]

        # Encode queries using the model's encode_queries method
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar)
        
        
        # Move query_embeddings to GPU
        query_embeddings = torch.tensor(query_embeddings).cuda()
        
        logger.info("Sorting Corpus by document length (Longest first)...")
        
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        if not self.dataname == 'cqadupstack':
            
            # Check if the embeddings file already exists
            if os.path.exists(embeddings_file):
                logger.info("Loading precomputed corpus embeddings from file...")
                corpus_embeddings = np.load(embeddings_file)
            else:
                logger.info("Encoding Corpus in batches... Warning: This might take a while!")
                logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
                
                itr = range(0, len(corpus), self.corpus_chunk_size)
                corpus_embeddings_list = []

                for batch_num, corpus_start_idx in enumerate(itr):
                    logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
                    corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

                    # Encode chunk of corpus    
                    sub_corpus_embeddings = self.model.encode_corpus(
                        corpus[corpus_start_idx:corpus_end_idx],
                        batch_size=self.batch_size,
                        # batch_size=2048,
                        show_progress_bar=self.show_progress_bar
                    )

                    corpus_embeddings_list.append(sub_corpus_embeddings)
                
                corpus_embeddings = np.concatenate(corpus_embeddings_list, axis=0)
                
                # Save the embeddings to file
                np.save(embeddings_file, corpus_embeddings)
        
        elif self.dataname == "cqadupstack":
            logger.info("Encoding Corpus in batches... Warning: This might take a while!")
            logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
            
            itr = range(0, len(corpus), self.corpus_chunk_size)
            corpus_embeddings_list = []

            for batch_num, corpus_start_idx in enumerate(itr):
                logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

                # Encode chunk of corpus    
                sub_corpus_embeddings = self.model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar
                )

                corpus_embeddings_list.append(sub_corpus_embeddings)
            
            corpus_embeddings = np.concatenate(corpus_embeddings_list, axis=0)
            
        
        logger.info("Successfully encoding Corpus!")
        # Move corpus_embeddings to GPU
        corpus_embeddings = torch.tensor(corpus_embeddings).cuda()

        # Compute similarities using either cosine-similarity or dot product
        distances, indices = self.score_functions[score_function](query_embeddings, corpus_embeddings)          
              
        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]
            self.results[query_id] = {}  # Initialize nested dictionary
            for sub_corpus_id, score in zip(indices[query_itr], distances[query_itr]):
                corpus_id = corpus_ids[sub_corpus_id]
                if corpus_id != query_id:
                    self.results[query_id][corpus_id] = float(score)  # Ensure score is float            
                    
        return self.results
    
    
    