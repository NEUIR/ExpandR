import json
import os
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, BatchEncoding, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


from typing import Optional, Dict

from evaluate_anchor.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
import logging

logger = logging.getLogger(__name__)


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True
    ):
        super(LinearPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q[:, 0])
        elif p is not None:
            return self.linear_p(p[:, 0])
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, 'pooler.pt')
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'pooler.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class DenseModel(nn.Module):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = self.lm_p.config # make deepspeed happy

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):

        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)

        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # if self.training:
        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)
            p_reps = self.dist_gather_tensor(p_reps)

        effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
            if self.train_args.negatives_x_device \
            else self.train_args.per_device_train_batch_size

        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        

        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )
        target = target * self.data_args.train_n_passages
        
        loss = self.cross_entropy(scores, target)

        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction
        return DenseOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps
        )

        

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode_passage(self, psg):
        if psg is None:
            return None, None
        psg = BatchEncoding(psg)
        if self.model_args.use_t5:
            decoder_input_ids = torch.zeros((psg.input_ids.shape[0], 1), dtype=torch.long)
            decoder_input_ids = decoder_input_ids.to(psg.input_ids.device)
            psg_out = self.lm_p(**psg, decoder_input_ids=decoder_input_ids, return_dict=True)
            p_hidden = psg_out.encoder_last_hidden_state
            p_reps = p_hidden.mean(dim=1)
        elif self.model_args.use_t5_decoder:
            decoder_input_ids = torch.zeros((psg.input_ids.shape[0], 1), dtype=torch.long)
            decoder_input_ids = decoder_input_ids.to(psg.input_ids.device)
            psg_out = self.lm_p(**psg, decoder_input_ids=decoder_input_ids, return_dict=True)
            p_hidden = psg_out.last_hidden_state
            p_reps = p_hidden[:, 0, :]
        else:
            psg_out = self.lm_p(**psg, return_dict=True)
            p_hidden = psg_out.last_hidden_state
            if self.pooler is not None:
                p_reps = self.pooler(p=p_hidden)  # D * d
            elif self.model_args.use_mean_pooler:
                p_reps = self.mean_pooling(psg_out[0], psg.attention_mask)
            else:
                p_reps = p_hidden[:, 0]
        return p_hidden, p_reps

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry = BatchEncoding(qry)
        if self.model_args.use_t5:
            decoder_input_ids = torch.zeros((qry.input_ids.shape[0], 1), dtype=torch.long)
            decoder_input_ids = decoder_input_ids.to(qry.input_ids.device)
            qry_out = self.lm_q(**qry, decoder_input_ids=decoder_input_ids, return_dict=True)
            q_hidden = qry_out.encoder_last_hidden_state
            q_reps = q_hidden.mean(dim=1)
        elif self.model_args.use_t5_decoder:
            decoder_input_ids = torch.zeros((qry.input_ids.shape[0], 1), dtype=torch.long)
            decoder_input_ids = decoder_input_ids.to(qry.input_ids.device)
            qry_out = self.lm_q(**qry, decoder_input_ids=decoder_input_ids, return_dict=True)
            q_hidden = qry_out.last_hidden_state
            q_reps = q_hidden[:, 0, :]
        else:
            qry_out = self.lm_q(**qry, return_dict=True)
            q_hidden = qry_out.last_hidden_state
            if self.pooler is not None:
                q_reps = self.pooler(q=q_hidden)
            elif self.model_args.use_mean_pooler:
                q_reps = self.mean_pooling(qry_out[0], qry.attention_mask)
            else:
                q_reps = q_hidden[:, 0]
        return q_hidden, q_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        if self.model_args.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DenseModelForInference(DenseModel):
    POOLER_CLS = LinearPooler

    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.model_args = model_args

    @torch.no_grad()
    def encode_passage(self, psg):
        return super(DenseModelForInference, self).encode_passage(psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return super(DenseModelForInference, self).encode_query(qry)

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)
        return DenseOutput(q_reps=q_reps, p_reps=p_reps)

    @classmethod
    def build(
            cls,
            model_name_or_path: str = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path

        # load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
                # print(lm_q)
                #你推理不是正常的吗，你在跑下推理代码，再这个地方看看模型的type
                # ok？
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.POOLER_CLS(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            model_args=model_args
        )
        return model
    
    
@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 model_args: ModelArguments = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True                 
                 ):
        super().__init__()
        # self.model = AutoModel.from_pretrained(model_name)
        self.lm_p = AutoModel.from_pretrained(model_name)
        self.lm_q = self.lm_p
        self.model_args = model_args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.lm_p.config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        import pdb
        pdb.set_trace()
    def gradient_checkpointing_enable(self, **kwargs):
        # self.model.gradient_checkpointing_enable(**kwargs)
        self.lm_p.gradient_checkpointing_enable(**kwargs)
        self.lm_q.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode_passage(self, psg):
        if psg is None:
            return None, None
        psg = BatchEncoding(psg)
        if self.model_args.use_t5:
            decoder_input_ids = torch.zeros((psg.input_ids.shape[0], 1), dtype=torch.long)
            decoder_input_ids = decoder_input_ids.to(psg.input_ids.device)
            psg_out = self.lm_p(**psg, decoder_input_ids=decoder_input_ids, return_dict=True)
            p_hidden = psg_out.encoder_last_hidden_state
            p_reps = p_hidden.mean(dim=1)
        elif self.model_args.use_t5_decoder:
            decoder_input_ids = torch.zeros((psg.input_ids.shape[0], 1), dtype=torch.long)
            decoder_input_ids = decoder_input_ids.to(psg.input_ids.device)
            psg_out = self.lm_p(**psg, decoder_input_ids=decoder_input_ids, return_dict=True)
            p_hidden = psg_out.last_hidden_state
            p_reps = p_hidden[:, 0, :]
        else:
            psg_out = self.lm_p(**psg, return_dict=True)
            p_hidden = psg_out.last_hidden_state
            if self.pooler is not None:
                p_reps = self.pooler(p=p_hidden)  # D * d
            elif self.model_args.use_mean_pooler:
                p_reps = self.mean_pooling(psg_out[0], psg.attention_mask)
            else:
                p_reps = p_hidden[:, 0]
        return p_hidden, p_reps

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry = BatchEncoding(qry)
        if self.model_args.use_t5:
            decoder_input_ids = torch.zeros((qry.input_ids.shape[0], 1), dtype=torch.long)
            decoder_input_ids = decoder_input_ids.to(qry.input_ids.device)
            qry_out = self.lm_q(**qry, decoder_input_ids=decoder_input_ids, return_dict=True)
            q_hidden = qry_out.encoder_last_hidden_state
            q_reps = q_hidden.mean(dim=1)
        elif self.model_args.use_t5_decoder:
            decoder_input_ids = torch.zeros((qry.input_ids.shape[0], 1), dtype=torch.long)
            decoder_input_ids = decoder_input_ids.to(qry.input_ids.device)
            qry_out = self.lm_q(**qry, decoder_input_ids=decoder_input_ids, return_dict=True)
            q_hidden = qry_out.last_hidden_state
            q_reps = q_hidden[:, 0, :]
        else:
            qry_out = self.lm_q(**qry, return_dict=True)
            q_hidden = qry_out.last_hidden_state
            if self.pooler is not None:
                q_reps = self.pooler(q=q_hidden)
            elif self.model_args.use_mean_pooler:
                q_reps = self.mean_pooling(qry_out[0], qry.attention_mask)
            else:
                q_reps = q_hidden[:, 0]
        return q_hidden, q_reps

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_hidden, q_reps = self.encode_query(query)
        q_reps = torch.nn.functional.normalize(q_reps, dim=-1)
        p_hidden, p_reps = self.encode_passage(passage)
        p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        # import pdb
        # pdb.set_trace()
        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps, p_reps.view(q_reps.size(0), group_size, -1)) / self.temperature # B G
                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    

    def save(self, output_dir: str):
        if self.model_args.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)