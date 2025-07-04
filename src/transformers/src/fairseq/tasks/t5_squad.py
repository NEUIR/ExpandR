import json
import logging
import os
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from omegaconf import MISSING, II

from fairseq import metrics, squad_utils, utils
from fairseq.data import (
    data_utils,
    ConcatSentencesDataset,
    ConstantDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadShiftDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task

logger = logging.getLogger(__name__)
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])


@dataclass
class T5SquadConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    no_shuffle: bool = field(
        default=False,
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed tokens_per_sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
                    'e.g., "train,valid" (default: all dataset splits)'
        },
    )

    seed: int = II("common.seed")

    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )

    # prompt
    init_token: Optional[int] = field(
        default=None,
        metadata={"help": "add token at the beginning of each batch item"},
    )
    separator_token: Optional[int] = field(
        default=None,
        metadata={"help": "add separator token between inputs"},
    )
    prefix_question: Optional[str] = field(
        default=None,
        metadata={"help": "prefix to prepend before question"}
    )
    prefix_context: Optional[str] = field(
        default=None,
        metadata={"help": "prefix to prepend before context"}
    )
    no_ans: str = field(
        default="",
        metadata={"help": "prediction string for no answer"}
    )

    # eval
    eval_generate: bool = field(
        default=False,
        metadata={
            "help": "evaluation generated sequence with metrics (acc/f1/mcc/corr)"
        }
    )
    eval_generate_args: str = field(
        default="{}",
        metadata={
            "help": "args for generation as a JSON string"
        }
    )
    eval_generate_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    eval_generate_fix_dashes: bool = field(
        default=False, metadata={"help": "apply post processing rules to fix dashes (need text context)"}
    )


@register_task('t5_squad', dataclass=T5SquadConfig)
class T5SquadTask(FairseqTask):
    cfg: T5SquadConfig
    dictionary: Dictionary

    def __init__(self, cfg: T5SquadConfig, dictionary):
        super().__init__(cfg)

        self.dictionary = dictionary
        self.seed = cfg.seed
        self.tokenizer = None
        self.bpe = None
        self.qa_id = {}
        self.text_context = {}
        self.eval_answer = {}

    @classmethod
    def setup_task(cls, cfg: T5SquadConfig, **kwargs):
        # load data dictionary
        dictionary = cls.load_dictionary(
            os.path.join(cfg.data, 'context', 'dict.txt')
        )
        # add mask token
        dictionary.add_symbol('<sen000>')
        for i in range(1, cfg.tokens_per_sample):
            dictionary.add_symbol(f'<sen{i:03d}>')
        logger.info("[input] dictionary: {} types".format(len(dictionary)))

        return cls(cfg, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        def get_path(key, split):
            return os.path.join(self.cfg.data, key, split)

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            try:
                dataset = data_utils.load_indexed_dataset(
                    split_path,
                    dictionary,
                    combine=combine,
                )
            except Exception as e:
                if "StorageException: [404] Path not found" in str(e):
                    logger.warning(f"dataset {e} not found")
                    dataset = None
                else:
                    raise e
            return dataset

        question = make_dataset("question", self.source_dictionary)
        assert question is not None
        context = make_dataset("context", self.source_dictionary)
        assert context is not None
        answer = make_dataset("answer", self.source_dictionary)
        assert answer is not None

        if self.cfg.prefix_question is not None:
            prefix_q = self.bpe.encode(self.cfg.prefix_question)
            prefix_q = self.source_dictionary.encode_line(prefix_q, add_if_not_exist=False, append_eos=False)
            question = ConcatSentencesDataset(ConstantDataset(prefix_q.tolist(), len(question)), question)
        if self.cfg.prefix_context is not None:
            prefix_c = self.bpe.encode(self.cfg.prefix_context)
            prefix_c = self.source_dictionary.encode_line(prefix_c, add_if_not_exist=False, append_eos=False)
            context = ConcatSentencesDataset(ConstantDataset(prefix_c.tolist(), len(context)), context)

        if self.cfg.init_token is not None:
            question = PrependTokenDataset(question, self.cfg.init_token)
        if self.cfg.separator_token is not None:
            context = PrependTokenDataset(context, self.cfg.separator_token)

        src_tokens = ConcatSentencesDataset(question, context)

        if split.startswith("train"):
            with data_utils.numpy_seed(self.cfg.seed):
                shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        tgt_dataset = RightPadDataset(
            answer,
            pad_idx=self.source_dictionary.pad()
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True)
        }
        if tgt_dataset is not None:
            dataset["target"] = tgt_dataset
            dataset["net_input"]["prev_output_tokens"] = PadShiftDataset(
                tgt_dataset,
                pad_idx=self.source_dictionary.pad(),
                start_idx=self.source_dictionary.eos()
            )
            dataset["sample_size"] = NumelDataset(tgt_dataset, reduce=True)

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes]
        )

        qa_id = torch.load(get_path("qa_id", split) + ".pt")
        assert len(qa_id) == len(dataset)
        text_context = None
        if not split.startswith("train"):
            if self.cfg.eval_generate_fix_dashes:
                text_context = torch.load(get_path("text_context", split) + ".pt")
                assert len(text_context) == len(dataset)
            eval_answer = torch.load(get_path("eval_answer", split) + ".pt")
            assert len(eval_answer) == len(dataset)
        else:
            eval_answer = None

        if split.startswith("train") and not self.cfg.no_shuffle:
            dataset = SortDataset(
                dataset,
                sort_order=[shuffle],
            )
            qa_id = [qa_id[idx] for idx in np.lexsort([shuffle])]

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))
        self.datasets[split] = dataset
        self.qa_id[split] = qa_id
        if text_context is not None:
            assert not self.text_context
            self.text_context[split] = text_context
        if eval_answer is not None:
            # make sure that `valid` is the only validation subset
            assert not self.eval_answer
            self.eval_answer[split] = eval_answer
        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True)
        }
        return NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes]
        )

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_generate:
            gen_args = json.loads(self.cfg.eval_generate_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_generate:
            pred_strs = self.inference(self.sequence_generator, sample, model)
            tot_em, tot_f1 = 0, 0
            for i in range(len(pred_strs)):
                # make sure that `valid` is the only validation subset
                em, f1 = squad_utils.get_raw_score_single(
                    self.eval_answer['valid'][sample["id"][i].item()],
                    pred_strs[i]
                )
                tot_em += em
                tot_f1 += f1
            logging_output["_em_sum"] = tot_em
            logging_output["_f1_sum"] = tot_f1
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_generate:
            if len(logging_outputs) > 0:
                em_sum = sum(log.get("_em_sum", 0) for log in logging_outputs)
                f1_sum = sum(log.get("_f1_sum", 0) for log in logging_outputs)
                total = sum(log.get("nsentences", 0) for log in logging_outputs)
                metrics.log_scalar_sum("_em_sum", em_sum)
                metrics.log_scalar_sum("_f1_sum", f1_sum)
                metrics.log_scalar_sum("_total", total)
                metrics.log_derived(
                    "em",
                    lambda meters: round(
                        utils.item(meters["_em_sum"].sum) * 100.0 / utils.item(meters["_total"].sum), 3
                    )
                    if meters["_total"].sum > 0
                    else 0,
                )
                metrics.log_derived(
                    "f1",
                    lambda meters: round(
                        utils.item(meters["_f1_sum"].sum) * 100.0 / utils.item(meters["_total"].sum), 3
                    )
                    if meters["_total"].sum > 0
                    else 0,
                )

    def max_positions(self):
        return self.cfg.tokens_per_sample

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def inference(self, generator, sample, model, with_id=False):
        def decode(toks):
            s = self.dictionary.string(toks.int().cpu())
            if self.bpe:
                s = self.bpe.decode(s)
            # we do not detokenize because it's not necessary for squad eval
            s = " ".join(s.strip().split())
            return s

        def fix_dashes(pred, context):
            def get_dash_spans(s):
                return [
                    (m.start(), s[max(m.start() - 1, 0): min(m.end() + 1, len(s))])
                    for m in re.finditer(r'[\-–]', s)
                ]

            if "-" not in pred or "–" not in context:
                return pred
            context_dash_spans = set(s for _, s in get_dash_spans(context))
            pred_arr = list(pred)
            for idx, s in get_dash_spans(pred):
                if pred_arr[idx] == '-':
                    ss = s.replace('-', '–')
                    if s not in context_dash_spans and ss in context_dash_spans:
                        pred_arr[idx] = '–'
            return ''.join(pred_arr)

        def longest_common_substring(s1, s2):
            p = 1000000007
            q = 1000000009
            x = 100003

            def get_substr_hash(s, k):
                x_neginvp = (p - pow(x, k - 1, p)) % p
                x_neginvq = (q - pow(x, k - 1, q)) % q
                hv1, hv2 = 0, 0
                res = {}
                for i in range(len(s)):
                    if i >= k:
                        hv1 = (hv1 + ord(s[i - k]) * x_neginvp) % p
                        hv2 = (hv2 + ord(s[i - k]) * x_neginvq) % q
                    hv1 = (hv1 * x + ord(s[i])) % p
                    hv2 = (hv2 * x + ord(s[i])) % q
                    if i >= k - 1:
                        res[(hv1, hv2)] = i - k + 1
                return res

            def k_common_substr(s1, s2, k):
                dic1 = get_substr_hash(s1, k)
                dic2 = get_substr_hash(s2, k)
                merged_keys = list(set(dic1.keys()) & set(dic2.keys()))
                if merged_keys:
                    return s1[dic1[merged_keys[0]]: dic1[merged_keys[0]] + k]
                else:
                    return None

            l, r = 0, min(len(s1), len(s2)) + 1
            ans = []
            while l + 1 < r:
                mid = (l + r) // 2
                r_mid = k_common_substr(s1, s2, mid)
                if r_mid is not None:
                    l = mid
                    ans = r_mid
                else:
                    r = mid
            return ans

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        pred_strs = []
        for i in range(len(gen_out)):
            pred_str = decode(gen_out[i][0]["tokens"])
            context_str = decode(sample["net_input"]["src_tokens"][i][:sample["net_input"]["src_lengths"][i].item()])
            lcs_context = longest_common_substring(pred_str, context_str)
            lcs_noans = longest_common_substring(pred_str, self.cfg.no_ans)
            if len(lcs_context) <= len(lcs_noans):
                pred_str = ""
            else:
                pred_str = lcs_context
            if self.cfg.eval_generate_fix_dashes:
                pred_str = fix_dashes(pred_str, self.text_context["valid"][sample["id"][i].item()])
            pred_strs.append(pred_str)
        if self.cfg.eval_generate_print_samples:
            logger.info("example prediction string: " + pred_strs[0])
            if "target" in sample:
                logger.info("example reference string: " + decode(sample["target"][0]))
        if with_id:
            return [
                (self.qa_id["valid"][sample["id"][i].item()], pred_str)
                for i, pred_str in enumerate(pred_strs)
            ]
        else:
            return pred_strs
