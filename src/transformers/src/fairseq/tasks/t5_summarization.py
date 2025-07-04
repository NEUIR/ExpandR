# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional

import torch
from omegaconf import II

from fairseq import metrics, utils, summarization_utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


def load_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        max_source_positions,
        max_target_positions,
        truncate_source=False,
        truncate_target=False,
        append_source_id=False,
        shuffle=True,
        pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())

        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            if truncate_target:
                tgt_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(tgt_dataset, tgt_dict.eos()),
                        max_target_positions - 1,
                    ),
                    tgt_dict.eos(),
                )
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=False,
        left_pad_target=False,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=0,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class T5SummarizationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
                    "in round-robin manner; however, valid and test data are always in the first directory "
                    "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default="src",
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default="tgt",
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    truncate_target: bool = field(
        default=False, metadata={"help": "truncate source to max-target-positions"}
    )
    train_subset: str = II("dataset.train_subset")
    valid_subset: str = II("dataset.valid_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
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


@register_task("t5_summarization", dataclass=T5SummarizationConfig)
class T5SummarizationTask(FairseqTask):
    cfg: T5SummarizationConfig

    def __init__(self, cfg: T5SummarizationConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.tokenizer = None
        self.bpe = None
        self.raw_target = {}

    @classmethod
    def setup_task(cls, cfg: T5SummarizationConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        # add mask token
        src_dict.add_symbol('<sen000>')
        for i in range(1, cfg.tokens_per_sample):
            src_dict.add_symbol(f'<sen{i:03d}>')
        tgt_dict.add_symbol('<sen000>')
        for i in range(1, cfg.tokens_per_sample):
            tgt_dict.add_symbol(f'<sen{i:03d}>')
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            max_source_positions=self.cfg.tokens_per_sample,
            max_target_positions=self.cfg.tokens_per_sample,
            truncate_source=self.cfg.truncate_source,
            truncate_target=self.cfg.truncate_target,
            shuffle=(split == "train"),
            pad_to_multiple=8,
        )

        if not split.startswith("train") and self.cfg.eval_generate:
            raw_target = torch.load(os.path.join(data_path, "raw_tgt", split + ".pt"))
            assert len(raw_target) == len(self.datasets[split])
            assert not self.raw_target
            self.raw_target[split] = raw_target

        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg, from_checkpoint=False):
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
            ref_strs = [
                self.raw_target[self.cfg.valid_subset][sample["id"][i].item()]
                for i in range(len(pred_strs))
            ]
            rouge_f_scores = summarization_utils.compute_rouge_f(
                pred_strs, ref_strs
            )
            for key in rouge_f_scores.keys():
                logging_output[f"_{key}_sum"] = rouge_f_scores[key] * len(pred_strs)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_generate:
            if len(logging_outputs) > 0:
                total = sum(log.get("nsentences", 0) for log in logging_outputs)
                metrics.log_scalar_sum("_total", total)
                for key in ["rouge_1_f", "rouge_2_f", "rouge_l_f"]:
                    rouge_f_sum = sum(log.get(f"_{key}_sum", 0) for log in logging_outputs)
                    metrics.log_scalar_sum(f"_{key}_sum", rouge_f_sum)
                    metrics.log_derived(
                        key,
                        lambda meters, key=key: round(
                            utils.item(meters[f"_{key}_sum"].sum) * 100.0 / utils.item(meters["_total"].sum), 3
                        )  # use default value to capture the value instead of the reference of `key`
                        if meters["_total"].sum > 0
                        else 0,
                    )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.cfg.tokens_per_sample

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def inference(self, generator, sample, model, with_id=False):
        def decode(toks):
            s = self.source_dictionary.string(toks.int().cpu())
            if self.bpe:
                s = self.bpe.decode(s)
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        pred_strs = []
        for i in range(len(gen_out)):
            pred_str = decode(gen_out[i][0]["tokens"])
            pred_strs.append(pred_str)
        if self.cfg.eval_generate_print_samples:
            logger.info("example prediction string: " + pred_strs[0])
            if "target" in sample:
                logger.info("example reference string: " + decode(sample["target"][0]))
        if with_id:
            return (
                pred_strs,
                [sample["id"][i].item() for i in range(len(gen_out))]
            )
        else:
            return pred_strs
