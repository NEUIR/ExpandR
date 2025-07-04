import os
import warnings
from shutil import copyfile
from typing import List, Optional

from fairseq.data import encoders, Dictionary
from omegaconf import DictConfig

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "sp_path": "sp.model",
    "dict_path": "dict.txt"
}


class FairseqT5Tokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        sp_path,
        dict_path,
        lower,
        n_sentinel_tokens=0,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        **kwargs
    ) -> None:

        self.sp_path = sp_path
        self.dict_path = dict_path
        self.lower = lower

        self.fs_tokenizer = encoders.build_tokenizer(
            DictConfig(dict(
                _name="guoke",
                lower=lower
            ))
        )
        self.fs_bpe = encoders.build_bpe(
            DictConfig(dict(
                _name="sentencepiece",
                sentencepiece_model=sp_path,
            ))
        )
        self.fs_dict = Dictionary.load(dict_path)
        for i in range(n_sentinel_tokens):
            self.fs_dict.add_symbol(f'<sen{i:03d}>')

        if "sep_token" in kwargs:
            assert kwargs["sep_token"] == eos_token
            kwargs.pop("sep_token")
        if "cls_token" in kwargs:
            assert kwargs["cls_token"] == bos_token
            kwargs.pop("cls_token")

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=eos_token,
            cls_token=bos_token,
            lower=self.lower,
            n_sentinel_tokens=n_sentinel_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.fs_dict)

    def get_vocab(self):
        return self.fs_dict.indices

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def _tokenize(self, text: str) -> List[str]:
        return self.fs_bpe.encode(self.fs_tokenizer.encode(text)).split(" ")

    def _convert_token_to_id(self, token):
        return self.fs_dict.index(token)

    def _convert_id_to_token(self, index):
        return self.fs_dict[index]

    def convert_tokens_to_string(self, tokens):
        return self.fs_bpe.decode(" ".join(tokens))

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_sp_path = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["sp_path"]
        )
        out_dict_path = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["dict_path"]
        )

        if os.path.abspath(self.sp_path) != os.path.abspath(out_sp_path):
            copyfile(self.sp_path, out_sp_path)
            logger.info(f"Copy from {self.sp_path} to {out_sp_path}")
        if os.path.abspath(self.dict_path) != os.path.abspath(out_dict_path):
            copyfile(self.dict_path, out_dict_path)
            logger.info(f"Copy from {self.dict_path} to {out_dict_path}")

        return out_sp_path, out_dict_path
