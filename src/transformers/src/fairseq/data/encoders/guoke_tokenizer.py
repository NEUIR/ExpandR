import re
from dataclasses import dataclass, field

import unicodedata

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass

TRANS_TABLE = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”—–-", u"'''\"\"---")])


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _handle_single_quote(tokens):
    line = ' '.join(tokens)
    line = re.sub(r"' ([smdSMDtT])\b", r"'\1", line)
    line = re.sub(r"' ll\b", "'ll", line)
    line = re.sub(r"' re\b", "'re", line)
    line = re.sub(r"' ve\b", "'ve", line)
    line = re.sub(r"' LL\b", "'LL ", line)
    line = re.sub(r"' RE\b", "'RE ", line)
    line = re.sub(r"' VE\b", "'VE ", line)
    return line.split()


def _split_on_cont_punc(tokens):
    new_tokens = []
    for token in tokens:
        if len(token) > 1:
            last_j = 0
            pre_is_punc = _is_punctuation(token[0])
            for j, ch in enumerate(token):
                is_punc = _is_punctuation(ch)
                if is_punc != pre_is_punc:
                    new_tokens.append(token[last_j: j])
                    last_j = j
                pre_is_punc = is_punc
            if last_j < len(token):
                new_tokens.append(token[last_j:])
        else:
            new_tokens.append(token)
    return new_tokens


def _split_pre_and_post_punc(tokens):
    def pre_punc(token):
        last_j = 0
        for j in range(1, len(token)):
            if not _is_punctuation(token[j]):
                last_j = j
                break
        return token[:last_j], token[last_j:]

    def post_punc(token):
        last_j = len(token)
        for j in range(len(token) - 2, -1, -1):
            if not _is_punctuation(token[j]):
                last_j = j + 1
                break
        return token[:last_j], token[last_j:]

    new_tokens = []
    for token in tokens:
        if len(token) > 1 and _is_punctuation(token[0]):
            a, b = pre_punc(token)
            if a:
                new_tokens.append(a)
            if b:
                if _is_punctuation(b[-1]):
                    c, d = post_punc(b)
                    if c:
                        new_tokens.append(c)
                    if d:
                        new_tokens.append(d)
                else:
                    new_tokens.append(b)
        elif len(token) > 1 and _is_punctuation(token[-1]):
            a, b = post_punc(token)
            if a:
                new_tokens.append(a)
            if b:
                new_tokens.append(b)
        else:
            new_tokens.append(token)
    return new_tokens


@dataclass
class GuokeTokenizerConfig(FairseqDataclass):
    lower: bool = field(default=False, metadata={"help": "convert to lower case"})


@register_tokenizer("guoke", dataclass=GuokeTokenizerConfig)
class GuokeTokenizer(object):
    def __init__(self, cfg: GuokeTokenizerConfig):
        self.cfg = cfg

    def encode(self, x: str) -> str:
        x = x.strip()
        x = x.replace("``", '"').replace("''", '"')
        x = x.translate(TRANS_TABLE)
        tokens = x.split()
        tokens = _split_pre_and_post_punc(tokens)
        tokens = _handle_single_quote(tokens)
        x = " ".join(tokens)
        if self.cfg.lower:
            x = x.lower()
        return x

    def decode(self, x: str) -> str:
        raise NotImplementedError()
