import re
import subprocess
from typing import Union, List

import filelock
import torch.distributed as dist


def whitespace_fix(s):
    return ' '.join(s.strip().split())


def ptb_tokenize(ss: Union[str, List[str]]):
    if isinstance(ss, str):
        return ptb_tokenize([ss])[0]
    proc = subprocess.Popen(
        ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdin = "".join(s + "\n" for s in ss).encode("utf-8")
    stdout, stderr = proc.communicate(stdin)
    tokenized = stdout.decode("utf-8").strip().splitlines()
    assert len(ss) == len(tokenized), f"{ss}\n\n{tokenized}\n\n{stderr.decode('utf-8')}"
    return [whitespace_fix(s) for s in tokenized]


def compute_rouge_f(hyp: Union[str, List[str]], ref: Union[str, List[str]]):
    if isinstance(hyp, str):
        assert isinstance(ref, str)
        return compute_rouge_f([hyp], [ref])
    hyp_tok = ptb_tokenize(hyp)
    ref_tok = ptb_tokenize(ref)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    with filelock.FileLock(f"rouge.{rank}.lock"):
        with open(f"rouge_hypothesis.{rank}.tok", "w", encoding="utf-8") as f:
            for line in hyp_tok:
                f.write(line + "\n")
        with open(f"rouge_references.{rank}.tok", "w", encoding="utf-8") as f:
            for line in ref_tok:
                f.write(line + "\n")
        proc = subprocess.Popen(
            ['files2rouge', f'rouge_references.{rank}.tok', f'rouge_hypothesis.{rank}.tok'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate()
        rouge_output = stdout.decode("utf-8")
        try:
            matches = re.findall(r'ROUGE-1 Average_F: ([0-9\.]+)', rouge_output)
            assert len(matches) == 1
            rouge_1_f = float(matches[0])
            matches = re.findall(r'ROUGE-2 Average_F: ([0-9\.]+)', rouge_output)
            assert len(matches) == 1
            rouge_2_f = float(matches[0])
            matches = re.findall(r'ROUGE-L Average_F: ([0-9\.]+)', rouge_output)
            assert len(matches) == 1
            rouge_l_f = float(matches[0])
        except (AssertionError, ValueError):
            raise AssertionError(
                f"rouge_out: {rouge_output}"
                f"files2rouge err msg: {stderr.decode('utf-8')}"
            )
    return {
        "rouge_1_f": rouge_1_f,
        "rouge_2_f": rouge_2_f,
        "rouge_l_f": rouge_l_f,
    }
