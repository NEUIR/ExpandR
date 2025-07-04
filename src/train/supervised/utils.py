from transformers import AutoConfig, AutoTokenizer
from transformers.models.t5 import FairseqT5Tokenizer



def load_stuff(model_args, data_args, use_fast=False):
    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    if model_args.use_converted:
        tokenizer = FairseqT5Tokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=use_fast,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=use_fast,
        )
    
    return config, tokenizer