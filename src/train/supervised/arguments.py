import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments



@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    use_t5: bool = field(
        default=False,
        metadata={"help": "Whether to use T5 model"}
    )

    use_t5_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to use T5 decoder"}
    )
    use_converted: bool = field(
        default=False,
        metadata={"help": "Whether to use model from fs"}
    )
    use_mean_pooler: bool = field(
        default=False,
        metadata={"help": "Whether to use mean pooler"}
    )
    vocab_path: str=field(
        default=None
    )

    # for trainning with hard negative
    iter_num: Optional[int]=field(
        default=0, metadata={"help": "Iteration of hard negative generation, used to decay learning rate"}
    )

    decay_rate: Optional[float]=field(
        default=0.6, metadata={"help": "Decay learning rate"}
    )



@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "Freeze the parameters of position embeddings"})
