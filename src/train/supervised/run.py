import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)


from supervised.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from supervised.data import TrainDatasetForEmbedding, EmbedCollator
# from modeling import BiEncoderModel
from supervised.modeling import DenseModel
from supervised.utils import load_stuff
from supervised.trainer import BiTrainer
from torch.utils.data import random_split
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    
    config, tokenizer = load_stuff(model_args, data_args)
    logger.info('Config: %s', config)
    model = DenseModel.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )


    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    train_size = int(0.95 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = random_split(train_dataset, [train_size, test_size])

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
