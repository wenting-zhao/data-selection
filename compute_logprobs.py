import argparse
import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import HfArgumentParser, DataCollatorForSeq2Seq
from dataclasses import dataclass, field
from typing import List, Optional, Union

import evaluate
from evaluate import logging

from get_training_dataset import get_training_dataset
from get_validation_dataset import get_dataset

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DataArguments:
    train_files: List[str] = field(
        default_factory=list,
        metadata={
            "help": "The input training data files (multiple files in glob format)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            )
        },
    )
    sample_data_seed: int = field(
        default=42,
        metadata={"help": ("The seed used for data sampling.")},
    )
    batch_size: int = field(
        default=16,
        metadata={"help": ("Batch size for inference.")},
    )
    percentage: float = field(
        default=1.0,
        metadata={"help": ("Sampling percentage for each dataset")},
    )
    analysis_dataset: str = field(
        default="mmlu",
        metadata={"help": ("The dataset to use for analysis mode. ")},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = get_training_dataset(
        data_args.train_files,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        sample_percentage=data_args.percentage,
        seed=data_args.sample_data_seed,
    )
    analysis_dataset = get_dataset(
        data_args.analysis_dataset,
        data_dir="./data",
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        use_chat_format=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
    )
    tokenizer.padding_side = "left"

    loss_fct = CrossEntropyLoss(reduction="none")
    for i, one in enumerate(analysis_dataset):
        concated = []
        l = len(one["input_ids"])
        for j, two in tqdm(
            enumerate(train_dataset),
            desc=f"creating dataloader for validation example {i}",
        ):
            input_ids = two["input_ids"].tolist() + one["input_ids"]
            if len(input_ids) > data_args.max_seq_length:
                input_ids = input_ids[-data_args.max_seq_length :]
            d = {"input_ids": input_ids}
            concated.append(d)
        concated = datasets.Dataset.from_list(concated)
        dataloader = DataLoader(
            concated, collate_fn=data_collator, batch_size=data_args.batch_size
        )
        logprobs = []
        mean_logprobs = []
        for batch in tqdm(dataloader):
            for key in batch:
                batch[key] = batch[key].to(device)
            labels = batch["input_ids"].clone().detach()
            labels[:, :-l] = -100

            with torch.no_grad():
                out_logits = model(**batch, labels=labels).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            logprob = -loss_fct(shift_logits.transpose(1, 2), shift_labels).sum(1)
            mean_logprob = logprob / l

            logprobs += logprob.tolist()
            mean_logprobs += mean_logprob.tolist()
        torch.save(logprobs, f"results/logprobs-valid{i}.pt")
        torch.save(mean_logprobs, f"results/mean-logprobs-valid{i}.pt")


if __name__ == "__main__":
    main()
