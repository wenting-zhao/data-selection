import argparse
import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import HfArgumentParser, DataCollatorForSeq2Seq
from dataclasses import dataclass, field

import evaluate
from evaluate import logging

from arguments import ModelArguments, DataArguments
from get_training_dataset import get_training_dataset
from get_validation_dataset import get_dataset

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)
    model = torch.compile(model)

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
        use_chat_format=True,
        chat_format="other",
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
