from pathlib import Path
import re
import torch
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import altair as alt

from arguments import ModelArguments, DataArguments
from get_training_dataset import get_training_dataset
from get_validation_dataset import get_dataset


def get_logprobs(pattern):
    # pattern = "logprobs-valid*.pt"
    # pattern = "mean-logprobs-valid*.pt"
    DIR = Path("results/")

    logprobs_by_problem = []
    problem_idx = []
    for file_path in DIR.glob(pattern):
        idx = int(re.findall(r"\d+", str(file_path))[0])
        logprobs = torch.load(file_path)

        problem_idx.append(idx)
        logprobs_by_problem.append(logprobs)

    idxs = torch.tensor(problem_idx)
    logprobs = torch.tensor(logprobs_by_problem)

    return logprobs, idxs


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

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

    train_lengths = torch.tensor([x["input_ids"].shape[0] for x in train_dataset])
    valid_lengths = torch.tensor([len(x["input_ids"]) for x in analysis_dataset])

    pattern = "logprobs-valid*.pt"
    # pattern = "mean-logprobs-valid*.pt"
    logprobs, val_idxs = get_logprobs(pattern)

    df = pd.DataFrame({"Length": train_lengths, "LogProb": logprobs[10]})
    chart = (
        alt.Chart(df.sample(n=5000, random_state=1234))
        .mark_point()
        .encode(x="Length", y="LogProb", tooltip=["Length", "LogProb"])
        .properties(
            width=600,
            height=400,
            title="Correlation between training example Length and LogProb",
        )
    )
    chart.save("figures/length-logprob-corr.png")

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
