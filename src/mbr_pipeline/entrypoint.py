from choose_dataset import DatasetArgs, get_dataset
from simple_parsing import ArgumentParser
from typing import Text
from dataclasses import dataclass

parser = ArgumentParser()
parser.add_arguments(DatasetArgs, dest="dataset")
args = parser.parse_args()

# get dataset
dataset = get_dataset(args.dataset)


@dataclass
class ModelArgs:
    model_name: Text
    tokenizer_name: Text = model_name

# sample
samples = sample()
