import argparse
from datasets import load_dataset
from enum import Enum

from dataclasses import dataclass
from datargs import parse


class SupportedDataset(Enum):
    samsum = ["samsum"]
    cnndm = ["cnn_dailymail", "3.0.0"]
    xsum = ["xsum"]

class DataSplit(Enum):
    val = "validation"
    test = "test"
    train = "train"

@dataclass 
class DatasetArgs:
    dataset: SupportedDataset
    split: DataSplit
    start_index: int = 0
    end_index: int = -1

def get_input():
    args = parse(DatasetArgs)
    print(args)


def get_dataset(args: DatasetArgs):
    full_data = load_dataset(*args.dataset.value, split=args.split)
    if args.end_index != -1:
        return full_data.select(range(args.start_index, args.end_index))
    elif args.start_index != 0:
        return full_data.select(range(args.start_index, len(full_data)))
    else:
        return full_data

def test():
    args = DatasetArgs(dataset=SupportedDataset.samsum, split="test", end_index=351)
    print(get_dataset(args))

#test()

