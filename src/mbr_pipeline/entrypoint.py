from typing import Text
from dataclasses import dataclass
import random
from argparse import ArgumentParser

import wandb
from transformers import AutoModel, AutoTokenizer

from choose_dataset import DatasetArgs, get_dataset
from args import Args, load_args

parser = ArgumentParser()
parser.add_argument("--config_file", type=str, help="file containing config for pipeline arguments", required=True)
parser.add_argument("--no_wandb", default=False, action="store_true", help="set this to avoid logging to wandb (will override config)", required=False)

setup = parser.parse_args()

args: Args = load_args(setup.config_file)
print(args)
# using PipelineArgs:
args.pipeline.wandb = (not setup.no_wandb)

print(args.pipeline.wandb)
if args.pipeline.wandb: 
    wandb.init(entity="gormleylab", project="lattice-decoding", name=args.pipeline.run_name, group=args.pipeline.wandb_group, config=args.to_dict())

random.seed(args.PipelineArgs.seed)

model = AutoModel.from_pretrained(args.pipeline.hf_model_name)
tokenizer = AutoTokenizer.from_pretrained(args.pipeline.hf_tokenizer_name if args.pipeline.hf_tokenizer_name is not None else args.pipeline.hf_model_name)

# using DatasetArgs:
# get dataset
dataset = get_dataset(**args.dataset.__dict__)
from typing import Text
from dataclasses import dataclass
import random
from argparse import ArgumentParser

import wandb
from transformers import AutoModel, AutoTokenizer

from choose_dataset import DatasetArgs, get_dataset
from args import Args, load_args

parser = ArgumentParser()
parser.add_argument("--config_file", type=str, help="file containing config for pipeline arguments", required=True)
parser.add_argument("--no_wandb", default=False, action="store_true", help="set this to avoid logging to wandb (will override config)", required=False)

setup = parser.parse_args()

args: Args = load_args(setup.config_file)
print(args)
# using PipelineArgs:
args.pipeline.wandb = (not setup.no_wandb)

print(args.pipeline.wandb)
if args.pipeline.wandb: 
    wandb.init(entity="gormelylab", project="lattice-decoding", name=args.pipeline.run_name, group=args.pipeline.wandb_group, config=args.to_dict())

random.seed(args.PipelineArgs.seed)

model = AutoModel.from_pretrained(args.pipeline.hf_model_name)
tokenizer = AutoTokenizer.from_pretrained(args.pipeline.hf_tokenizer_name if args.pipeline.hf_tokenizer_name is not None else args.pipeline.hf_model_name)

# using DatasetArgs:
# get dataset
dataset = get_dataset(**args.dataset.__dict__)
