from typing import Text
from dataclasses import dataclass
import random
from argparse import ArgumentParser

import wandb
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import jsonlines

from src.mbr_pipeline.utils.choose_dataset import get_dataset
from src.mbr_pipeline.args import Args, load_args
from src.mbr_pipeline.list_gen import sample
from src.mbr_pipeline.list_eval.evaluate import Metrics

assert __name__ == '__main__'

parser = ArgumentParser()
parser.add_argument("--config_file", type=str, help="file containing config for pipeline arguments", required=True)
parser.add_argument("--no_wandb", default=False, action="store_true", help="set this to avoid logging to wandb (will override config)", required=False)
parser.add_argument("--no_gpu", default=False, action="store_true", help="set this to avoid using GPU, e.g. for testing (will override config)", required=False)

setup = parser.parse_args()
args: Args = load_args(setup.config_file)

# update args from setup
args.pipeline.wandb = (not setup.no_wandb) and args.pipeline.wandb # use either flag to disable wandb
args.pipeline.no_gpu = args.pipeline.no_gpu or setup.no_gpu # use either flag to disable gpu
print(args)


# using PipelineArgs:
random.seed(args.PipelineArgs.seed)

if args.pipeline.wandb: 
    wandb.init(entity="gormleylab", project="lattice-decoding", name=args.pipeline.run_name, group=args.pipeline.wandb_group, config=args.to_dict())

device = torch.device("cuda" if (not args.pipeline.no_gpu) and torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(args.pipeline.hf_model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.pipeline.hf_tokenizer_name if args.pipeline.hf_tokenizer_name is not None else args.pipeline.hf_model_name)

# using DatasetArgs:
# get dataset
dataset = get_dataset(**args.dataset.__dict__)



from list_gen.sample import SamplingMethods, listgen

# using ListGenArgs:
match args.gen.method_args:
    # check if the data exists
    # add the option to regen anyway 
    case Args.ListGenArgs.LatticeMBRArgs() | Args.ListGenArgs.LatticeSamplingArgs():
        # check if the lattices exist

        # load the lattices

        # call the relevant func
        raise NotImplementedError()
    case Args.ListGenArgs.BeamSearchArgs():
        strategy_fn = SamplingMethods.beam_search
        method_name="beam"
    case Args.ListGenArgs.ModelSamplingArgs():
        # TODO: handle temp and nucl differently
        strategy_fn = SamplingMethods.nucl_sample
        method_name="nucl"
        raise NotImplementedError()

if args.gen.outfile is None:
    import os
    thisdir = ["sampling_outputs", args.dataset.dataset.name, args.pipeline.hf_model_name.replace("/","-"), str(args.gen.k)]
    constructed_path = ""
    for item in thisdir:
        constructed_path = os.path.join(constructed_path, item)
        if not os.path.exists(constructed_path):
            os.mkdir(constructed_path)

    print(args.gen.method_args.__dict__)

    str_args = "".join(f"{key}={args.gen.method_args.__dict__[key]}" for key in args.gen.method_args.__dict__)
    args.gen.outfile = os.path.join(constructed_path, f"{method_name}-{str_args}.jsonl")

sampling_outputs = listgen(strategy_fn=strategy_fn, model=model, tokenizer=tokenizer, dataset=dataset, \
    device=device, num_seqs=args.gen.k, max_length=args.gen.max_length, \
    strategy_args=args.gen.method_args.__dict__)


with jsonlines.open(args.gen.outfile, 'w') as f:
    f.write_all(sampling_outputs)


# reranking section

# evaluation section
metric_tracker = Metrics(args.eval.eval_metrics.split(",")) 


metrics_outputs = metric_tracker.score_set(sampling_outputs)
metric_tracker.output()

if args.eval.outfile is None:
    args.eval.outfile = args.gen.outfile
    
with jsonlines.open(args.eval.outfile, 'w') as f:
    f.write_all(metrics_outputs)