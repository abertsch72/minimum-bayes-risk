from typing import Text
from dataclasses import dataclass
import random
from argparse import ArgumentParser
import os

from tqdm import tqdm
import wandb
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import jsonlines

from src.mbr_pipeline.utils.choose_dataset import get_dataset
from src.mbr_pipeline.args import Args, load_args
from src.mbr_pipeline.list_gen.sample import SamplingMethods, listgen
from src.mbr_pipeline.list_eval.evaluate import Metrics
from src.mbr_pipeline.reranking.rerank import Reranker
from src.mbr_pipeline.list_gen.lattice import Lattice
from src.mbr_pipeline.list_gen.lattice_sample import lattice_sample_k

def pipeline(args: Args):
    # using PipelineArgs:
    random.seed(args.pipeline.seed)

    if args.pipeline.wandb: 
        wandb.init(entity="gormleylab", project="lattice-decoding", name=args.pipeline.run_name, group=args.pipeline.wandb_group, config=args.to_dict())

    device = torch.device("cuda" if (not args.pipeline.no_gpu) and torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pipeline.hf_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pipeline.hf_tokenizer_name if args.pipeline.hf_tokenizer_name is not None else args.pipeline.hf_model_name)

    # using DatasetArgs:
    # get dataset
    dataset = get_dataset(**args.dataset.__dict__, seed=args.pipeline.seed)

    def get_lattices(lattice_dir: str):
        # check if the lattices exist
        if not os.path.exists(str(args.pipeline.lattice_dir)):
            raise NotImplementedError() # create the lattices
        
        # load the lattices
        lattices = Lattice.load_lattices(args.pipeline.lattice_dir, no_tqdm=args.pipeline.no_tqdm)

        # todo: some sort of check to make sure we got the right # of lattices

        return lattices
        
    # using ListGenArgs:
    match args.gen.method_args:
        # check if the data exists
        # add the option to regen anyway 
        case Args.ListGenArgs.LatticeMBRArgs(): 
            lattices = get_lattices()
            method_name="lattice_mbr"
            strategy_fn = None
            raise NotImplementedError()

        case Args.ListGenArgs.LatticeSamplingArgs():
            lattices = get_lattices()
            method_name="lattice_sampling"
            strategy_fn = lattice_sample_k
            raise NotImplementedError()

        case Args.ListGenArgs.BeamSearchArgs():
            strategy_fn = SamplingMethods.beam_search
            method_name="beam"

        case Args.ListGenArgs.ModelSamplingArgs() | _:
            # TODO: handle temp + nucl differently
            strategy_fn = SamplingMethods.model_sample
            method_name="temp" if args.gen.method_args.temp != 1 else "top-p"

    if args.gen.outfile is None:
        thisdir = ["new-sampling_outputs", args.dataset.dataset.name, args.pipeline.hf_model_name.replace("/","-"), str(args.gen.k)]
        constructed_path = ""
        for item in thisdir:
            constructed_path = os.path.join(constructed_path, item)
            if not os.path.exists(constructed_path):
                os.mkdir(constructed_path)

        print(args.gen.method_args.__dict__)

        str_args = "".join(f"{key}={args.gen.method_args.__dict__[key]}" for key in args.gen.method_args.__dict__)
        args.gen.outfile = os.path.join(constructed_path, f"{method_name}-{str_args}.jsonl")

    if not os.path.exists(args.gen.outfile):
        sampling_outputs = listgen(strategy_fn=strategy_fn, model=model, tokenizer=tokenizer, dataset=dataset, \
            device=device, num_seqs=args.gen.k, max_length=args.gen.max_length, \
            strategy_args=args.gen.method_args.__dict__)


        with jsonlines.open(args.gen.outfile, 'w') as f:
            f.write_all(sampling_outputs)
    else:
        with jsonlines.open(args.gen.outfile, 'r') as f:
            sampling_outputs = list(f.iter())
        

    # reranking section
    reranker = Reranker(rerank_temp = args.rerank.rerank_temp, rerank_metric = args.rerank.rerank_metric, \
        rerank_geo_mean=args.rerank.rerank_geo_mean) 


    rerank_metric = args.rerank.rerank_metric
    if "rouge" in rerank_metric and args.rerank.rerank_geo_mean:
        rerank_metric += "_geo"
    rerank_metric += f"temp-{args.rerank.rerank_temp}"

    for line in tqdm(sampling_outputs):
        scores_key = f'rerank_scores_{rerank_metric}'
        if scores_key in line:
            continue
        scores = reranker.rerank(line)
        line[scores_key] = scores

    with jsonlines.open(args.gen.outfile, "w") as f:
        f.write_all(sampling_outputs)

    # evaluation section
    metric_tracker = Metrics(args.eval.eval_metrics.split(",")) 

    metrics_outputs = metric_tracker.score_set(sampling_outputs)
    wandb_metrics = metric_tracker.output()

    if args.eval.outfile is None:
        args.eval.outfile = args.gen.outfile

    with jsonlines.open(args.eval.outfile, 'w') as f:
        f.write_all(metrics_outputs)

    wandb.log(wandb_metrics) 


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, help="file containing config for pipeline arguments", required=True)
    parser.add_argument("--no_wandb", default=False, action="store_true", help="set this to avoid logging to wandb (will override config)", required=False)
    parser.add_argument("--no_gpu", default=False, action="store_true", help="set this to avoid using GPU, e.g. for testing (will override config)", required=False)

    setup = parser.parse_args()
    args: Args = load_args(setup.config_file)

    # update args from setup
    args.pipeline.wandb = (not setup.no_wandb) and args.pipeline.wandb # use either flag to disable wandb
    args.pipeline.no_gpu = args.pipeline.no_gpu or setup.no_gpu # use either flag to disable gpu
    pipeline(args)
