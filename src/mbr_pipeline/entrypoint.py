import os
import random
from argparse import ArgumentParser

import jsonlines
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import wandb
from src.mbr_pipeline.args import Args, load_args
from src.mbr_pipeline.list_eval.evaluate import Metrics
from src.mbr_pipeline.list_gen.lattice import Lattice
from src.mbr_pipeline.list_gen.lattice_mbr import decode_hypos_from_lattice
from src.mbr_pipeline.list_gen.lattice_sample import lattice_sample_k
from src.mbr_pipeline.list_gen.sample import SamplingMethods, listgen
from src.mbr_pipeline.reranking.rerank import Reranker
from src.mbr_pipeline.utils.choose_dataset import get_dataset


def pipeline(args):
    # using PipelineArgs:
    
    # fix all the seeds
    random.seed(args.pipeline.seed)
    np.random.seed(args.pipeline.seed)
    torch.random.manual_seed(args.pipeline.seed)

    if args.pipeline.wandb:
        wandb.init(
            entity="gormleylab",
            project="lattice-decoding",
            name=args.pipeline.run_name,
            group=args.pipeline.wandb_group,
            config=args.to_dict(),
        )

    device = torch.device(
        "cuda" if (not args.pipeline.no_gpu) and torch.cuda.is_available() else "cpu"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pipeline.hf_model_name).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pipeline.hf_tokenizer_name
        if args.pipeline.hf_tokenizer_name is not None
        else args.pipeline.hf_model_name
    )

    # using DatasetArgs:
    # get dataset
    dataset = get_dataset(**args.dataset.__dict__, seed=args.pipeline.seed)

    def get_lattices(lattice_dir: str):
        # check if the lattices exist
        if not os.path.exists(str(lattice_dir)):
            raise NotImplementedError()  # create the lattices

        # load the lattices
        lattices = Lattice.load_lattices(lattice_dir, no_tqdm=args.pipeline.no_tqdm)

        # todo: some sort of check to make sure we got the right # of lattices

        return lattices

    lattices = model = None
    # using ListGenArgs:
    match args.gen.method_args:
        # check if the data exists
        # add the option to regen anyway
        case Args.ListGenArgs.LatticeMBRArgs():
            lattices = get_lattices(args.pipeline.lattice_dir)
            method_name = "lattice_mbr"
            strategy_fn = decode_hypos_from_lattice

        case Args.ListGenArgs.LatticeSamplingArgs():
            lattices = get_lattices(args.pipeline.lattice_dir)
            method_name = "lattice_sampling"
            strategy_fn = lattice_sample_k

        case Args.ListGenArgs.BeamSearchArgs():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.pipeline.hf_model_name
            ).to(device)
            strategy_fn = SamplingMethods.beam_search
            method_name = "beam"

        case Args.ListGenArgs.ModelSamplingArgs() | _:
            # TODO: handle temp + nucl differently
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.pipeline.hf_model_name
            ).to(device)
            strategy_fn = SamplingMethods.model_sample
            method_name = "temp" if args.gen.method_args.temp != 1 else "top-p"

    if args.gen.outfile is None:
        thisdir = [
            "new-sampling_outputs",
            args.dataset.dataset.name,
            args.pipeline.hf_model_name.replace("/", "-"),
            str(args.gen.k),
        ]
        constructed_path = ""
        for item in thisdir:
            constructed_path = os.path.join(constructed_path, item)
            if not os.path.exists(constructed_path):
                os.mkdir(constructed_path)

        print(args.gen.method_args.__dict__)

        str_args = "".join(
            f"{key}={args.gen.method_args.__dict__[key]}"
            for key in args.gen.method_args.__dict__
        )
        fname = f"{method_name}-{str_args}.jsonl"

        if len(fname) > 255:  # max file name length on Mac & Linux
            # if file name is too long, abbreviate it
            def acronymize(name: str):
                return "".join(w[0] for w in name.split("_"))

            str_args = "".join(
                f"{acronymize(key)}{args.gen.method_args.__dict__[key]}"
                for key in args.gen.method_args.__dict__
            )
            fname = f"{method_name}-{str_args}.jsonl"

        args.gen.outfile = os.path.join(constructed_path, fname)

    if not os.path.exists(args.gen.outfile):
        sampling_outputs = listgen(
            strategy_fn=strategy_fn,
            model=model,
            lattices=lattices,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            num_seqs=args.gen.k,
            max_length=args.gen.max_length,
            strategy_args=args.gen.method_args.__dict__,
        )

        with jsonlines.open(args.gen.outfile, "w") as f:
            f.write_all(sampling_outputs)
    else:
        with jsonlines.open(args.gen.outfile, "r") as f:
            sampling_outputs = list(f.iter())

    # reranking section
    if args.rerank.rerank_metric is not None:
        reranker = Reranker(
            rerank_temp=args.rerank.rerank_temp,
            rerank_metric=args.rerank.rerank_metric,
            rerank_geo_mean=args.rerank.rerank_geo_mean,
        )

        rerank_metric = args.rerank.rerank_metric
        if "rouge" in rerank_metric and args.rerank.rerank_geo_mean:
            rerank_metric += "_geo"
        rerank_metric += f"temp-{args.rerank.rerank_temp}"

        for line in tqdm(sampling_outputs):
            scores_key = f"rerank_scores_{rerank_metric}"
            if scores_key in line:
                continue
            scores = reranker.rerank(line)
            line[scores_key] = scores

        with jsonlines.open(args.gen.outfile, "w") as f:
            f.write_all(sampling_outputs)

    # evaluation section
    if args.eval.eval_metrics is not None:
        metric_tracker = Metrics(args.eval.eval_metrics.split(","))

        metrics_outputs = metric_tracker.score_set(sampling_outputs)
        metric_tracker.output()

        if args.eval.outfile is None:
            args.eval.outfile = args.gen.outfile

        with jsonlines.open(args.eval.outfile, "w") as f:
            f.write_all(metrics_outputs)

        if args.pipeline.wandb:
            wandb.log(metrics_outputs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="file containing config for pipeline arguments",
        required=True,
    )
    parser.add_argument(
        "--no_wandb",
        default=False,
        action="store_true",
        help="set this to avoid logging to wandb (will override config)",
        required=False,
    )
    parser.add_argument(
        "--no_gpu",
        default=False,
        action="store_true",
        help="set this to avoid using GPU, e.g. for testing (will override config)",
        required=False,
    )

    setup = parser.parse_args()
    args: Args = load_args(setup.config_file)

    # update args from setup
    args.pipeline.wandb = (
        not setup.no_wandb
    ) and args.pipeline.wandb  # use either flag to disable wandb
    args.pipeline.no_gpu = (
        args.pipeline.no_gpu or setup.no_gpu
    )  # use either flag to disable gpu
    pipeline(args)
