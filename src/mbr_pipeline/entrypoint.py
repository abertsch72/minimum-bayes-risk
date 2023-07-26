import os
import random
from argparse import ArgumentParser

import jsonlines
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    MarianMTModel,
)

import wandb
from src.mbr_pipeline.args import Args, load_args
from src.mbr_pipeline.list_eval.evaluate import Metrics
from src.mbr_pipeline.list_gen.lattice import Lattice
from src.mbr_pipeline.list_gen.lattice_mbr import decode_hypos_from_lattice
from src.mbr_pipeline.list_gen.lattice_sample import lattice_sample_k
from src.mbr_pipeline.list_gen.sample import SamplingMethods, listgen
from src.mbr_pipeline.list_gen.sampler_with_scores import SamplerWithScoresMixin
from src.mbr_pipeline.list_gen.stochastic_beam_search import add_mixin, get_sbs_mixin

# from src.mbr_pipeline.list_gen.wide_beam_search import WideBeamSearchMixin
from src.mbr_pipeline.reranking.rerank import Reranker
from src.mbr_pipeline.utils.choose_dataset import get_dataset


def get_base_model_cls(dataset):
    flores_datasets = [
        Args.DatasetArgs.SupportedDataset.flores,
        Args.DatasetArgs.SupportedDataset.flores_isl,
    ]
    if dataset in flores_datasets:
        return MarianMTModel
    return BartForConditionalGeneration


def pipeline(args: Args):
    # using PipelineArgs:

    # fix all the seeds
    random.seed(args.pipeline.seed)
    np.random.seed(args.pipeline.seed)
    torch.random.manual_seed(args.pipeline.seed)

    print(args)

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
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.pipeline.hf_model_name).to(
    #     device
    # )
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
    # match args.gen.method_args:

    # check if the data exists
    # add the option to regen anyway
    if isinstance(args.gen.method_args, Args.ListGenArgs.LatticeMBRArgs):
        lattices = get_lattices(args.pipeline.lattice_dir)
        method_name = "lattice_mbr"
        strategy_fn = decode_hypos_from_lattice

    elif isinstance(args.gen.method_args, Args.ListGenArgs.LatticeSamplingArgs):
        lattices = get_lattices(args.pipeline.lattice_dir)
        method_name = "lattice_sampling"
        strategy_fn = lattice_sample_k

    elif isinstance(args.gen.method_args, Args.ListGenArgs.BeamSearchArgs):
        strategy_fn = SamplingMethods.beam_search
        MODEL_CLS = AutoModelForSeq2SeqLM
        if (
            args.gen.method_args.num_beam_groups > 1
            and args.gen.method_args.diversity_penalty > 0.0
        ):
            method_name = "diverse_beam"
        elif args.gen.method_args.stochastic:
            method_name = "stochastic_beam_search"
            strategy_fn = SamplingMethods.stochastic_beam_search
            BASE_MODEL_CLS = get_base_model_cls(args.dataset.dataset)
            MODEL_CLS = add_mixin(
                BASE_MODEL_CLS, get_sbs_mixin(args.gen.method_args.memoryless)
            )
        else:
            # if args.gen.method_args.num_beams > 100:
            #     BASE_MODEL_CLS = get_base_model_cls(args.dataset.dataset)
            method_name = "beam"

        model = MODEL_CLS.from_pretrained(args.pipeline.hf_model_name).to(device)

    else:
        # elif args.gen.method_args == Args.ListGenArgs.ModelSamplingArgs():
        # TODO: handle temp + nucl differently
        BASE_MODEL_CLS = get_base_model_cls(args.dataset.dataset)
        MODEL_CLS = add_mixin(BASE_MODEL_CLS, SamplerWithScoresMixin)
        model = MODEL_CLS.from_pretrained(args.pipeline.hf_model_name).to(device)
        strategy_fn = SamplingMethods.model_sample
        method_name = "temp" if args.gen.method_args.top_p == 1.0 else "top-p"

    print(method_name)

    if args.gen.outfile is None:
        thisdir = [
            args.pipeline.save_directory, #"test-outputs",
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

    print(args.gen.outfile)
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
            unique_k=args.gen.unique_k,
            strategy_args=args.gen.method_args.__dict__,
        )

        with jsonlines.open(args.gen.outfile, "w") as f:
            f.write_all(sampling_outputs)
    else:
        with jsonlines.open(args.gen.outfile, "r") as f:
            sampling_outputs = list(f.iter())

    # reranking section
    reevaluate = False
    if args.rerank.rerank_metric is not None:
        reranker = Reranker(
            rerank_temp=args.rerank.rerank_temp,
            rerank_metric=args.rerank.rerank_metric,
            rerank_geo_mean=args.rerank.rerank_geo_mean,
            rank_by_freq=args.rerank.rank_by_freq,
            importance_sample=args.rerank.importance_sample,
        )

        rerank_metric = args.rerank.rerank_metric
        if "rouge" in rerank_metric and args.rerank.rerank_geo_mean:
            rerank_metric += "_geo"
        if args.rerank.rank_by_freq:
            rerank_metric += "_freq"
        else:
            rerank_metric += "_logprobs"
        rerank_metric += f"temp-{args.rerank.rerank_temp}"

        evidence_outputs = None
        if args.rerank.evidence_set_file is not None:
            rerank_metric += f"_{args.rerank.evidence_set_file}"

            with jsonlines.open(args.rerank.evidence_set_file, "r") as f:
                evidence_outputs = list(f.iter())

            if args.rerank.num_evidence is not None:
                for out in evidence_outputs:
                    out["hypos"] = out["hypos"][: args.rerank.num_evidence]
                    out["lprobs"] = out["lprobs"][: args.rerank.num_evidence]
                rerank_metric += f"_first{args.rerank.num_evidence}"

        print("Rerank metric:", rerank_metric)

        for idx, line in enumerate(tqdm(sampling_outputs)):
            scores_key = f"rerank_scores_{rerank_metric}"
            if scores_key in line:
                eval_key = f"top_rerank_{rerank_metric}"
                if eval_key not in line:
                    reevaluate = True
                continue
            else:
                reevaluate = True
            evidence_set = None
            if evidence_outputs is not None:
                evidence_set = evidence_outputs[idx]
            scores = reranker.rerank(line, evidence_set)
            line[scores_key] = scores

        with jsonlines.open(args.gen.outfile, "w") as f:
            f.write_all(sampling_outputs)

    # evaluation section
    if args.eval.eval_metrics is not None:
        if args.eval.outfile is None:
            args.eval.outfile = args.gen.outfile
        table_outfile = f"{args.eval.outfile}.txt"

        if os.path.isfile(table_outfile) and not reevaluate:
            with open(table_outfile, "r") as f:
                print(f.read())
        else:
            metric_tracker = Metrics(args.eval.eval_metrics.split(","))

            metrics_outputs = metric_tracker.score_set(sampling_outputs)
            metric_tracker.output()

            with jsonlines.open(args.eval.outfile, "w") as f:
                f.write_all(metrics_outputs)

            with open(table_outfile, "w+") as f:
                metric_tracker.output(outfile=f)

            if args.pipeline.wandb:
                wandb.log(metric_tracker.to_dict())


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
