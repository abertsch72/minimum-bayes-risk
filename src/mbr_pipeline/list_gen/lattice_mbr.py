from typing import List, Optional

import jsonlines
import numpy as np
import torch
from scipy.special import softmax

from src.mbr_pipeline.list_gen.lattice import Lattice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mean_length(length_dict, uniform_length: bool, length_temp: float):
    lengths = np.array(sorted(length_dict.keys()))

    length_dist = np.array([length_dict[n][uniform_length] for n in lengths])
    if uniform_length:
        length_dist = length_dist / np.sum(length_dist)
    else:
        length_dist = softmax(length_dist / length_temp)
    mean_length = np.dot(length_dist, lengths)
    return mean_length


def get_partition_function(length_dict, uniform_match: bool):
    """
    Compute partition function (i.e. normalizing factor for probabilities)
    """
    lengths = np.array(sorted(length_dict.keys()))
    vals = np.array([length_dict[n][uniform_match] for n in lengths])
    if uniform_match:
        return np.sum(vals)
    else:  # log probs
        return 0.0
        # return logsumexp(vals) # log of partition function since we have lprobs


def normalize_probs(ngram_dict, Z, uniform_match: bool, match_temp: float):
    """
    Z is normalizing constant
    """
    if uniform_match:
        return {word: count / Z for word, (_, count) in ngram_dict.items()}
    else:
        return {
            word: np.exp((lprob - Z) / match_temp)
            for word, (lprob, _) in ngram_dict.items()
        }


def get_match_dict(
    ngram_dict, Z: float, lattice_metric: str, uniform_match: bool, match_temp: float
):
    if "exact" in lattice_metric:
        return {
            k: normalize_probs(v, Z, uniform_match, match_temp)
            for k, v in ngram_dict.items()
        }
    else:
        return normalize_probs(ngram_dict, Z, uniform_match, match_temp)


def decode_hypos_from_lattice(
    lattice,
    tokenizer,
    num_seqs,
    output,
    max_length=float("inf"),
    lattice_metric: Optional[str] = None,
    uniform_length: Optional[bool] = None,
    length_temp: Optional[float] = None,
    uniform_match: Optional[bool] = None,
    match_temp=None,
    target_evidence_length=None,
    evidence_length_deviation=None,
    target_candidate_length=None,
    candidate_length_deviation=None,
    mean_override=None,
    lattice_score_temp=None,
    count_aware=None,
    k_per_node=None,
) -> List[str]:
    lattice.apply_temperature(lattice_score_temp)

    get_dict_fn, get_path_fn = get_lattice_methods(lattice, lattice_metric, count_aware)

    length_dict, all_node_length_dict = lattice.get_length_dict()
    mean_length = get_mean_length(length_dict, uniform_length, length_temp)
    Z = get_partition_function(length_dict, uniform_match)

    if target_evidence_length == -1:
        evidence_length_deviation = float("inf")
    elif target_evidence_length == 0:
        target_evidence_length = mean_length

    ngram_dict, _ = get_dict_fn(
        all_node_length_dict,
        target_length=target_evidence_length,
        allowed_deviation=evidence_length_deviation,
    )
    match_dict = get_match_dict(
        ngram_dict, Z, lattice_metric, uniform_match, match_temp
    )

    if k_per_node < 1:
        k_per_node = num_seqs
    if target_candidate_length == -1:
        candidate_length_deviation = float("inf")
    elif target_candidate_length == 0:
        target_candidate_length = mean_length

    topk_paths, *_ = get_path_fn(
        mean_length,
        match_dict,
        target_length=target_candidate_length,
        d_length=candidate_length_deviation,
        lattice_topk=k_per_node,
        return_topk=num_seqs,
    )

    topk_hypos = tokenizer.batch_decode(
        [lattice.get_path_tokens(path) for path in topk_paths], skip_special_tokens=True
    )
    hypo_lprobs = [lattice.get_path_lprob(path) for path in topk_paths]

    return {
        "gold": output.reference,
        "id": output.doc_id,
        "hypos": topk_hypos,
        "lprobs": hypo_lprobs,
        "num_unique": len(set(topk_hypos)),
    }


def get_lattice_methods(lattice, lattice_metric: str, count_aware: bool):
    use_exact = lattice_metric.startswith("exact")
    order = lattice_metric[-1:]
    if use_exact:
        assert order == "1"
        dict_method_name = "get_1gram_dict_by_length"
        path_method_name = "get_top_rouge1_path_exact"
    else:
        dict_method_name = f"get_{order}gram_dict"
        path_method_name = f"get_top_rouge{order}_path"
        if count_aware:
            dict_method_name += "_count_aware"
            path_method_name += "_count_aware"

    return getattr(lattice, dict_method_name), getattr(lattice, path_method_name)


"""
def run_lattice_mbr(
    input_ids,
    latticedir,
    tokenizer,
    num_seqs,
    k,
    sample_uniform,
    max_len,
    no_repeats,
    lattice_score_temp,
):
    all_topk_hypos = []
    for lattice, output in Lattice.load_lattices(args.lattice_dir):
        topk_hypos, hypo_lprobs = decode_hypos_from_lattice(args, lattice, tokenizer)

        all_topk_hypos.append(
            {
                "hypos": topk_hypos,
                "lprobs": hypo_lprobs,
                "gold": output.reference,
            }
        )

    with jsonlines.open(args.outfile, "w") as f:
        f.write_all(all_topk_hypos)

    if args.wandb:
        wandb.log({"topk": all_topk_hypos})

if __name__ == "__main__":
    parser = get_parser(latticembr=True)
    args = parser.parse_args()
    set_seed(args.seed)
    if args.wandb:
        *_, mbr_args = parser.parse_args_into_dataclasses()
        wandb.init(
            project="lattice-decoding",
            entity="gormleylab",
            group=args.wandb_group,
            config=vars(mbr_args),
        )
        if args.run_name:
            wandb.run.name = args.run.name
    run_lattice_mbr(args)
"""
