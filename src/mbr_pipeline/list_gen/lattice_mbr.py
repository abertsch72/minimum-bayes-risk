import torch
import numpy as np
import wandb
import jsonlines

from transformers import AutoTokenizer
from typing import List
from scipy.special import softmax, logsumexp

from lattice import Lattice
from mbr_pipeline.args import get_parser
from mbr_pipeline.utils.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mean_length(args, length_dict):
    lengths = np.array(sorted(length_dict.keys()))

    length_dist = np.array([length_dict[n][args.uniform_length] for n in lengths])
    if args.uniform_length:
        length_dist = length_dist / np.sum(length_dist)
    else:
        length_dist = softmax(length_dist / args.length_temp)
    mean_length = np.dot(length_dist, lengths)
    return mean_length

def get_partition_function(args, length_dict):
    '''
    Compute partition function (i.e. normalizing factor for probabilities)
    '''
    lengths = np.array(sorted(length_dict.keys()))
    vals = np.array([length_dict[n][args.uniform_match] for n in lengths])
    if args.uniform_match:
        return np.sum(vals)
    else: # log probs
        return 0.0
        # return logsumexp(vals) # log of partition function since we have lprobs

def normalize_probs(args, ngram_dict, Z):
    '''
    Z is normalizing constant
    '''
    if args.uniform_match:
        return {word: count / Z for word, (_, count) in ngram_dict.items()}
    else:
        return {word: np.exp((lprob - Z) / args.match_temp) 
                for word, (lprob, _) in ngram_dict.items()}

def get_match_dict(args, ngram_dict, Z):
    if "exact" in args.lattice_metric:
        return {k: normalize_probs(args, v, Z) for k, v in ngram_dict.items()}
    else:
        return normalize_probs(args, ngram_dict, Z)

def decode_hypos_from_lattice(args, lattice, tokenizer) -> List[str]:
    lattice.apply_temperature(args.lattice_score_temp)

    get_dict_fn, get_path_fn = get_lattice_methods(args, lattice)

    length_dict, all_node_length_dict = lattice.get_length_dict()
    mean_length = get_mean_length(args, length_dict)
    Z = get_partition_function(args, length_dict)

    if args.target_evidence_length == -1:
        args.evidence_length_deviation = float('inf')
    elif args.target_evidence_length == 0:
        args.target_evidence_length = mean_length

    ngram_dict, _ = get_dict_fn(
        all_node_length_dict, target_length=args.target_evidence_length,
        allowed_deviation=args.evidence_length_deviation
    )
    match_dict = get_match_dict(args, ngram_dict, Z)
    
    if args.k_per_node < 1:
        args.k_per_node = args.k
    if args.target_candidate_length == -1:
        args.candidate_length_deviation = float('inf')
    elif args.target_candidate_length == 0:
        args.target_candidate_length = mean_length

    topk_paths, *_ = get_path_fn(
        mean_length, 
        match_dict,
        target_length=args.target_candidate_length,
        d_length=args.candidate_length_deviation,
        lattice_topk=args.k_per_node,
        return_topk=args.k,
    )

    topk_hypos = tokenizer.batch_decode([
        lattice.get_path_tokens(path) for path in topk_paths
    ], skip_special_tokens=True)
    hypo_lprobs = [
        lattice.get_path_lprob(path) for path in topk_paths
    ]
    return topk_hypos, hypo_lprobs


def get_lattice_methods(args, lattice):
    use_exact = args.lattice_metric.startswith('exact')
    order = args.lattice_metric[-1:]
    if use_exact:
        assert order == "1"
        dict_method_name = "get_1gram_dict_by_length"
        path_method_name = "get_top_rouge1_path_exact"
    else:
        dict_method_name = f"get_{order}gram_dict"
        path_method_name = f"get_top_rouge{order}_path"
        if args.count_aware:
            dict_method_name += "_count_aware"
            path_method_name += "_count_aware"

    return getattr(lattice, dict_method_name), getattr(lattice, path_method_name)


def run_lattice_mbr(input_ids, latticedir, tokenizer, num_seqs, k, sample_uniform, max_len, no_repeats, lattice_score_temp):
  
    all_topk_hypos = []
    for lattice, output in Lattice.load_lattices(args.lattice_dir):
        topk_hypos, hypo_lprobs = decode_hypos_from_lattice(args, lattice, tokenizer)

        all_topk_hypos.append({
            "hypos": topk_hypos, 
            "lprobs": hypo_lprobs,
            "gold": output.reference,
        })

    with jsonlines.open(args.outfile, "w") as f:
        f.write_all(all_topk_hypos)

    if args.wandb:
        wandb.log({'topk': all_topk_hypos})


if __name__ == '__main__':
    parser = get_parser(latticembr=True)
    args = parser.parse_args()
    set_seed(args.seed)
    if args.wandb:
        *_, mbr_args = parser.parse_args_into_dataclasses()
        wandb.init(project='lattice-decoding', entity='gormleylab', 
                   group=args.wandb_group, config=vars(mbr_args))
        if args.run_name:
            wandb.run.name = args.run.name
    run_lattice_mbr(args)
