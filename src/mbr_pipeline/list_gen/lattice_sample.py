import os
import random

import jsonlines
import numpy as np
from scipy.special import logsumexp, softmax
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from mbr_pipeline.list_gen.lattice import Lattice
from src.mbr_pipeline.utils.utils import set_seed


def sample_path(lattice: Lattice, sample_uniform=False, max_len=float("inf"), temp=1.0):
    path = [lattice.sos]
    total_lprob = 0.0
    while path[-1] not in lattice.eos_set and len(path) < max_len:
        curr_node = path[-1]
        neighbors = lattice.edges[curr_node]
        if sample_uniform:
            options = list(neighbors)
            next_node = random.sample(options, 1)[0]
        else:
            options, lprobs = [], np.empty(len(neighbors))
            for idx, (n, lp) in enumerate(neighbors.items()):
                options.append(n)
                lprobs[idx] = lp
            if temp != 1.0:
                lprobs /= temp
            probs = softmax(lprobs)
            next_node = np.random.choice(options, p=probs)
        total_lprob += lattice.edges[curr_node][next_node]
        path.append(next_node)
    return path, total_lprob


def lattice_sample_k(
    lattice,
    tokenizer,
    num_seqs,
    output,
    max_length=float("inf"),
    sample_uniform=False,
    no_repeats=False,
    lattice_score_temp=1.0,
):
    topk_hypos = {}
    count = 0

    gold = output.reference
    id = output.doc_id
    while count < num_seqs:
        path, hypo_lprob = sample_path(
            lattice, sample_uniform, max_length, lattice_score_temp
        )
        hypo = tokenizer.decode(lattice.get_path_tokens(path), skip_special_tokens=True)

        hypo_hash = hash(hypo)
        if hypo_hash in topk_hypos:  # hopefully hash collisions don't happen
            if no_repeats:
                continue
            _, old_lprob = topk_hypos[hypo_hash]
            topk_hypos[hypo_hash] = (hypo, logsumexp([old_lprob, hypo_lprob]))
        else:
            topk_hypos[hypo_hash] = (hypo, hypo_lprob)
        count += 1

    hypos = [v[0] for v in topk_hypos.values()]
    lprobs = [v[1] for v in topk_hypos.values()]

    num_unique = len(set(hypos))
    return {
        "gold": gold,
        "id": id,
        "hypos": hypos,
        "lprobs": lprobs,
        "num_unique": num_unique,
    }


"""
def run_lattice_sampling(lattices, k: int, sample_sample_uniform: bool, max_len: int, no_repeats: bool, lattice_score_temp: float, tokenizer: AutoTokenizer):

    all_hypos = []
    for lattice, output in lattices:
        topk_hypos = sample_k(
            lattice, tokenizer,
            k=k,
            sample_uniform=sample_sample_uniform,
            max_len=max_len,
            no_repeats=no_repeats,
            temp=lattice_score_temp
        )
        hypos = [v[0] for v in topk_hypos]
        lprobs = [v[1] for v in topk_hypos]
        # sort from greatest to least
        sorted_order = np.argsort(lprobs)[::-1]
        hypos = [hypos[i] for i in sorted_order]
        lprobs = [lprobs[i] for i in sorted_order]
        all_hypos.append({
            "hypos": hypos,
            "lprobs": lprobs, # LOG probabilities
            "gold": output.reference,
            "id": output.doc_id,
            "num_unique": len(set(hypos))
        })

    return all_hypos
"""
