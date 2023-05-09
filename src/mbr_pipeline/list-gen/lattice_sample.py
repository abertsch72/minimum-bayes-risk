import os
from mbr_pipeline.args import get_parser
from lattice import Lattice
import numpy as np
from transformers import AutoTokenizer
import random
from scipy.special import softmax, logsumexp
import jsonlines
from tqdm import tqdm
import wandb
from utils import set_seed


def sample_path(lattice: Lattice, uniform=False, max_len=float('inf'), temp=1.0):
    path = [lattice.sos]
    total_lprob = 0.0
    while path[-1] not in lattice.eos_set and len(path) < max_len:
        curr_node = path[-1]
        neighbors = lattice.edges[curr_node]
        if uniform:
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


def sample_k(lattice, tokenizer, k, uniform=False, max_len=float('inf'), 
             no_repeats=False, temp=1.0):
    topk_hypos = dict()
    count = 0
    while count < k:
        path, hypo_lprob = sample_path(lattice, uniform, max_len, temp)
        hypo = tokenizer.decode(lattice.get_path_tokens(path), skip_special_tokens=True)
        hypo_hash = hash(hypo)
        if hypo_hash in topk_hypos: # hopefully hash collisions don't happen
            if no_repeats:
                continue
            _, old_lprob = topk_hypos[hypo_hash]
            topk_hypos[hypo_hash] = (hypo, logsumexp([old_lprob, hypo_lprob]))
        else:
            topk_hypos[hypo_hash] = (hypo, hypo_lprob)
        count += 1
    return list(topk_hypos.values())


def run_lattice_sampling(args):
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name, local_files_only=True)
    
    all_hypos = []
    for lattice, output in Lattice.load_lattices(args.lattice_dir, no_tqdm=args.no_tqdm):
        topk_hypos = sample_k(
            lattice, tokenizer, 
            k=args.k, 
            uniform=args.sample_uniform,
            max_len=args.max_len,
            no_repeats=args.no_repeats,
            temp=args.lattice_score_temp
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
            "doc_id": output.doc_id
        })

    with jsonlines.open(args.outfile, "w") as f:
        f.write_all(all_hypos)

    if args.wandb:
        wandb.log({"topk": all_hypos})


if __name__ == "__main__":
    parser = get_parser(latticesamp=True)
    args = parser.parse_args()
    set_seed(args.seed)
    if args.wandb:
        *_, ls_args = parser.parse_args_into_dataclasses()
        wandb.init(project='lattice-decoding', entity='gormleylab', 
                   group=args.wandb_group, config=vars(ls_args))
        if args.run_name:
            wandb.run.name = args.run_name
    run_lattice_sampling(args)
