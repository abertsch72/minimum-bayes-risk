import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random
import json
from functools import lru_cache
from collections import deque
from pprint import pprint

from lattice import Lattice

import sys
sys.path.append("./")
sys.path.append("./src/")

from rouge_score import rouge_scorer
from transformers import AutoTokenizer

from src.recom_search.evaluation.analysis import derive_path
from src.recom_search.model.exec_setup import args

###################################################
# Short helper functions
###################################################

def read_result(dir, file):
    with open(os.path.join(dir, file), 'rb') as f:
        x = pickle.load(f)
    return x    

def get_all_paths(graph_ends):
    all_nodes, all_edges = get_graph(graph_ends)

    flag_sum = True
    return derive_path(all_nodes, all_edges, flag_sum)[0]

def get_graph(graph_ends):
    all_nodes = {}
    all_edges = {}
    for end in graph_ends:
        nodes, edges = end.visualization()
        all_nodes.update(nodes)
        all_edges.update(edges)
    return all_nodes, all_edges

def main():
    if args.outfile is None:
        args.outfile = f"mbr_result_rouge{args.rouge}_dlen={args.d_length}"
        args.outfile += '_unif' if args.uniform else ''
        args.outfile += '_ca' if args.count_aware else ''
        args.outfile += '.json'
    print(args.outfile)
    import pdb; pdb.set_trace()

    global log_json
    log_json = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
    full_rouge_scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Results using best-first search with recombination + zip
    results_dir = "output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9"
    result_files = os.listdir(results_dir)

    get_ngram_dict_method_name = f"get_{args.rouge}gram_dict{'_count_aware' if args.count_aware else ''}"
    get_top_path_method_name = f"get_top_rouge{args.rouge}_path{'_count_aware' if args.count_aware else''}"
    
    i = 0
    for file in tqdm(result_files):
        output = read_result(results_dir, file)
        
        if output.output is not None: # using beam search
            raise Exception("must use lattice, cannot use beam search")

        # using bfs or bfs+recomb
        graph_data = get_graph(output.ends)
        lattice = Lattice(*graph_data)
        
        # get length dict + compute (un)weighted mean length E[|h|]
        length_dict, all_node_length_dict = lattice.get_length_dict_reverse_dfs()
        total_num_paths = sum(num for (_, num) in length_dict.values())

        lengths = sorted(length_dict.keys())
        length_dist_unnorm = np.exp([length_dict[n][0] for n in lengths])
        length_dist = length_dist_unnorm / np.sum(length_dist_unnorm)
        avg_len_weighted = np.sum(length_dist * lengths)

        avg_len_unweighted = 0
        for length, (lprob, count) in length_dict.items():
            avg_len_unweighted += length * count / total_num_paths

        # get n-gram match dictionary
        get_ngram_dict_fn = getattr(lattice, get_ngram_dict_method_name)
        ngram_dict, all_node_ngram_dict = get_ngram_dict_fn(all_node_length_dict)

        match_unweighted = {word: count / total_num_paths for word, (_, count) in ngram_dict.items()}
        match_weighted = {word: np.exp(lprob) for word, (lprob, _) in ngram_dict.items()}

        mean_length = avg_len_unweighted if args.uniform else avg_len_weighted
        expected_match = match_unweighted if args.uniform else match_weighted

        get_top_path_fn = getattr(lattice, get_top_path_method_name)
        best_path, best_rouge, all_node_rouge_dict = get_top_path_fn(
            mean_length, 
            expected_match, 
            d_length=args.d_length, 
            uniform=args.uniform
        )

        best_token_ids = lattice.get_path_tokens(best_path)
        best_detokenized = tokenizer.decode(best_token_ids, skip_special_tokens=True)

        rouge_scores = full_rouge_scorer.score(
            output.reference,
            best_detokenized
        )

        log_json.append({
            'file': file,
            'max_rouge': best_rouge,
            'max_path': best_path,
            'mbr_hypo': best_detokenized,
            'ref': output.reference,
            'mean_weighted_length': avg_len_weighted,
            'mean_unweighted_length': avg_len_unweighted,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        })

    print("Average rouge-1 using MBR:", sum(data['rouge1'] for data in log_json) / len(log_json))
    print("Average rouge-2 using MBR:", sum(data['rouge2'] for data in log_json) / len(log_json))
    print("Average rouge-L using MBR:", sum(data['rougeL'] for data in log_json) / len(log_json))

    with open(args.outfile, 'w+') as f:
        json.dump(log_json, f)

if __name__ == '__main__':
    main()
