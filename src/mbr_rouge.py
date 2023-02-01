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
import wandb

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

def pairwise_similarity(topk_hypos, rerank_rouge_scorer, rerank_metrics):
    actual_topk = len(topk_hypos)
    sim_matrix = np.zeros((actual_topk, actual_topk))
    eps = 1e-4
    for i in range(actual_topk):
        for j in range(i+1, actual_topk):
            pairwise_scores = rerank_rouge_scorer.score(topk_hypos[i], topk_hypos[j])
            # take geometric mean of rerank metrics
            geo_mean = 1
            for m in rerank_metrics:
                geo_mean *= (pairwise_scores[m].fmeasure + eps)
            geo_mean **= (1/len(rerank_metrics))
            sim_matrix[i, j] = sim_matrix[j, i] = geo_mean
    return sim_matrix

def main():
    wandb.init(project="lattice-decoding", entity="gormleylab")
    if args.run_name != '':
        wandb.run.name = args.run_name
    wandb.config.update(args)
    
    if args.outfile is None:
        args.outfile = f"mbr_result_{args.lattice_metric}_dlen={args.d_length}"
        args.outfile += '_unif' if args.uniform else ''
        args.outfile += '_ca' if args.count_aware else ''
        args.outfile += f'_topk={args.lattice_topk}'
        args.outfile += '.json'
    print("Results will be saved to", args.outfile)

    global log_json
    log_json = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
    full_rouge_scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    if args.rerank_rouge == 'L':
        rerank_metrics = [f'rougeL']
    else:
        assert args.rerank_rouge.isdigit()
        rerank_metrics = [f'rouge{i+1}' for i in range(int(args.rerank_rouge))]
    rerank_rouge_scorer = rouge_scorer.RougeScorer(
        rerank_metrics, use_stemmer=True
    )

    # Results using best-first search with recombination + zip
    results_dir = "output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9"
    result_files = os.listdir(results_dir)

    use_rouge = args.lattice_metric.startswith('rouge')
    order = args.lattice_metric[5:]
    get_ngram_dict_method_name = f"get_{order}gram_dict{'_count_aware' if args.count_aware else ''}"
    get_top_path_method_name = f"get_top_rouge{order}_path{'_count_aware' if args.count_aware else''}"
    
    global lattice_topk_results
    lattice_topk_results = []
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

        lengths = np.array(sorted(length_dict.keys()))
        length_dist_unnorm = np.exp([length_dict[n][0] for n in lengths]) / lengths**args.length_alpha
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
        expected_match = match_unweighted if (args.uniform or args.match_uniform) else match_weighted

        # get top-k paths through lattice
        get_top_path_fn = getattr(lattice, get_top_path_method_name)
        topk_paths, topk_rouges, all_node_rouge_dict = get_top_path_fn(
            mean_length,
            expected_match,
            d_length=args.d_length,
            uniform=args.uniform,
            lattice_topk=args.lattice_topk,
            return_topk=args.rerank_topk,
            use_rouge=use_rouge
        )

        '''
        playing around with different values of rerank_topk
        (i.e. only track lattice_topk in DP search, but consider rerank_topk
        in second-stage MBR reranking)

        topk_results = {}
        for rerank_topk in (1, 5, 10, 20, 30, 50, 70, 100):
            ret_topk_paths, _ = lattice._extract_topk_gain_paths(
                all_node_rouge_dict,
                min_length=mean_length-args.d_length,
                max_length=mean_length+args.d_length,
                topk=rerank_topk
            )
            ret_topk_token_ids = [lattice.get_path_tokens(path) for path in ret_topk_paths]
            ret_topk_hypos = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in ret_topk_token_ids]

            sim_matrix = pairwise_similarity(ret_topk_hypos, rerank_rouge_scorer, rerank_metrics)
            max_idx_ = np.argmax(np.sum(sim_matrix, axis=-1))
            ret_best_detokenized = ret_topk_hypos[max_idx_]

            ret_rouge_scores = full_rouge_scorer.score(
                output.reference,
                ret_best_detokenized
            )
            topk_results[rerank_topk] = {
                'hypo': ret_best_detokenized,
                'rouge1': ret_rouge_scores['rouge1'].fmeasure,
                'rouge2': ret_rouge_scores['rouge2'].fmeasure,
                'rougeL': ret_rouge_scores['rougeL'].fmeasure
            }
        lattice_topk_results.append(topk_results)
        '''

        topk_token_ids = [lattice.get_path_tokens(path) for path in topk_paths]
        topk_hypos = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in topk_token_ids]

        sim_matrix = pairwise_similarity(topk_hypos, rerank_rouge_scorer, rerank_metrics)
        max_idx = np.argmax(np.sum(sim_matrix, axis=-1))

        oracle_topk_rouges = [full_rouge_scorer.score(output.reference, hypo) for hypo in topk_hypos]
        oracle_idx = max(range(len(topk_rouges)), key=lambda i: oracle_topk_rouges[i]['rouge2'].fmeasure)

        best_detokenized = topk_hypos[max_idx]
        best_path = topk_paths[max_idx]
        best_rouge = topk_rouges[max_idx]

        oracle_detokenized = topk_hypos[oracle_idx]

        rouge_scores = full_rouge_scorer.score(output.reference, best_detokenized)
        oracle_scores = full_rouge_scorer.score(output.reference, oracle_detokenized)

        log_json.append({
            'file': file,
            'max_rouge': best_rouge,
            'max_path': best_path,
            'ref': output.reference,
            'mbr_hypo': best_detokenized,
            'oracle_mbr_hypo': oracle_detokenized,
            'mean_weighted_length': avg_len_weighted,
            'mean_unweighted_length': avg_len_unweighted,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'oracle_rouge1': oracle_scores['rouge1'].fmeasure,
            'oracle_rouge2': oracle_scores['rouge2'].fmeasure,
            'oracle_rougeL': oracle_scores['rougeL'].fmeasure
        })
        wandb.log(log_json[-1])



    with open(args.outfile, 'w+') as f:
        json.dump(log_json, f)

    # for rerank_topk in lattice_topk_results[0].keys():
    #     scores = np.zeros(3)
    #     for all_results in lattice_topk_results:
    #         results = all_results[rerank_topk]
    #         scores += [results['rouge1'], results['rouge2'], results['rougeL']]
    #     print(f'rerank_topk = {rerank_topk}: ', scores / len(lattice_topk_results))

    print("Average rouge-1 using MBR:", sum(data['rouge1'] for data in log_json) / len(log_json))
    print("Average rouge-2 using MBR:", sum(data['rouge2'] for data in log_json) / len(log_json))
    print("Average rouge-L using MBR:", sum(data['rougeL'] for data in log_json) / len(log_json))
    print("Oracle rouge-1 using MBR:", sum(data['oracle_rouge1'] for data in log_json) / len(log_json))
    print("Oracle rouge-2 using MBR:", sum(data['oracle_rouge2'] for data in log_json) / len(log_json))
    print("Oracle rouge-L using MBR:", sum(data['oracle_rougeL'] for data in log_json) / len(log_json))


if __name__ == '__main__':
    main()
