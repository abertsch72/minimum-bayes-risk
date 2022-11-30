import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random
from functools import lru_cache
from collections import deque
from pprint import pprint
from lattice import Lattice

import sys
sys.path.append("./")
sys.path.append("./src/")

import src
from rouge_score import rouge_scorer

# from transformers import AutoTokenizer

from src.recom_search.evaluation.analysis import derive_path

# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")

full_rouge_scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# from src.recom_search.model.exec_setup import dataset

# results_dir = "output/data/sum_xsum_bs_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9"
'''
Results using best-first search with recombination + zip
'''
results_dir = "output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9"

result_files = os.listdir(results_dir)
        
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

i = 0
all_data = []

for file in tqdm(result_files):
    if i == 0:
        i += 1
        continue
    output = read_result(results_dir, file)
    
    if output.output is not None: # beam search
        raise Exception("must use lattice, cannot use beam search")
    else: # bfs / bfs+recomb
        graph_data = get_graph(output.ends)
        lattice = Lattice(*graph_data)
        print("Starting Lattice.get_length_dict()...")
        start = time.time()
        length_dict, all_node_length_dict = lattice.get_length_dict_reverse_dfs()
        paths_per_node = {node: sum(n for (_, n) in data.values()) for node, data in all_node_length_dict.items()}
        print(f"Getting length dict took {time.time() - start} seconds.")
        pprint(length_dict)
        total_num_paths = sum(num for (_, num) in length_dict.values())
        print(f"Total number paths from get_length_dict() = {total_num_paths}")
        
        # all_paths = get_all_paths(output.ends)
        # print(f"Total number of paths from get_all_paths() = {len(all_paths)}")
        # gt_length_dict = {}
        # for path in all_paths:
        #     if len(path) in gt_length_dict:
        #         gt_length_dict
        # avg_len_sampled = sum(len(path.tokens) for path in all_paths) / len(all_paths)
        # print('Avg length sampled:', avg_len_sampled)

        avg_len_unweighted = 0
        for length, (_, count) in length_dict.items():
            avg_len_unweighted += length * count / total_num_paths
        print('Avg length exact:', avg_len_unweighted)

        word_dict, all_node_word_dict = lattice.get_word_dict(all_node_length_dict)
        pprint(word_dict)

        import pdb; pdb.set_trace()