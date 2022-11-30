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

import sys
sys.path.append("./")
sys.path.append("./src/")

import src
from rouge_score import rouge_scorer

from transformers import AutoTokenizer

from src.recom_search.evaluation.analysis import derive_path, find_start_end

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")

full_rouge_scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# from src.recom_search.model.exec_setup import dataset

# results_dir = "output/output/data/sum_xsum_bs_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9"
results_dir = "output/output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9"

result_files = os.listdir(results_dir)

class Lattice(object):
    def __init__(self, node_dict, edge_dict):
        self.nodes = node_dict
        self.edges = {node: {} for node in self.nodes} # node -> node -> score
        for edge_data in edge_dict.values():
            src, tgt, score = edge_data['src'], edge_data['tgt'], edge_data['score']
            assert tgt not in self.edges[src]
            self.edges[src][tgt] = score
        self.sos, self.eos_list, _ = find_start_end(node_dict, edge_dict)
        for eos in self.eos_list: # validate that eos nodes don't have any outgoing edges
            assert len(self.edges[eos]) == 0

    @lru_cache(maxsize=1)
    def get_length_dict(self):
        node_queue = deque()
        node_queue.append(self.sos)
        # node_length_dict = {
        #   node: {
        #       length: (score, count)
        #   }
        # }
        node_length_dict = {node: {} for node in self.nodes}
        node_length_dict[self.sos] = {0: (0, 0)}
        visited = set()

        while len(node_queue) > 0:
            curr_node = node_queue.popleft()
            if curr_node in visited:
                continue
            for child_node, edge_score in self.edges[curr_node].items():
                child_node_length_dict = node_length_dict[child_node]
                for length, (length_score, length_count) in node_length_dict[curr_node].items():
                    score_delta = edge_score * length_count + length_score
                    if length + 1 in child_node_length_dict:
                        old_score, old_count = child_node_length_dict[length + 1]
                        new_score = old_score + score_delta
                        new_count = old_count + length_count
                    else:
                        new_score, new_count = score_delta, length_count
                    child_node_length_dict[length + 1] = (new_score, new_count)

                node_queue.append(child_node)
        
        length_dict = {}
        for eos in self.eos:
            for length, (length_score, length_count) in node_length_dict[eos].items();
                if length in length_dict:
                    old_score, old_count = length_dict[length]
                    length_dict[length] = (old_score + length_score, old_count + length_count)
                else:
                    length_dict[length] = (length_score, length_count)
        return length_dict
        
def read_result(dir, file):
    with open(os.path.join(dir, file), 'rb') as f:
        x = pickle.load(f)
    return x    

all_results = []

r1_scores = []
r2_scores = []
rl_scores = []

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

def get_highest_model_score_path(graph_ends):
    all_paths = get_all_paths(graph_ends)

    max_idx = np.argmax([p.scores / (len(p.tokens)-1) for p in all_paths])
    return all_paths[max_idx]

i = 0

all_data = []

for file in tqdm(result_files):
    if i == 0:
        i += 1
        continue
    output = read_result(results_dir, file)
    
    if output.output is not None: # beam search
        max_idx = np.argmax(output.score_avg) 
        pred = output.output[0]
    else: # bfs / bfs+recomb
        # max_idx = np.argmax([x.get_score_avg() for x in output.ends])
        # pred = tokenizer.decode(output.ends[max_idx].all_token_idx, skip_special_tokens=True)

        start = time.time()
        all_paths = get_all_paths(output.ends)
        if len(all_paths) > 200:
            all_paths = random.sample(all_paths, 200)
        print(f"get_all_paths took {time.time() - start} seconds")
        all_texts = [tokenizer.decode(path.token_ids, skip_special_tokens=True) for path in all_paths]
        print("\n".join(all_texts[:200]))
        rouges = [
            full_rouge_scorer.score(
                output.reference, 
                text
            )['rouge2'].fmeasure for text in all_texts
        ]
        scores = [p.scores / (len(p.tokens)-1) for p in all_paths]

        all_data.append({
            'paths': all_paths,
            'texts': all_texts,
            'ref': output.reference,
            'rouges': rouges,
            'scores': scores,
        })

        1/0

        # plt.scatter(scores, rouges)
        # plt.xlabel("Model Score")
        # plt.ylabel("ROUGE-2")
        # plt.show()


with open("results.pkl", 'wb+') as f:
    pickle.dump(all_data, f)

# for file in tqdm(result_files):
#     output = read_result(results_dir, file)
#     all_results.append(output)
    

#     if output.output is not None: # beam search
#         max_idx = np.argmax(output.score_avg) 
#         pred = output.output[0]
#     else: # bfs / bfs+recomb
#         # max_idx = np.argmax([x.get_score_avg() for x in output.ends])
#         # pred = tokenizer.decode(output.ends[max_idx].all_token_idx, skip_special_tokens=True)

#         max_path = get_highest_model_score_path(output.ends)
#         pred = tokenizer.decode(max_path.token_ids, skip_special_tokens=True)
        
#     scores = full_rouge_scorer.score(output.reference, pred)
    
#     r1_scores.append(scores['rouge1'].fmeasure)
#     r2_scores.append(scores['rouge2'].fmeasure)
#     rl_scores.append(scores['rougeL'].fmeasure)

# print(f"rouge-1: mean = {np.mean(r1_scores)}, stddev = {np.std(r1_scores)}")
# print(f"rouge-2: mean = {np.mean(r2_scores)}, stddev = {np.std(r2_scores)}")
# print(f"rouge-l: mean = {np.mean(rl_scores)}, stddev = {np.std(rl_scores)}")

'''
Using adjusted score = score / length**0.8
rouge-1: mean = 0.3544601738984228, stddev = 0.14672593396017597
rouge-2: mean = 0.13802258224221106, stddev = 0.12266036875786063
rouge-l: mean = 0.29541395084545, stddev = 0.13910430327303128
'''