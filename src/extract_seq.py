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

results_dir = "output/data/sum_xsum_bs_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9" # beam search
# results_dir = "output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9" # bfs recomb + zip

result_files = os.listdir(results_dir)

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
        # rouges = [
        #     full_rouge_scorer.score(
        #         output.reference, 
        #         tokenizer.decode(hypo.all_token_idx, skip_special_tokens=True)
        #     )
        #     for hypo in output.ends
        # ]
        rouges = [full_rouge_scorer.score(output.reference, pred)]
        
        r1_scores.append(sum(r['rouge1'].fmeasure for r in rouges) / len(rouges))
        r2_scores.append(sum(r['rouge2'].fmeasure for r in rouges) / len(rouges))
        rl_scores.append(sum(r['rougeL'].fmeasure for r in rouges) / len(rouges))

    else: # bfs / bfs+recomb
        # max_idx = np.argmax([x.get_score_avg() for x in output.ends])
        # pred = tokenizer.decode(output.ends[max_idx].all_token_idx, skip_special_tokens=True)

        start = time.time()
        all_paths = get_all_paths(output.ends)
        if len(all_paths) > 1000:
            all_paths = random.sample(all_paths, 1000)
        print(f"get_all_paths took {time.time() - start} seconds")
        all_texts = [tokenizer.decode(path.token_ids, skip_special_tokens=True) for path in all_paths]
        rouges = [
            full_rouge_scorer.score(
                output.reference, 
                text
            )#['rouge2'].fmeasure 
            for text in all_texts
        ]
        scores = np.array([p.scores / (len(p.tokens)-1) for p in all_paths])

        metrics = {}
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            oracle_rouge = max(r[metric].fmeasure for r in rouges)
            avg_rouge = sum(r[metric].fmeasure for r in rouges) / len(rouges)
            max_score_idx = np.argmax(scores)
            max_score_rouge = rouges[max_score_idx][metric].fmeasure
            metrics[metric] = [oracle_rouge, avg_rouge, max_score_rouge]

        print(f"Oracle = {oracle_rouge} | Avg = {avg_rouge} | Max score = {max_score_rouge}")

        all_data.append(metrics)

        # all_data.append({
        #     'paths': all_paths,
        #     'texts': all_texts,
        #     'ref': output.reference,
        #     'rouges': rouges,
        #     'scores': scores,
        # })

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

print(f"rouge-1: mean = {np.mean(r1_scores)}, stddev = {np.std(r1_scores)}")
print(f"rouge-2: mean = {np.mean(r2_scores)}, stddev = {np.std(r2_scores)}")
print(f"rouge-l: mean = {np.mean(rl_scores)}, stddev = {np.std(rl_scores)}")

'''
Using adjusted score = score / length**0.8
rouge-1: mean = 0.3544601738984228, stddev = 0.14672593396017597
rouge-2: mean = 0.13802258224221106, stddev = 0.12266036875786063
rouge-l: mean = 0.29541395084545, stddev = 0.13910430327303128
'''

'''
rouge-1: mean = 0.3801218799819402, stddev = 0.11898826728814539
rouge-2: mean = 0.14291105807623625, stddev = 0.10256476002469918
rouge-l: mean = 0.3051743357451931, stddev = 0.11795751681461064
'''
