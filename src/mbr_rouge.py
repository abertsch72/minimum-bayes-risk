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
from transformers import AutoTokenizer
import datasets
try:
    from bart_score.bart_score import BARTScorer
except:
    print("Unable to import bart_score.")
import wandb
import jsonlines

from lattice import Lattice

import sys
sys.path.append("./")
sys.path.append("./src/")

from rouge_score import rouge_scorer
from transformers import AutoTokenizer, BertModel

from src.recom_search.evaluation.analysis import derive_path
from src.recom_search.model.exec_setup import args, grouped_args

import wandb

from typing import List
import statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

result_files = ['sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38944626_Tess-Newal.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38841897_Helen-Ross.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36951809_The-first-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_37329266_The-conser.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38972281_Satnam-Sin.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_34688872_Media-play.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_32083717_The-musici.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_26237638_Prime-Mini.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_32319927_Mark-Jones.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_37439513_The-distri.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_34276797_The-polls-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_37685083_Mladen-Gru.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_37480369_The-Irish-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_28359094_Transport-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_37328868_The-38-yea.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36414580_Manager-Ch.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_40134723_The-former.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36538967_The-24-yea.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_18173273_Researcher.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_37584589_7-October-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36918140_Members-of.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35837959_The-cards,.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36518750_The-46-yea.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36033395_East-Linds.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38722428_Shock-2015.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35919641_29-March-2.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_40849171_Suleiman-A.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_39134538_He-will-ai.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_28872944_Researcher.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_29507419_Cornwall-C.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_30275513_Seven-of-h.pkl', "sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35998891_Jon-Lewis'.pkl", 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_40141033_Armed-offi.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_40470133_Police,-fi.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36784182_The-Dow-Jo.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35899243_The-author.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_32318906_Media-play.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38844702_Matthew-Ge.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_34784906_The-Russia.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35027395_Both-men-w.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38897462_Nyom-also-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38704934_The-Robins.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_39640064_The-Ports-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36986643_Striker-Em.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_40761453_The-vetera.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_34589891_The-power-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_32242131_The-airlin.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_30792462_The-White-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_31370822_The-blaze-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35994279_Three-sepa.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_37134709_AFB-was-co.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_40908381_Dave-Tarpe.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_28205563_The-victim.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35625097_The-UK-wil.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_39801988_The-Anglo-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36588482_Ian-"Jacko.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_37472975_Government.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38105023_Bribes-of-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_26780897_The-Austra.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_26393852_Sosefina-A.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36598609_Almost-1.7.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36308067_Abdul-Hafi.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_39072865_It-is-the-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_33810603_Neil-Fears.pkl', "sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_33849042_O'Grady-de.pkl", 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_30759868_The-18-ske.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38721046_The-money-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_32196037_On-5-April.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35569627_Media-play.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_40594126_The-Juno-s.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_34709664_It-seemed-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38838606_Only-Itali.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_22152699_Cleveland-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_33243677_Hughes,-19.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_25518137_In-the-196.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_24209153_She-bought.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_39005107_Media-play.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_33359978_The-idea-w.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35953521_The-81-yea.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_34051870_The-photos.pkl', "sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_31067802_There's-no.pkl", 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_38211788_Nicholas-E.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_40962385_The-Welsh-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_32860648_The-49-yea.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_35602332_Programs-s.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_34698579_There-were.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_30250624_He-returne.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_30704751_Mr-Mallon-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_39254234_Isabel-Gen.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_39789892_Amarjeet-S.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36397500_The-member.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_29026398_The-Englan.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_37662690_The-news-c.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_33594654_Researcher.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_19533038_GB-started.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_17745366_The-transi.pkl', "sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_40947448_Women's-ri.pkl", 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_36678976_They-were-.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_31030136_28-January.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_32963741_Paul-Thoma.pkl', 'sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9_34744153_Lawyers-fo.pkl']


def compute_bertscore(topk_hypos, bertscore):
    score = bertscore.compute(
        predictions=topk_hypos, 
        references=[topk_hypos for _ in range(len(topk_hypos))],
        lang='en',
        device=device,
        batch_size=32
    )
    return score['f1']

def compute_bartscore(topk_hypos, bartscore):
    return bartscore.multi_ref_score(
        srcs=topk_hypos, 
        hypos=[topk_hypos for _ in range(len(topk_hypos))], 
        agg="mean"
    )

from sacrebleu.metrics import BLEU
bleu_scorer = BLEU(effective_order=True)

def self_bleu(inp_group: List[str]):
    # tok_inputs = tokenize_sentences(inp_group)
    bleu_scores = []
    for idx, inp in enumerate(inp_group):
        # bleu_score = nltk.translate.bleu_score.sentence_bleu([x for jdx, x in enumerate(tok_inputs) if jdx != idx], inp)
        bleu_score = bleu_scorer.sentence_score(
            inp, [x for jdx, x in enumerate(inp_group) if jdx != idx])
        bleu_scores.append(bleu_score.score)
    return statistics.mean(bleu_scores)


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
    wandb.init(project="lattice-decoding", entity="gormleylab", group="sweep-reg-rouge1", 
               config=vars(grouped_args['mbr']))
    if args.run_name != '':
        wandb.run.name = args.run_name
    
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

    rerank_metrics = ['rouge1']
    if args.rerank == 'L':
        rerank_metrics = [f'rougeL']
    elif args.rerank == 'bertscore':
        bertscore = datasets.load_metric("bertscore")
    elif args.rerank == 'bartscore':
        bartscore = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        bartscore.load(path='/data/alexx/lattice-search/bart_score/bart_score.pth')
    else:
        assert args.rerank.isdigit()
        rerank_metrics = [f'rouge{i+1}' for i in range(int(args.rerank))]
    rerank_rouge_scorer = rouge_scorer.RougeScorer(
        rerank_metrics, use_stemmer=True
    )

    # Results using best-first search with recombination + zip
    results_dir = "output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9"
    # results_dir = "output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_rcb_0.75_0.0_0.9"
    # results_dir = "output/data/sum_xsum_bs_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9"
    global result_files
    result_files = os.listdir(results_dir)
    # print(result_files)
    # result_files = result_files[:101]

    use_rouge = args.lattice_metric.startswith('rouge')
    exact = 'exact' in args.lattice_metric
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
        #lengths = np.array(sorted(all_node_length_dict.keys()))
        length_dist_raw = [length_dict[n][0] for n in lengths]
        length_dist = length_dist_unnorm / np.sum(length_dist_unnorm)
        avg_len_weighted = np.sum(length_dist * lengths)

        avg_len_unweighted = 0
        for length, (lprob, count) in length_dict.items():
            avg_len_unweighted += length * count / total_num_paths

        # get n-gram match dictionary
        if exact:
            get_ngram_dict_fn = lattice.get_1gram_dict_by_length
            ngram_dict, all_node_ngram_dict = get_ngram_dict_fn(
                all_node_length_dict, 
                max_length=1.5 * avg_len_weighted
            )
        else:
            get_ngram_dict_fn = getattr(lattice, get_ngram_dict_method_name)
            ngram_dict, all_node_ngram_dict = get_ngram_dict_fn(all_node_length_dict, target_length=args.target_length, allowed_deviation = args.length_deviation)


            match_unweighted = {word: count / total_num_paths for word, (_, count) in ngram_dict.items()}
            match_weighted = {word: np.exp(lprob) for word, (lprob, _) in ngram_dict.items()}

        mean_length = avg_len_unweighted if args.uniform else avg_len_weighted
        if args.mean_override != -1:
            mean_length = args.mean_override

        expected_match = match_unweighted if (args.uniform or args.match_uniform) else match_weighted

        # get top-k paths through lattice
        if exact:
            get_top_path_fn = lattice.get_top_rouge1_path_exact
            topk_paths, topk_rouges, all_node_rouge_dict = get_top_path_fn(
                # mean_length,
                avg_len_weighted,
                # gold_length,
                # expected_match,
                ngram_dict,
                # d_length=args.d_length,
                # uniform=args.uniform,
                lattice_topk=args.lattice_topk,
                return_topk=args.rerank_topk,
                # use_rouge=use_rouge
            )
        else:
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

        topk_token_ids = [lattice.get_path_tokens(path) for path in topk_paths]
        topk_hypos = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in topk_token_ids]
        # topk_hypos = output.output
        
        if args.rerank == 'bertscore':
            f1s = compute_bertscore(topk_hypos, bertscore)
            max_idx = np.argmax(f1s)
        elif args.rerank == 'bartscore':
            bartscores = compute_bartscore(topk_hypos, bartscore)
            max_idx = np.argmax(bartscores)
        else:
            sim_matrix = pairwise_similarity(topk_hypos, rerank_rouge_scorer, rerank_metrics)
            max_idx = np.argmax(np.sum(sim_matrix, axis=-1))

        oracle_topk_rouges = [full_rouge_scorer.score(output.reference, hypo) for hypo in topk_hypos]
        def eval_idx(i):
            eps = 1e-4
            r1 = oracle_topk_rouges[i]['rouge1'].fmeasure + eps
            r2 = oracle_topk_rouges[i]['rouge2'].fmeasure + eps
            rL = oracle_topk_rouges[i]['rougeL'].fmeasure + eps
            return (r1 * r2 * rL)**(1/3)

        oracle_idx = max(range(len(topk_hypos)), key=eval_idx)
        worst_oracle_idx = min(range(len(topk_hypos)), key=eval_idx)

        best_detokenized = topk_hypos[max_idx]
        # best_path = topk_paths[max_idx]
        # best_rouge = topk_rouges[max_idx]
        best_path, best_rouge = None, None
        avg_len_weighted = avg_len_unweighted = None

        oracle_detokenized = topk_hypos[oracle_idx]
        worst_oracle_detokenized = topk_hypos[worst_oracle_idx]

        rouge_scores = full_rouge_scorer.score(output.reference, best_detokenized)
        oracle_scores = full_rouge_scorer.score(output.reference, oracle_detokenized)
        worst_oracle_scores = full_rouge_scorer.score(output.reference, worst_oracle_detokenized)

        topk_self_bleu = self_bleu(topk_hypos) if len(topk_hypos) > 1 else 1.0

        dups = []
        seen = set()
        for i, hypo in enumerate(topk_hypos):
            if hypo in seen:
                dups.append((hypo, i))
            else:
                seen.add(hypo)
        if len(dups) > 0:
            import pdb; pdb.set_trace()

        log_json.append({
            'file': file,
            'max_rouge': best_rouge,
            'topk_hypos': topk_hypos,
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
            'oracle_rougeL': oracle_scores['rougeL'].fmeasure,
            'worst_oracle_rouge1': worst_oracle_scores['rouge1'].fmeasure,
            'worst_oracle_rouge2': worst_oracle_scores['rouge2'].fmeasure,
            'worst_oracle_rougeL': worst_oracle_scores['rougeL'].fmeasure,
            'self_bleu': topk_self_bleu,
            'unique': len(set(topk_hypos)) / len(topk_hypos),
            'average_rouge1': sum(r['rouge1'].fmeasure for r in oracle_topk_rouges) / len(oracle_topk_rouges),
            'average_rouge2': sum(r['rouge2'].fmeasure for r in oracle_topk_rouges) / len(oracle_topk_rouges),
            'average_rougeL': sum(r['rougeL'].fmeasure for r in oracle_topk_rouges) / len(oracle_topk_rouges),
        })
        #wandb.log(log_json[-1])
        lattice_topk_results.append({"all_50": topk_hypos, "gold": output.reference, "num_unique": len(set(topk_hypos))})


    with jsonlines.open(args.outfile + "hypos", "w") as f:
        f.write_all(lattice_topk_results)
    with open(args.outfile, 'w+') as f:
        json.dump(log_json, f)

    # for rerank_topk in lattice_topk_results[0].keys():
    #     scores = np.zeros(3)
    #     for all_results in lattice_topk_results:
    #         results = all_results[rerank_topk]
    #         scores += [results['rouge1'], results['rouge2'], results['rougeL']]
    #     print(f'rerank_topk = {rerank_topk}: ', scores / len(lattice_topk_results))

    k = len(topk_hypos)

    print("Average rouge-1 using MBR:", sum(data['rouge1'] for data in log_json) / len(log_json))
    print("Average rouge-2 using MBR:", sum(data['rouge2'] for data in log_json) / len(log_json))
    print("Average rouge-L using MBR:", sum(data['rougeL'] for data in log_json) / len(log_json))
    print()
    print("Oracle rouge-1 using MBR:", sum(data['oracle_rouge1'] for data in log_json) / len(log_json))
    print("Oracle rouge-2 using MBR:", sum(data['oracle_rouge2'] for data in log_json) / len(log_json))
    print("Oracle rouge-L using MBR:", sum(data['oracle_rougeL'] for data in log_json) / len(log_json))
    print()
    print("Lowest Oracle rouge-1 using MBR:", sum(data['worst_oracle_rouge1'] for data in log_json) / len(log_json))
    print("Lowest Oracle rouge-2 using MBR:", sum(data['worst_oracle_rouge2'] for data in log_json) / len(log_json))
    print("Lowest Oracle rouge-L using MBR:", sum(data['worst_oracle_rougeL'] for data in log_json) / len(log_json))
    print()
    print(f"Average rouge-1 of top-{k}:", sum(data['average_rouge1'] for data in log_json) / len(log_json))
    print(f"Average rouge-2 of top-{k}:", sum(data['average_rouge2'] for data in log_json) / len(log_json))
    print(f"Average rouge-L of top-{k}:", sum(data['average_rougeL'] for data in log_json) / len(log_json))
    print(f"Self-BLEU of top-{k}:", sum(data['self_bleu'] for data in log_json) / len(log_json))
    print(f"% Unique among top-{k}:", sum(data['unique'] for data in log_json) / len(log_json))

    keys = list(log_json[0].keys())
    table_data = [[row[k] for k in keys] for row in log_json]
        
    wandb.log({
        'rouge1': sum(data['rouge1'] for data in log_json) / len(log_json),
        'rouge2': sum(data['rouge2'] for data in log_json) / len(log_json),
        'rougeL': sum(data['rougeL'] for data in log_json) / len(log_json),
        'oracle_rouge1': sum(data['oracle_rouge1'] for data in log_json) / len(log_json),
        'oracle_rouge2': sum(data['oracle_rouge2'] for data in log_json) / len(log_json),
        'oracle_rougeL': sum(data['oracle_rougeL'] for data in log_json) / len(log_json),
        'self_bleu': sum(data['self_bleu'] for data in log_json) / len(log_json),
        'outputs': wandb.Table(data=table_data, columns=keys),
    })

    wandb.log({'topk': lattice_topk_results})
    print(lattice_topk_results)


if __name__ == '__main__':
    main()
