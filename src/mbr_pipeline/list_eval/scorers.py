import argparse
import random
import re
import statistics
from collections import defaultdict
from functools import lru_cache
from numbers import Number
from typing import Dict, List

import datasets
import evaluate
import six
import torch
from sacrebleu.metrics import BLEU

try:
    from bart_score.bart_score import BARTScorer
except ImportError as e:
    print("Unable to import bart_score.")
import numpy as np
from rouge_score import rouge_scorer, tokenize
from sacrebleu import sentence_chrf


@lru_cache(maxsize=1600)
def cache_tokenize(text, stemmer):
    """Tokenize input text into a list of tokens.

    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.

    Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.

    Returns:
    A list of string tokens extracted from input text.
    """
    # Convert everything to lowercase.
    text = text.lower()
    # Replace any non-alpha-numeric characters with spaces.
    text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))
    tokens = re.split(r"\s+", text)
    if stemmer:
        # Only stem words more than 3 characters long.
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]
    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]
    return tokens


tokenize.tokenize = cache_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def get_bertscore():
    return datasets.load_metric("bertscore")


@lru_cache(maxsize=1)
def get_bleurt():
    return evaluate.load("bleurt", "BLEURT-20-D6")


# @lru_cache(maxsize=1)
# def get_fast_bleurt():
#     return evaluate.load("bleurt", "BLEURT-20-D6")


@lru_cache(maxsize=1)
def get_bartscore():
    bartscore = BARTScorer(device=device, checkpoint="facebook/bart-large-cnn")
    bartscore.load(path="/data/alexx/lattice-search/bart_score/bart_score.pth")
    return bartscore


@lru_cache(maxsize=5)
def get_rouge_scorer(rouges):
    return rouge_scorer.RougeScorer(list(rouges), use_stemmer=True)


def rescore_bertscore(topk_hypos, probs, bertscore=None, evidence_set=None):
    if evidence_set is None:
        evidence_set = topk_hypos
    if bertscore is None:
        bertscore = get_bertscore()
    score = bertscore.compute(
        predictions=topk_hypos,
        references=[evidence_set for _ in range(len(topk_hypos))],
        lang="en",
        device=device,
        batch_size=32,
    )
    return score["f1"]


def rescore_bleurt(topk_hypos, probs, bleurt=None, evidence_set=None):
    if evidence_set is None:
        evidence_set = topk_hypos
    if bleurt is None:
        bleurt = get_bleurt()
    predictions = []
    references = []
    for hypo in topk_hypos:
        for ref in evidence_set:
            predictions.append(hypo)
            references.append(ref)
    scores = bleurt.compute(
        predictions=predictions,
        references=references,
        # device=device,
        # batch_size=32,
    )["scores"]

    mean_scores = []
    num_ref = len(evidence_set)
    for i in range(len(topk_hypos)):
        score_subset = scores[i * num_ref : (i + 1) * num_ref]
        assert len(score_subset) == num_ref
        mean_scores.append(sum(score_subset) / num_ref)

    return mean_scores


def rescore_bartscore(topk_hypos, probs, bartscore=None, evidence_set=None):
    if evidence_set is None:
        evidence_set = topk_hypos
    # TODO: possibly buggy currently
    if bartscore is None:
        bartscore = get_bartscore()
    return bartscore.multi_ref_score(
        srcs=topk_hypos,
        agg="mean",
        hypos=[
            [h for j, h in enumerate(topk_hypos) if j != i]
            for i in range(len(topk_hypos))
        ],
    )


def rescore_rouge(topk_hypos, probs, rouge, eps=1e-4, evidence_set=None):
    same_evidence = False
    if evidence_set is None:
        same_evidence = True
        evidence_set = topk_hypos
    if rouge is None:
        rouge = get_rouge_scorer(("rouge1", "rouge2"))
    elif not isinstance(rouge, rouge_scorer.RougeScorer):
        # assume rouge is an iterable of the rouges we want
        rouge = get_rouge_scorer(rouge)

    k = len(topk_hypos)
    m = len(evidence_set)
    sim_matrix = np.zeros((k, m)) - 1
    for i in range(k):
        for j in range(m):
            if sim_matrix[i, j] != -1:
                continue
            pair_scores = rouge.score(topk_hypos[i], evidence_set[j])
            geo_mean = 1.0
            for score in pair_scores.values():
                geo_mean *= score.fmeasure + eps
            geo_mean = geo_mean ** (1 / len(pair_scores))
            sim_matrix[i, j] = geo_mean
            if same_evidence:
                sim_matrix[j, i] = geo_mean
    if probs is None:
        return sim_matrix.mean(axis=-1)
    return (sim_matrix @ probs[:, None])[:, 0]


def rescore_bleu(topk_hypos, probs, max_ngram_order=4, evidence_set=None):
    same_evidence = False
    if evidence_set is None:
        same_evidence = True
        evidence_set = topk_hypos

    bleu = BLEU(effective_order=True, max_ngram_order=max_ngram_order)

    k = len(topk_hypos)
    m = len(evidence_set)
    sim_matrix = np.zeros((k, m)) - 1
    for i in range(k):
        for j in range(m):
            if sim_matrix[i, j] != -1:
                continue
            pair_scores = bleu.sentence_score(topk_hypos[i], [evidence_set[j]])
            sim_matrix[i, j] = pair_scores.score
            if same_evidence:
                sim_matrix[j, i] = pair_scores.score
    if probs is None:
        return sim_matrix.mean(axis=-1)
    return (sim_matrix @ probs[:, None])[:, 0]


bleu_scorer = BLEU(effective_order=True)


def self_bleu(inp_group: List[str], num_refs=5):
    # tok_inputs = tokenize_sentences(inp_group)
    assert len(inp_group) >= num_refs
    bleu_scores = []
    for idx, inp in enumerate(inp_group):
        ref_indices = [idx]
        while idx in ref_indices:
            ref_indices = random.sample(range(len(inp_group)), k=num_refs)
        refs = [inp_group[i] for i in ref_indices]
        bleu_score = bleu_scorer.sentence_score(inp, refs)
        bleu_scores.append(bleu_score.score)
    return statistics.mean(bleu_scores)


class Score:
    def __init__(self, score_dict, pct=True):
        self.score_dict = score_dict
        self.is_pct = pct

    def __str__(self):
        sorted_keys = sorted(self.score_dict)
        if self.is_pct:
            fmt_fn = lambda n: f"{100 * n:.2f}".rjust(6)
        else:
            fmt_fn = lambda n: f"{n:.3f}".rjust(6)
        return " | ".join(fmt_fn(self.score_dict[k]) for k in sorted_keys)

    def __add__(self, other):
        assert isinstance(other, Score)
        assert self.score_dict.keys() == other.score_dict.keys()
        new_score_dict = {
            k: self.score_dict[k] + other.score_dict[k] for k in self.score_dict
        }
        return Score(new_score_dict, self.is_pct and other.is_pct)

    def __truediv__(self, num):
        assert isinstance(num, Number)
        new_score_dict = {k: v / num for k, v in self.score_dict.items()}
        return Score(new_score_dict, self.is_pct)

    def geomean(self):
        eps = 1e-4
        geo_mean = 1
        for v in self.score_dict.values():
            geo_mean *= v + eps
        geo_mean **= 1 / len(self.score_dict)
        return geo_mean

    def __getitem__(self, key):
        return self.score_dict[key]


class Scorer(object):
    def __init__(self, metrics):
        self.rouge_metrics = [
            metric for metric in metrics if metric.startswith("rouge")
        ]
        self.rouge_scorer = None
        if len(self.rouge_metrics) > 0:
            self.rouge_scorer = get_rouge_scorer(tuple(self.rouge_metrics))

        self.bertscore = None
        if "bertscore" in metrics:
            self.bertscore = get_bertscore()

        self.bartscore = None
        if "bartscore" in metrics:
            self.bartscore = get_bartscore()

        self.bleurt = None
        if "bleurt" in metrics:
            self.bleurt = get_bleurt()

        self.chrf_scorer = None
        if "chrf" in metrics:
            self.chrf_scorer = lambda hypo, ref: sentence_chrf(hypo, [ref]).score

        self.chrfpp_scorer = None
        if "chrf++" in metrics:
            self.chrfpp_scorer = lambda hypo, ref: sentence_chrf(
                hypo, [ref], word_order=2
            ).score

        self.bleu_scorer = None
        if "bleu" in metrics:
            self.bleu_scorer = BLEU(effective_order=True)

    def score(self, gold, hypos) -> List[Score]:
        scores = [{} for _ in range(len(hypos))]
        if self.rouge_scorer is not None:
            for i, hypo in enumerate(hypos):
                rouge = self.rouge_scorer.score(gold, hypo)
                for m, s in rouge.items():
                    scores[i][m] = s.fmeasure
        if self.bertscore is not None:
            bert_scores = self.bertscore.compute(
                predictions=hypos,
                references=[gold] * len(hypos),
                lang="en",
                device=device,
                batch_size=min(len(hypos), 64),
            )
            for i in range(len(hypos)):
                scores[i]["bertscore"] = bert_scores["f1"][i]
        if self.bleurt is not None:
            bleurt_scores = self.bleurt.compute(
                predictions=hypos,
                references=[gold] * len(hypos),
            )
            for i in range(len(hypos)):
                scores[i]["bleurt"] = bleurt_scores["scores"][i]
        if self.bartscore is not None:
            bart_scores = self.bartscore.score(
                srcs=[gold for _ in range(len(hypos))],
                tgts=hypos,
                batch_size=min(len(hypos), 64),
            )
            for i in range(len(hypos)):
                scores[i]["bartscore"] = bart_scores[i]
        if self.chrf_scorer is not None:
            for i, hypo in enumerate(hypos):
                chrf = self.chrf_scorer(hypo, gold) / 100.0
                scores[i]["chrf"] = chrf
        if self.chrfpp_scorer is not None:
            for i, hypo in enumerate(hypos):
                chrfpp = self.chrfpp_scorer(hypo, gold) / 100.0
                scores[i]["chrf++"] = chrfpp
        if self.bleu_scorer is not None:
            for i, hypo in enumerate(hypos):
                bleu = self.bleu_scorer.sentence_score(hypo, [gold]).score / 100.0
                scores[i]["bleu"] = bleu

        return [Score(s) for s in scores]
