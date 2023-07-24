import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from rouge_score import rouge_scorer
from scipy.stats import permutation_test, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
length_penalty = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys

metric_name = sys.argv[-1]

all_at_once = False
if "rouge" in metric_name:
    rouge = metric_name
    scorer = rouge_scorer.RougeScorer([rouge], use_stemmer=True)

    def get_score(ref_, hypo_):
        return scorer.score(ref_, hypo_)[rouge].fmeasure

elif metric_name == "bartscore":
    from bart_score.bart_score import BARTScorer

    scorer = BARTScorer(device=device, checkpoint="facebook/bart-large-cnn")
    scorer.load(path="/data/alexx/lattice-search/bart_score/bart_score.pth")

    def get_score(ref_, hypo_):
        return None

elif metric_name == "bertscore":
    all_at_once = True
    from datasets import load_metric

    scorer = load_metric("bertscore")

    def get_scores(ref_, hypos_):
        return scorer.compute(
            predictions=hypos_,
            references=[ref_ for _ in range(len(hypos_))],
            lang="en",
            device=device,
            batch_size=100,
        )["f1"]

elif metric_name == "chrf":
    from sacrebleu import sentence_chrf

    def get_score(ref_, hypo_):
        return sentence_chrf(hypo_, [ref_]).score

elif metric_name == "chrf++":
    from sacrebleu import sentence_chrf

    def get_score(ref_, hypo_):
        return sentence_chrf(hypo_, [ref_], word_order=2).score

elif metric_name == "comet":
    from datasets import load_metric

    scorer = load_metric("comet")
    all_at_once = True

    def get_scores(ref_, hypos_, source_):
        scores_ = scorer.compute(
            predictions=hypos_,
            references=[ref_ for _ in range(len(hypos_))],
            sources=[source_ for _ in range(len(hypos_))],
        )
        return scores_["scores"]

elif metric_name == "bleurt":
    from datasets import load_metric

    scorer = load_metric("bleurt")
    all_at_once = True

    def get_scores(ref_, hypos_, source_):
        scores_ = scorer.compute(
            predictions=hypos_,
            references=[ref_ for _ in range(len(hypos_))],
        )
        return scores_["scores"]

else:
    raise Exception("metric not yet implemented:", metric_name)

# result_file = "/home/alexx/lattice-search/fixed-test-sampling_outputs/cnndm/facebook-bart-large-cnn/200/temp-temp=1.0top_p=1.0epsilon_cutoff=0.0.jsonl"
result_file = "result.jsonl"
with jsonlines.open(result_file, "r") as f:
    results = list(f.iter())


def compute_correlation(x, y):
    def _statistic(x):
        return spearmanr(x, y).statistic

    res = permutation_test(
        (x,), _statistic, permutation_type="pairings", n_resamples=500
    )
    return res


norm_corrs = []
unnorm_corrs = []

use_cached = False
all_scores = np.empty((len(results), 200))
all_lprobs_norm = np.empty_like(all_scores)
all_lprobs_unnorm = np.empty_like(all_scores)

import os

if os.path.isfile(f"all_scores_{metric_name}.npy"):
    use_cached = True
    all_scores = np.load(f"all_scores_{metric_name}.npy")
    all_lprobs_norm = np.load("all_lprobs_norm.npy")
    all_lprobs_unnorm = np.load("all_lprobs_unnorm.npy")

for i, result in enumerate(tqdm(results)):
    if use_cached:
        lprobs_norm = all_lprobs_norm[i]
        lprobs_unnorm = all_lprobs_unnorm[i]
        scores = all_scores[i]
    else:
        ref = result["gold"]
        hypos = result["hypos"]
        evidence_tokenized = tokenizer(hypos)["input_ids"]
        lengths = np.array([len(h) + 1 for h in evidence_tokenized])
        lprobs_norm = np.array(result["lprobs"])
        lprobs_unnorm = np.array(result["lprobs"]) * (lengths**2)

        scores = []
        if all_at_once:
            scores = get_scores(ref, hypos, result["document"])
        else:
            for hypo in hypos:
                # score = scorer.compute(predictions=[hypo], references=[ref])[rouge]
                score = get_score(ref, hypo)
                # scores.append(score.mid.fmeasure)
                scores.append(score)
        scores = np.array(scores)

    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.scatter(lprobs_norm, scores)
    # ax1.set_xlabel("Normalized Scores")
    # ax2.scatter(lprobs_unnorm, scores)
    # ax2.set_xlabel("Unnormalized Scores")
    # plt.show()

    norm_corr = compute_correlation(lprobs_norm, scores)
    unnorm_corr = compute_correlation(lprobs_unnorm, scores)

    norm_corrs.append(norm_corr)
    unnorm_corrs.append(unnorm_corr)

    all_scores[i] = scores
    all_lprobs_norm[i] = lprobs_norm
    all_lprobs_unnorm[i] = lprobs_unnorm

print("*** Length normalized ***")
print("corr =", sum([c.statistic for c in norm_corrs]) / len(norm_corrs))
print("pvalue =", sum([c.pvalue for c in norm_corrs]) / len(norm_corrs))

print("*** Length unnormalized ***")
print("corr =", sum([c.statistic for c in unnorm_corrs]) / len(unnorm_corrs))
print("pvalue =", sum([c.pvalue for c in unnorm_corrs]) / len(unnorm_corrs))
print()

norm_corrs = np.array([c.statistic for c in norm_corrs])
unnorm_corrs = np.array([c.statistic for c in unnorm_corrs])

norm_win_rate = np.sum(norm_corrs > unnorm_corrs) / len(norm_corrs)
unnorm_win_rate = 1 - norm_win_rate
print("Length normalized win rate:", norm_win_rate)
print("Length unnormalized win rate:", unnorm_win_rate)

np.save(f"all_scores_{metric_name}.npy", all_scores)
np.save("all_lprobs_norm.npy", all_lprobs_norm)
np.save("all_lprobs_unnorm.npy", all_lprobs_unnorm)
