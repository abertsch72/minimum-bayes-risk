from collections import defaultdict
from typing import List

import jsonlines
import numpy as np
from rouge_score import rouge_scorer
from scipy.special import softmax
from scipy.stats import permutation_test, spearmanr
from tqdm import tqdm

from src.mbr_pipeline.list_eval.scorers import (
    Score,
    Scorer,
    rescore_bartscore,
    rescore_bertscore,
    rescore_bleu,
    rescore_rouge,
    self_bleu,
)


class Reranker:
    def __init__(self, rerank_temp, rerank_metric, rerank_geo_mean, rank_by_freq):
        self.rerank_temp = rerank_temp
        self.rank_by_freq = rank_by_freq

        self.rerank_metric = (rerank_metric,)
        if "rouge" in rerank_metric:
            if rerank_geo_mean:
                order = int(rerank_metric[5:])
                self.rerank_metric = tuple(f"rouge{i}" for i in range(1, order + 1))
            self.rerank_fn = lambda hypos, probs, evidence_set: rescore_rouge(
                hypos, probs, self.rerank_metric, evidence_set=evidence_set
            )
        elif rerank_metric == "bertscore":
            self.rerank_fn = rescore_bertscore
        elif rerank_metric == "bleu":
            self.rerank_fn = rescore_bleu
        else:
            self.rerank_fn = None
        self.metrics = defaultdict(lambda: [])

    def geomean(self, score):
        eps = 1e-4

        geo_mean = 1
        for v in score.values():
            geo_mean *= v + eps
        geo_mean **= 1 / len(self.rerank_metric)

        return geo_mean

    def rerank(self, item: dict, evidence_set: List[dict] = None) -> List[int]:
        hypos = item["hypos"]
        lprobs = evidence_hypos = None
        if evidence_set is not None:
            evidence_hypos = evidence_set["hypos"]
            if "lprobs" in evidence_set:
                lprobs = np.array(evidence_set["lprobs"])
        elif "lprobs" in item:
            lprobs = np.array(item["lprobs"])
        if lprobs is None:
            lprobs = np.ones_like(lprobs)

        if self.rerank_temp == float("inf") or self.rank_by_freq:
            probs = np.ones_like(lprobs) / len(lprobs)
        else:
            probs = softmax(lprobs / self.rerank_temp)
        rerank_scores = np.array(
            self.rerank_fn(hypos, probs, evidence_set=evidence_hypos)
        )

        return rerank_scores.tolist()


def run_rerank(args):
    reranker = Reranker(args, args.rerank_metric)

    with jsonlines.open(args.outfile) as f:
        lines = list(iter(f))

    rerank_metric = args.rerank_metric
    if "rouge" in rerank_metric and args.rerank_geo_mean:
        rerank_metric += "_geo"
    if args.rank_by_freq:
        rerank_metric += "_freq"
    rerank_metric += f"temp-{args.rerank_temp}"

    for line in tqdm(lines):
        scores_key = f"rerank_scores_{rerank_metric}"
        if scores_key in line:
            continue
        scores = reranker.rerank(line)
        line[scores_key] = scores

    with jsonlines.open(args.outfile, "w") as f:
        f.write_all(lines)


if __name__ == "__main__":
    parser = get_parser(rerank=True)
    args = parser.parse_args()
    run_rerank(args)
