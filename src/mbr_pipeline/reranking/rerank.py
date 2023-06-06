from rouge_score import rouge_scorer
from src.mbr_pipeline.list_eval.scorers import (
    Scorer,
    Score,
    self_bleu,
    rescore_bartscore,
    rescore_bertscore,
    rescore_rouge,
)
from collections import defaultdict
import numpy as np
import jsonlines
from tqdm import tqdm
from scipy.special import softmax
from scipy.stats import spearmanr, permutation_test
from typing import List


class Reranker:
    def __init__(self, rerank_temp, rerank_metric, rerank_geo_mean):
        self.rerank_temp = rerank_temp

        self.rerank_metric = (rerank_metric,)
        if "rouge" in rerank_metric:
            if rerank_geo_mean:
                order = int(rerank_metric[5:])
                self.rerank_metric = tuple(f"rouge{i}" for i in range(1, order + 1))
            self.rerank_fn = lambda hypos, probs: rescore_rouge(
                hypos, probs, self.rerank_metric
            )
        elif rerank_metric == "bertscore":
            self.rerank_fn = rescore_bertscore
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

    def rerank(self, item: dict) -> List[int]:
        hypos = item["hypos"]
        if "lprobs" in item:
            lprobs = np.array(item["lprobs"])
        else:
            lprobs = np.zeros(len(hypos))

        probs = softmax(lprobs / self.rerank_temp)
        rerank_scores = np.array(self.rerank_fn(hypos, probs))

        return rerank_scores.tolist()


def run_rerank(args):
    reranker = Reranker(args, args.rerank_metric)

    with jsonlines.open(args.outfile) as f:
        lines = list(iter(f))

    rerank_metric = args.rerank_metric
    if "rouge" in rerank_metric and args.rerank_geo_mean:
        rerank_metric += "_geo"
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
