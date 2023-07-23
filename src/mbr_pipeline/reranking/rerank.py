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
    def __init__(
        self,
        rerank_temp,
        rerank_metric,
        rerank_geo_mean,
        rank_by_freq,
        importance_sample,
        length_corrected,
        length_penalty=0.0,
    ):
        self.rerank_temp = rerank_temp
        self.rank_by_freq = rank_by_freq
        self.importance_sample = importance_sample
        self.length_corrected = length_corrected
        self.length_penalty = length_penalty

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

    def rerank_lattice(self, item, tokenizer, lattice):
        from src.mbr_pipeline.list_gen.lattice_mbr import (
            get_lattice_methods,
            get_match_dict,
            get_mean_length,
            get_partition_function,
        )

        get_dict_fn, _ = get_lattice_methods(
            lattice, lattice_metric="rouge1", count_aware=False
        )
        length_dict, all_node_length_dict = lattice.get_length_dict()
        mean_length = get_mean_length(
            length_dict, uniform_length=False, length_temp=1.0
        )
        # Z = get_partition_function(length_dict, uniform_match=False)
        ngram_dict, _ = get_dict_fn(
            all_node_length_dict,
            target_length=mean_length,
            allowed_deviation=float("inf"),
        )
        match_dict = get_match_dict(
            ngram_dict, 0, lattice_metric="rouge1", uniform_match=False, match_temp=0.01
        )
        # import pdb; pdb.set_trace()
        hypos = item["hypos"]

        scores = []
        for hypo in hypos:
            score = 0.0
            ids = tokenizer.encode(hypo)
            for token_id in ids:
                score += match_dict.get(tokenizer.decode(token_id), 0)
            score /= len(ids)
            scores.append(score)
        return scores

    def rerank(
        self, item: dict, evidence_set: List[dict] = None, tokenizer=None, lattice=None
    ) -> List[int]:
        if lattice is not None:
            return self.rerank_lattice(item, tokenizer, lattice)

        hypos = item["hypos"]

        lprobs = true_lprobs = evidence_hypos = None
        if evidence_set is not None:
            evidence_hypos = evidence_set["hypos"]
            if "lprobs" in evidence_set:
                lprobs = np.array(evidence_set["lprobs"])
            if self.importance_sample:
                assert "unbiased_lprobs" in evidence_set
                true_lprobs = np.array(evidence_set["unbiased_lprobs"])
        elif "lprobs" in item:
            lprobs = np.array(item["lprobs"])
            if self.importance_sample:
                assert "unbiased_lprobs" in item
                true_lprobs = np.array(item["unbiased_lprobs"])
                lprobs = true_lprobs - lprobs

        if lprobs is not None and self.length_corrected:
            assert tokenizer is not None
            old_lprobs = lprobs
            evidence_tokenized = tokenizer(
                evidence_hypos if evidence_hypos is not None else hypos
            )["input_ids"]
            lengths = np.array([len(h) + 1 for h in evidence_tokenized])
            lprobs = lprobs * (lengths**self.length_penalty)
            # lprobs = lprobs * (lengths)
            ### experimental code
            # lprobs = old_lprobs / lprobs
            ### experimental code

        if lprobs is None or self.rank_by_freq:
            lprobs = np.ones_like(lprobs)

        if self.importance_sample:
            assert true_lprobs is not None
            # if importance sampling, don't use a reranking temperature
            # correct up to a constant of proportionality
            probs = softmax(lprobs)
        elif self.rerank_temp == float("inf") or self.rank_by_freq:
            probs = np.ones_like(lprobs) / len(lprobs)
        else:
            probs = softmax(lprobs / self.rerank_temp)
            # import pdb; pdb.set_trace()

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
