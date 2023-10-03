import sys
from collections import defaultdict

import jsonlines
import numpy as np
from scipy.stats import permutation_test, spearmanr
from tqdm import tqdm

from src.mbr_pipeline.list_eval.scorers import (
    Score,
    Scorer,
    rescore_bartscore,
    rescore_bertscore,
    rescore_rouge,
    self_bleu,
)


class C:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Metrics:
    def __init__(self, eval_metrics):
        self.scorer = Scorer(eval_metrics)
        self.eval_metrics = eval_metrics

        self.report_corr = False

        self.zero_score = Score({m: 0 for m in eval_metrics})
        self.metrics = defaultdict(lambda: [])

    def average(self, seq):
        if len(seq) == 0:
            return 0
        start = 0
        if isinstance(seq[0], Score):
            start = self.zero_score
        return sum(seq, start=start) / len(seq)

    def output(self, outfile=sys.stdout):
        color = outfile == sys.stdout
        table_keys = [k for k, v in self.metrics.items() if isinstance(v[0], Score)]
        sorted_cols = sorted(self.metrics[table_keys[0]][0].score_dict)
        longest_key = max(len(k) for k in self.metrics)
        header = (
            " " * longest_key + " | " + " | ".join(k[:6].rjust(6) for k in sorted_cols)
        )

        print(header, file=outfile)
        output_dict = {}
        for key in table_keys:
            avg_metric = self.average(self.metrics[key])
            if sum(avg_metric.score_dict.values()) == 0.0:
                continue
            output_dict[key] = str(avg_metric)
            if "top" in key:
                print(
                    f"{C.OKBLUE if color else ''}{key.ljust(longest_key)} | {avg_metric}{C.ENDC if color else ''}",
                    file=outfile,
                )
            else:
                print(f"{key.ljust(longest_key)} | {avg_metric}", file=outfile)
        print(file=outfile)

        for key, value in self.metrics.items():
            if key in table_keys:
                continue
            print(
                f"{key.ljust(longest_key)} | " + f"{self.average(value):.2f}".rjust(6),
                file=outfile,
            )

    def to_dict(self):
        result = {}
        for key, value in self.metrics.items():
            avg_metric = self.average(value)
            if isinstance(avg_metric, Score):
                if sum(avg_metric.score_dict.values()) == 0.0:
                    continue
                for k, v in avg_metric.score_dict.items():
                    result[f"{key}_{k}"] = v
            else:
                result[key] = avg_metric
        return result

    def geomean(self, score):
        eps = 1e-4

        geo_mean = 1
        for v in score.values():
            geo_mean *= v + eps
        geo_mean **= 1 / len(self.eval_metrics)

        return geo_mean

    def evaluate(self, item):
        gold = item["gold"]
        hypos = item["hypos"]
        cached_scores = item.get("cached_scores", [])
        geomeans = []
        if cached_scores == []:
            scores = self.scorer.score(gold, hypos)
        else:
            scores = [Score(d["score_dict"], d["is_pct"]) for d in cached_scores]

        # TODO: actually cache scores in this location!
        item["cached_scores"] = [score.__dict__ for score in scores]

        geomeans = []
        for score in scores:
            geomeans.append(score.geomean())

        max_idx = np.argmax(geomeans)
        min_idx = np.argmin(geomeans)

        amt_to_strip = len("rerank_scores_")
        rerank_types = [
            k[amt_to_strip:] for k in item if k.startswith("rerank_scores_")
        ]
        rerank_types.append("lprobs")

        for rerank_type in rerank_types:
            top_rerank_key = f"top_rerank_{rerank_type}"
            corr_key = f"corr_{rerank_type}"
            pvalue_key = f"pvalue_{rerank_type}"
            if (
                rerank_type != "lprobs"
                and top_rerank_key in item
                and corr_key in item
                and pvalue_key in item
            ):
                max_rerank_score = item[top_rerank_key]
                correlations = item[corr_key]
                pvalues = item[pvalue_key]
            else:
                rerank_scores = (
                    item[f"rerank_scores_{rerank_type}"]
                    if rerank_type != "lprobs"
                    else item["lprobs"]
                )
                max_rerank_idx = np.argmax(rerank_scores)
                max_rerank_score = scores[max_rerank_idx].score_dict

                correlations = {}
                pvalues = {}
                for metric in self.eval_metrics:

                    def statistic(x):
                        return spearmanr(x, rerank_scores).statistic

                    if len(hypos) > 1 and self.report_corr:
                        res = permutation_test(
                            ([s[metric] for s in scores],),
                            statistic,
                            permutation_type="pairings",
                            n_resamples=500,
                        )

                        correlations[metric] = (
                            0.0 if np.isnan(res.statistic) else res.statistic
                        )
                        pvalues[metric] = 0.0 if np.isnan(res.pvalue) else res.pvalue
                    else:
                        correlations[metric] = 0.0
                        pvalues[metric] = 0.0

                item[f"top_rerank_{rerank_type}"] = max_rerank_score
                item[f"corr_{rerank_type}"] = correlations
                item[f"pvalue_{rerank_type}"] = pvalues

            self.metrics[f"top_rerank_{rerank_type}"].append(Score(max_rerank_score))
            self.metrics[f"corr_{rerank_type}"].append(Score(correlations, pct=False))
            self.metrics[f"pvalue_{rerank_type}"].append(Score(pvalues, pct=False))

        self.metrics["top_orig"].append(scores[0])
        self.metrics["top1"].append(scores[max_idx])
        self.metrics["bottom1"].append(scores[min_idx])
        self.metrics["avg_score"].append(
            sum(scores, start=self.zero_score) / len(hypos)
        )
        self.metrics["unique"].append(len(set(hypos)))
        if len(hypos) > 1:
            self.metrics["selfbleu"].append(
                self_bleu(hypos, num_refs=min(5, len(hypos) - 1))
            )

    def score_set(self, outputs):
        for output in tqdm(outputs):
            self.evaluate(output)
        return outputs


def run_eval(args):
    eval_metrics = [x.strip() for x in args.eval_metrics.split(",")]

    metric_tracker = Metrics(eval_metrics)

    with jsonlines.open(args.outfile) as f:
        lines = list(iter(f))

    metric_tracker.score_set(lines)
    metric_tracker.output()

    with jsonlines.open(args.outfile, "w") as f:
        f.write_all(lines)


if __name__ == "__main__":
    parser = get_parser()  # false for all topk gen implies we're doing reranking/eval
    args = parser.parse_args()
    run_eval(args)
