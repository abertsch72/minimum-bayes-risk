from rouge_score import rouge_scorer
import sacrebleu
from src.mbr_pipeline.list_eval.scorers import (
    Scorer, Score, self_bleu,
    rescore_bartscore, rescore_bertscore, rescore_rouge
)
from collections import defaultdict
import numpy as np
import jsonlines
from tqdm import tqdm
from scipy.special import softmax
from scipy.stats import spearmanr, permutation_test


class C:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Metrics:
    def __init__(self, eval_metrics):
        self.scorer = Scorer(eval_metrics)
        self.eval_metrics = eval_metrics

        self.report_corr = False
            
        self.zero_score = Score({m: 0 for m in eval_metrics})
        self.metrics = defaultdict(lambda: [])

    def output(self):
        def average(seq):
            if len(seq) == 0:
                return 0
            start = 0
            if isinstance(seq[0], Score):
                start = self.zero_score
            return sum(seq, start=start) / len(seq)

        table_keys = [k for k, v in self.metrics.items() if isinstance(v[0], Score)]
        sorted_cols = sorted(self.metrics[table_keys[0]][0].score_dict)
        longest_key = max(len(k) for k in self.metrics)
        header = " "*longest_key + " | " + " | ".join(k[:6].rjust(6) for k in sorted_cols)
        
        print(header)
        for key in table_keys:
            avg_metric = average(self.metrics[key])
            if sum(avg_metric.score_dict.values()) == 0.0:
                continue
            if "top" in key:
                print(f"{C.OKBLUE}{key.ljust(longest_key)} | {avg_metric}{C.ENDC}")
            else:
                print(f"{key.ljust(longest_key)} | {avg_metric}")
        print()

        for key in self.metrics.keys():
            if key in table_keys:
                continue
            print(f"{key.ljust(longest_key)} | " + f"{average(self.metrics[key]):.2f}".rjust(6))

    def geomean(self, score):
        eps = 1e-4

        geo_mean = 1
        for v in score.values():
            geo_mean *= (v + eps)
        geo_mean **= (1/len(self.rerank_metrics))
        
        return geo_mean

    def evaluate(self, item):
        gold = item['gold']
        hypos = item['hypos']
        geomeans = []
        scores = self.scorer.score(gold, hypos)

        geomeans = []
        for score in scores:
            geomeans.append(score.geomean())
        
        max_idx = np.argmax(geomeans)
        min_idx = np.argmin(geomeans)

        amt_to_strip = len("rerank_scores_")
        rerank_types = [k[amt_to_strip:] for k in item if k.startswith("rerank_scores_")]
        rerank_types.append("lprobs")

        for rerank_type in rerank_types:
            top_rerank_key = f'top_rerank_{rerank_type}'
            corr_key = f'corr_{rerank_type}'
            pvalue_key = f'pvalue_{rerank_type}'
            if top_rerank_key in item and corr_key in item and pvalue_key in item:
                max_rerank_score = item[top_rerank_key]
                correlations = item[corr_key]
                pvalues = item[pvalue_key]
            else:
                rerank_scores = item[f"rerank_scores_{rerank_type}"] if rerank_type != "lprobs" else [-prob for prob in item["lprobs"]]
                max_rerank_idx = np.argmax(rerank_scores)
                max_rerank_score = scores[max_rerank_idx].score_dict

                correlations = {}
                pvalues = {}
                for metric in self.eval_metrics:
                    def statistic(x):
                        return spearmanr(x, rerank_scores).statistic
                    
                    if len(hypos) > 1 and self.report_corr:
                        res = permutation_test(([s[metric] for s in scores],), statistic, 
                                            permutation_type='pairings', n_resamples=500)

                        correlations[metric] = 0.0 if np.isnan(res.statistic) else res.statistic
                        pvalues[metric] = 0.0 if np.isnan(res.pvalue) else res.pvalue
                    else:
                        correlations[metric] = 0.0
                        pvalues[metric] = 0.0

                item[f'top_rerank_{rerank_type}'] = max_rerank_score
                item[f'corr_{rerank_type}'] = correlations
                item[f'pvalue_{rerank_type}'] = pvalues

            self.metrics[f'top_rerank_{rerank_type}'].append(Score(max_rerank_score))
            self.metrics[f'corr_{rerank_type}'].append(Score(correlations, pct=False))
            self.metrics[f'pvalue_{rerank_type}'].append(Score(pvalues, pct=False))

        self.metrics['top_orig'].append(scores[0])
        self.metrics['top1'].append(scores[max_idx])
        self.metrics['bottom1'].append(scores[min_idx])
        self.metrics['avg_score'].append(sum(scores, start=self.zero_score) / len(hypos))
        self.metrics['unique'].append(len(set(hypos)))
        if len(hypos) > 1:
            self.metrics['selfbleu'].append(
                self_bleu(hypos, num_refs=min(5, len(hypos)))
            )

    def score_set(self, outputs):
        for output in tqdm(outputs):
            self.evaluate(output)
        return outputs

def run_eval(args):
    eval_metrics = [x.strip() for x in args.eval_metrics.split(',')]

    metric_tracker = Metrics(eval_metrics) 

    with jsonlines.open(args.outfile) as f:
        lines = list(iter(f))

    metric_tracker.score_set(lines)
    metric_tracker.output()

    with jsonlines.open(args.outfile, "w") as f:
        f.write_all(lines)



if __name__ == '__main__':
    parser = get_parser() # false for all topk gen implies we're doing reranking/eval
    args = parser.parse_args()
    run_eval(args)