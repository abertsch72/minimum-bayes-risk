from args import get_parser
from rouge_score import rouge_scorer
from scorers import (
    Scorer, Score, self_bleu,
    rescore_bartscore, rescore_bertscore, rescore_rouge
)
from collections import defaultdict
import numpy as np
import jsonlines
from tqdm import tqdm
from scipy.special import softmax
from scipy.stats import spearmanr, permutation_test


class Metrics:
    class Rouge:
        def __init__(self, r1=0, r2=0, rL=0):
            self.r1 = r1
            self.r2 = r2
            self.rL = rL
        def format(self, num):
            return str(round(100 * num, 2))
        def __str__(self):
            return f"{self.format(self.r1)} / {self.format(self.r2)} / {self.format(self.rL)}"

    class RougeList:
        def __init__(self):
            self.seq = []
        
        def append(self, item):
            assert isinstance(item, Metrics.Rouge)

            self.seq.append(item)
        
        def average(self):
            if len(self.seq) == 0:
                return Metrics.Rouge()

            r1 = sum([item.r1 for item in self.seq]) / len(self.seq)
            r2 = sum([item.r2 for item in self.seq]) / len(self.seq)
            rL = sum([item.rL for item in self.seq]) / len(self.seq)

            return Metrics.Rouge(r1, r2, rL)
        
        def __len__(self):
            return len(self.seq)


    def __init__(self, args, eval_metrics, rerank_metric):
        self.args = args
        self.scorer = Scorer(eval_metrics)
        self.eval_metrics = eval_metrics
        self.rerank_metric = rerank_metric
        if "rouge" in rerank_metric:
            self.rerank_fn = lambda hypos, probs: rescore_rouge(hypos, probs, (self.rerank_metric,))
        elif rerank_metric == 'bertscore':
            self.rerank_fn = rescore_bertscore
        else:
            self.rerank_fn = None
            # raise Exception("Unknown rerank metric:", rerank_metric)
            
        self.zero_score = Score({m: 0 for m in eval_metrics})

        # self.metrics = {"unique": [], "top1": [], "bottom1": [], "selfbleu": [], "avg_score":  [], "correlation": []}
        self.metrics = defaultdict(lambda: [])


    def output(self):
        def average(seq):
            if len(seq) == 0:
                return 0
            start = 0
            if isinstance(seq[0], Score):
                start = self.zero_score
            return sum(seq, start=start) / len(seq)

        table_keys = ['top1', 'bottom1', 'avg_score', 'top_rerank', 'corr', 'pvalue']
        sorted_keys = sorted(self.metrics[table_keys[0]][0].score_dict)
        longest_key = max(len(k) for k in self.metrics)
        header = " "*longest_key + " | " + " | ".join(k[:6].rjust(6) for k in sorted_keys)
        
        print(header)
        for key in table_keys:
            print(f"{key.ljust(longest_key)} | {average(self.metrics[key])}")
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
        lprobs = np.array(item['lprobs'])
        hypos = item['hypos']
        geomeans = []
        scores = self.scorer.score(gold, hypos)

        geomeans = []
        for score in scores:
            geomeans.append(score.geomean())
        
        max_idx = np.argmax(geomeans)
        min_idx = np.argmin(geomeans)
        
        if self.rerank_fn is not None:
            probs = softmax(lprobs / self.args.rerank_temp)
            rerank_scores = self.rerank_fn(hypos, probs)
            max_rerank_idx = np.argmax(rerank_scores)
            correlations = {}
            pvalues = {}
            for metric in self.eval_metrics:
                def statistic(x):
                    return spearmanr(x, rerank_scores).statistic
                
                res = permutation_test(([s[metric] for s in scores],), statistic, 
                                       permutation_type='pairings', n_resamples=500)

                correlations[metric] = 0.0 if np.isnan(res.statistic) else res.statistic
                pvalues[metric] = 0.0 if np.isnan(res.pvalue) else res.pvalue

            self.metrics['corr'].append(Score(correlations))
            self.metrics['pvalue'].append(Score(pvalues))
            self.metrics['top_rerank'].append(scores[max_rerank_idx])

        self.metrics['top1'].append(scores[max_idx])
        self.metrics['bottom1'].append(scores[min_idx])
        self.metrics['avg_score'].append(sum(scores, start=self.zero_score) / len(hypos))
        self.metrics['unique'].append(len(set(hypos)))
        self.metrics['selfbleu'].append(self_bleu(hypos))

    def score_set(self, outputs):
        for output in tqdm(outputs):
            self.evaluate(output)


if __name__ == '__main__':
    parser = get_parser() # false for all topk gen implies we're doing reranking/eval
    args = parser.parse_args()
    eval_metrics = [x.strip() for x in args.eval_metrics.split(',')]

    metric_tracker = Metrics(args, eval_metrics, args.rerank_metric) 

    with jsonlines.open(args.outfile) as f:
        lines = list(iter(f))

    metric_tracker.score_set(lines)
    metric_tracker.output()

