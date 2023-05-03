from args import get_parser
from rouge_score import rouge_scorer
from recom_search.evaluation.eval_bench import self_bleu
import datasets

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
        def __init__(self, r1=0, r2=0, rL=0, B=0):
            self.r1 = r1
            self.r2 = r2
            self.rL = rL
            #self.B = B
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

            r1 = round(100 * (sum([item.r1 for item in self.seq]) / len(self.seq)), 2)
            r2 = round(100 * (sum([item.r2 for item in self.seq]) / len(self.seq)), 2)
            rL = round(100 * (sum([item.rL for item in self.seq]) / len(self.seq)), 2)
            #B =  round(100 * (sum([item.B  for item in self.seq]) / len(self.seq)), 2)
            return Metrics.Rouge(r1, r2, rL)
        
        def __len__(self):
            return len(self.seq)

    def __init__(self, args, eval_metrics, rerank_metric):
        self.rerank_metrics = ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(self.rerank_metrics)
        #self.bertscorer = datasets.load_metric("bertscore")
        self.metrics = {"unique_summaries": [], "top1_rouge": self.RougeList(), "bottom1_rouge": self.RougeList(), "selfbleu": [], "avg_score":  self.RougeList(), "correlation": []}

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
        #bertscores = []
        for opt in unique:
            score = self.scorer.score(gold, opt)
            #bertscores.append(self.bertscorer.compute(predictions=[opt], references=[gold], lang='en')['f1'][0])

            geomeans.append(self.geomean(score))
            all_scores.append([score[m].fmeasure for m in self.rerank_metrics])

            for metric in self.rerank_metrics:
                weighted_scores[metric] = weighted_scores.get(metric, 0) + score[metric].fmeasure * options.count(opt)

            #weighted_scores['bertscore'] = weighted_scores.get('bertscore', 0) + bertscores[-1] * options.count(opt)

        maxsc = geomeans.index(max(geomeans))
        minsc = geomeans.index(min(geomeans))
        max_score = all_scores[maxsc]
        min_score = all_scores[minsc]

        self.metrics['unique_summaries'].append(item['num_unique'])
        self.metrics['top1_rouge'].append(self.Rouge(*max_score))
        self.metrics['bottom1_rouge'].append(self.Rouge(*min_score))
        scores = self.scorer.score(gold, hypos)

        geomeans = []
        for score in scores:
            geomeans.append(score.geomean())
        
        all_r1 = weighted_scores['rouge1']/len(options)
        all_r2 = weighted_scores['rouge2']/len(options)
        all_rL = weighted_scores['rougeL']/len(options)
        #all_B = weighted_scores['bertscore']/len(options)
        self.metrics['avg_score'].append(self.Rouge(all_r1, all_r2, all_rL))

        #self.output()
        self.most_freq_summ_rouge = []
        self.correlation = []

        self.metrics['selfbleu'].append(self_bleu(options))

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

