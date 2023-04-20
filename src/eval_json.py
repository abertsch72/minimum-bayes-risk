from rouge_score import rouge_scorer
from recom_search.evaluation.eval_bench import self_bleu

class Metrics:
    class Rouge:
        def __init__(self, r1=0, r2=0, rL=0):
            self.r1 = r1
            self.r2 = r2
            self.rL = rL
        def format(self, num):
            return str(round(100 * num, 2))
        def __str__(self):
            return f"{format(self.r1)} / {format(self.r2)} / {format(self.rL)}"

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

    def __init__(self):
        self.rerank_metrics = ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(self.rerank_metrics)

        self.metrics = {"unique_summaries": [], "top1_rouge": self.RougeList(), "bottom1_rouge": self.RougeList(), "selfbleu": [], "avg_score":  self.RougeList(), "correlation": []}


    def output(self):
        def average(seq):
            if isinstance(seq, self.RougeList):
                return seq.average()
            elif len(seq) == 0:
                return 0
            else:
                return sum(seq)/len(seq)

        for key in self.metrics.keys():
            print(f"{key}\t{average(self.metrics[key])}")

    def geomean(self, score):
        eps = 1e-4

        geo_mean = 1
        for m in self.rerank_metrics:
            geo_mean *= (score[m].fmeasure + eps)
            geo_mean **= (1/len(self.rerank_metrics))
        
        return geo_mean

    def eval(self, item):
        gold = item['gold']
        options = item['all_50']
        unique = set(options)
        all_scores = []
        weighted_scores = {}
        geomeans = []
        for opt in unique:
            score = self.scorer.score(gold, opt)
            geomeans.append(self.geomean(score))
            all_scores.append([score[m].fmeasure for m in self.rerank_metrics])

            for metric in self.rerank_metrics:
                weighted_scores[metric] = weighted_scores.get(metric, 0) + score[metric].fmeasure * options.count(opt)


        max_score = all_scores[geomeans.index(max(geomeans))]
        min_score = all_scores[geomeans.index(min(geomeans))]

        self.metrics['unique_summaries'].append(item['num_unique'])
        self.metrics['top1_rouge'].append(self.Rouge(*max_score))
        self.metrics['bottom1_rouge'].append(self.Rouge(*min_score))
        
        all_r1 = weighted_scores['rouge1']/len(options)
        all_r2 = weighted_scores['rouge2']/len(options)
        all_rL = weighted_scores['rougeL']/len(options)

        self.metrics['avg_score'].append(self.Rouge(all_r1, all_r2, all_rL))


        self.most_freq_summ_rouge = []
        self.metrics['selfbleu'].append(self_bleu(options))
        self.correlation = []

    def score_set(self, outputs):
        for output in outputs:
            self.eval(output)



metric_tracker = Metrics() 

import sys
file = sys.argv[1] #"topk-outputs/temp07.jsonl"
import jsonlines
with jsonlines.open(file) as f:
    lines = list(iter(f))

metric_tracker.score_set(lines)
metric_tracker.output()

