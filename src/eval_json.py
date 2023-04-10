from rouge_score import rouge_scorer


def eval(item, scorer):
    rerank_metrics = ['rouge1', 'rouge2', 'rougeL']
    eps = 1e-4
    gold = item['gold']
    options = item['all_50']
    unique = set(options)
    all_scores = []
    geomeans = []
    for opt in unique:
        score = scorer.score(gold, opt)
        geo_mean = 1
        for m in rerank_metrics:
            geo_mean *= (score[m].fmeasure + eps)
            geo_mean **= (1/len(rerank_metrics))
        geomeans.append(geomean)
        all_scores.append([score[m].fmeasure for m in rerank_metrics])
    max_score = all_scores[geomeans.index(max(geomeans))]
    print(max_score)


