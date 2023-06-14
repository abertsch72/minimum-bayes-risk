import jsonlines
import sys

filename = sys.argv[1]


avg_scores = []
top_scores = []
num_unique = []
self_bleus = []
bottom_scores = []
logprob_tops = []

logprob_rerankings = {}
freq_rerankings = {}
with jsonlines.open(filename) as f:
    data = list(f.iter())

for dp in data:
    logprob_tops.append(dp['top_rerank_lprobs'])
    num_unique.append(dp['num_unique'])

    for key in dp.keys():
        if key.startswith('top_rerank') and key != 'top_rerank_lprobs':
            if 'freq' in key:
                freq_rerankings[key] = freq_rerankings.get(key, [])
                freq_rerankings[key].append(dp[key])
            else:
                logprob_rerankings[key] = logprob_rerankings.get(key, [])
                logprob_rerankings[key].append(dp[key])

    #print(dp['top_rerank_lprobs'])
    #break    


def avg_over_dicts(list_dicts):
    keys = list_dicts[0].keys()
    count = len(list_dicts)
    avg_map = {k:0 for k in keys}
    for d in list_dicts:
        for key in keys:
            avg_map[key] += d[key]
    for key in keys:
        avg_map[key] /= count
    #print(avg_map)
    return avg_map

logprob_tops = avg_over_dicts(logprob_tops)    
print(f"logprob_tops:\t{logprob_tops}")
num_unique = sum(num_unique) / len(num_unique)
print(f"num_unique:\t{num_unique}")

print("freq results")
for key in freq_rerankings:
    print(f"{key}:\t{avg_over_dicts(freq_rerankings[key])}")

print("logprob results")
for key in logprob_rerankings:
    print(f"{key}:\t{avg_over_dicts(logprob_rerankings[key])}")

"""
dict_keys(['document', 'gold', 'id', 'hypos', 'lprobs', 'num_unique', 'top_rerank_lprobs', 'corr_lprobs', 'pvalue_lprobs', 
'rerank_scores_bertscoretemp-inf', 'top_rerank_bertscoretemp-inf', 'corr_bertscoretemp-inf', 'pvalue_bertscoretemp-inf', 
'rerank_scores_rouge1temp-inf', 'top_rerank_rouge1temp-inf', 'corr_rouge1temp-inf', 'pvalue_rouge1temp-inf', 
'rerank_scores_rouge2temp-inf', 'top_rerank_rouge2temp-inf', 'corr_rouge2temp-inf', 'pvalue_rouge2temp-inf', 
'rerank_scores_rouge3temp-inf', 'top_rerank_rouge3temp-inf', 'corr_rouge3temp-inf', '
pvalue_rouge3temp-inf', 'rerank_scores_rouge4temp-inf', 'top_rerank_rouge4temp-inf', 'corr_rouge4temp-inf', 
'pvalue_rouge4temp-inf', 'rerank_scores_rouge5temp-inf', 'top_rerank_rouge5temp-inf', 'corr_rouge5temp-inf', 
'pvalue_rouge5temp-inf', 'rerank_scores_rouge6temp-inf', 
'top_rerank_rouge6temp-inf', 'corr_rouge6temp-inf', 'pvalue_rouge6temp-inf'])
"""
