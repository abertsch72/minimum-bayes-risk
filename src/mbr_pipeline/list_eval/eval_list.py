import jsonlines
import sys
import os


def get_scores(filename):
    score_info = ""

    empty = " , , , , ,"
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

    all_expected_keys = []
    all_present_keys = []

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
                all_present_keys.append(key)

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
            avg_map[key] = round(100 * avg_map[key], 2)
        #print(avg_map)
        return avg_map

    logprob_tops = avg_over_dicts(logprob_tops)    
    #print(f"logprob_tops:\t{logprob_tops}")
    num_unique = sum(num_unique) / len(num_unique)
    #print(f"num_unique:\t{num_unique}")
    score_info += str(round(num_unique,2)) + ", "
    for metric in logprob_tops:
        score_info += str(logprob_tops[metric]) + ", "
    #print(score_info)

    max_diff = 0
    max_result = {}
    #print("freq results")
    for key in freq_rerankings:
        result = avg_over_dicts(freq_rerankings[key])
        #print(f"{key}:\t{result}")
        diff = 0
        for metric in result:
            diff += (result[metric] - logprob_tops[metric])
        if diff > max_diff:
            max_diff = diff
        max_result = result


    def add_to_score(max_result, score_info):
        if max_result != {}:
            abs, rel = "", "" 
            for key in max_result:
                abs += str(max_result[key]) + ", "
                rel += str(round(max_result[key] - logprob_tops[key], 2)) + ", "
            score_info += abs + rel
        else:
            score_info += empty + empty
        return score_info

    """print("!!!!!!!max freq gain over logprobs is")
    print(max_result)
    for key in max_result:
        print(f"\t{key}:\t{max_result[key] - logprob_tops[key]}")
    """
    score_info = add_to_score(max_result, score_info)
    #print(score_info)        
   
    max_diff = 0
    max_result = {}
    #print("logprob results")
    for key in logprob_rerankings:
        result = avg_over_dicts(logprob_rerankings[key])
        #print(f"{key}:\t{result}")
        diff = 0
        for metric in result:
            diff += (result[metric] - logprob_tops[metric])
        if diff > max_diff:
            max_diff = diff
        max_result = result

    score_info = add_to_score(max_result, score_info)
    """print("!!!!!!!max logprobs mbr gain over logprobs is")
    print(max_result)
    for key in max_result:
        print(f"\t{key}:\t{max_result[key] - logprob_tops[key]}")
    """
    return score_info



csv = "dataset,model,sample size,sample method, num unique, top_rerank_lprobs,,,,,gain from freq MBR,,,,, gain from logprobs MBR"


sampling_dir = sys.argv[1]
for dataset_name in os.listdir(sampling_dir):
    for modelname in os.listdir(os.path.join(sampling_dir, dataset_name)):
        for size in os.listdir(os.path.join(sampling_dir, dataset_name, modelname)):
            for strategy_name in os.listdir(os.path.join(sampling_dir, dataset_name, modelname, size)):
                model_name = '-'.join(modelname.split('-')[1:])
                this_csv = f"{dataset_name},{model_name},{size},{strategy_name.strip('.jsonl')}"
                try:
                    this_csv += ", " + get_scores(os.path.join(sampling_dir, dataset_name, modelname, size, strategy_name))
                    print(this_csv)
                except:
                    pass #print("error")


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
