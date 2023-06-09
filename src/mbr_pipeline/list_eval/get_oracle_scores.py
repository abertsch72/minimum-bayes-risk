import json
import os
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np

full_rouge_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)

results_dir = (
    "output/output/text/sum_xsum_bs_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9"
)
# results_dir = "output/output/text/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9"
result_files = os.listdir(results_dir)

beam_search = True

r1_scores = []
r2_scores = []
rl_scores = []

random_r1_scores = []
random_r2_scores = []
random_rl_scores = []

for file in tqdm(result_files):
    if not file.endswith("json"):
        continue
    with open(os.path.join(results_dir, file)) as f:
        metrics_dict = json.load(f)

    pred = metrics_dict["oracle_1"][0][0]
    tgt = metrics_dict["tgt"]

    scores = full_rouge_scorer.score(tgt, pred)
    r1_scores.append(scores["rouge1"].fmeasure)
    r2_scores.append(scores["rouge2"].fmeasure)
    rl_scores.append(scores["rougeL"].fmeasure)

    avg_random_r1 = 0
    avg_random_r2 = 0
    avg_random_rl = 0

    key = "oracle_20" if beam_search else "sample"

    num_samples = len(metrics_dict[key])
    for sample in metrics_dict[key]:
        random_pred = sample[0]
        random_scores = full_rouge_scorer.score(tgt, random_pred)
        avg_random_r1 += random_scores["rouge1"].fmeasure
        avg_random_r2 += random_scores["rouge2"].fmeasure
        avg_random_rl += random_scores["rougeL"].fmeasure
    random_r1_scores.append(avg_random_r1 / num_samples)
    random_r2_scores.append(avg_random_r2 / num_samples)
    random_rl_scores.append(avg_random_rl / num_samples)

    break


print(f"oracle rouge-1: mean = {np.mean(r1_scores)}, stddev = {np.std(r1_scores)}")
print(f"oracle rouge-2: mean = {np.mean(r2_scores)}, stddev = {np.std(r2_scores)}")
print(f"oracle rouge-l: mean = {np.mean(rl_scores)}, stddev = {np.std(rl_scores)}")

print(
    f"random rouge-1: mean = {np.mean(random_r1_scores)}, stddev = {np.std(random_r1_scores)}"
)
print(
    f"random rouge-2: mean = {np.mean(random_r2_scores)}, stddev = {np.std(random_r2_scores)}"
)
print(
    f"random rouge-l: mean = {np.mean(random_rl_scores)}, stddev = {np.std(random_rl_scores)}"
)
