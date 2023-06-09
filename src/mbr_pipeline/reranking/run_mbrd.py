"""
Taken with minor modification from crowd sampling repo
https://github.com/suzgunmirac/crowd-sampling/blob/main/run_mbrd.py
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter

import datasets
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("./")
sys.path.append("./src/")
import src


def main():
    """
    Example:
        python run_mbrd.py \
            --path outputs/bigbench/bigbench_ipa_codex_original_N16.json \
            --mode bertscore \
            --save_path outputs/bigbench/bigbench_ipa_codex_bertscore_N16.json
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', type=str, default='outputs/bigbench/bigbench_ipa_codex_original_N16.json')
    parser.add_argument("--mode", type=str, default="bertscore")
    # parser.add_argument('--save_path', type=str, default='outputs/bigbench/bigbench_ipa_codex_bertscore_N16.json')
    args = parser.parse_args()

    from rouge_score import rouge_scorer

    full_rouge_scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    # Load Metric
    if args.mode == "mbrd":
        bleurt = datasets.load_metric("bleurt")
    elif args.mode == "bertscore":
        bertscore = datasets.load_metric("bertscore")
    elif args.mode == "lcs":
        # pip3 install -i https://test.pypi.org/simple/ string2string==0.0.3
        from string2string.edit_distance import EditDistAlgs

        algs_unit = EditDistAlgs()

    mode = args.mode

    # Read

    # results_dir = "/Users/axie/Desktop/sum_xsum_temp_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9" # temperature sampling
    # results_dir = "/home/alexx/lattice-search/output/data/sum_xsum_temp_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9"

    results_dir = "/home/alexx/lattice-search/output/data/sum_xsum_temp_16_35_False_0.4_False_False_4_5_zip_-1_0.0_0.9_temp0.1"

    result_files = os.listdir(results_dir)

    def read_result(dir, file):
        with open(os.path.join(dir, file), "rb") as f:
            x = pickle.load(f)
        return x

    all_scores = []

    # Loop over Outputs
    pbar = tqdm(total=len(result_files))

    for i in range(len(result_files)):
        file = result_files[i]
        output = read_result(results_dir, file)

        outputs = output.output

        n = len(outputs)
        matrix = np.zeros((n, n))

        # MBR with BLEURT
        if mode == "bleurt":
            for j1, cand1 in enumerate(outputs[i]):
                for j2, cand2 in enumerate(outputs[i]):
                    with torch.inference_mode():
                        score = bleurt.compute(predictions=[cand1], references=[cand2])[
                            "scores"
                        ][0]
                        matrix[j1][j2] = score
            matrix = np.sum(matrix, axis=1)
            index = np.argmax(matrix)

        # MBRD with BERTScore
        elif mode == "bertscore":
            for j1 in range(len(outputs)):
                for j2 in range(j1 + 1, len(outputs)):
                    cand1 = outputs[j1]
                    cand2 = outputs[j2]
                    with torch.inference_mode():
                        score = bertscore.compute(
                            predictions=[cand1],
                            references=[cand2],
                            lang="en",
                            device=device,
                        )["f1"][0]
                        matrix[j1][j2] = matrix[j2][j1] = score
            matrix = np.sum(matrix, axis=1)
            index = np.argmax(matrix)

        # Majority Voting
        elif mode == "majority":
            counter = Counter(outputs[i])
            txt, count = counter.most_common(1)[0]
            if count > 1:
                index = outputs[i].index(txt)
            else:
                index = 0

        # LCS (Longest Common Substring)
        elif mode == "lcs":
            for j1, cand1 in enumerate(outputs[i]):
                cand1_split = cand1.split(" ")
                for j2, cand2 in enumerate(outputs[i]):
                    cand2_split = cand2.split(" ")
                    max_length = max(len(cand1_split), len(cand2_split))
                    dist, _ = algs_unit.longest_common_subsequence(
                        cand1_split,
                        cand2_split,
                        printBacktrack=False,
                        boolListOfList=True,
                    )
                    score = dist / max_length
                    matrix[j1][j2] = score
            matrix = np.sum(matrix, axis=1)
            index = np.argmax(matrix)

        # Choose First / Sample Once
        elif mode == "sample_once":
            index = 0

        # Random Choice
        elif mode == "random":
            index = np.random.randint(0, n)

        # Not Implemented Yet...
        else:
            raise NotImplementedError()

        hypo = output.output[index]
        score = full_rouge_scorer.score(output.reference, hypo)
        all_scores.append(score)

        curr_r1 = sum(s["rouge1"].fmeasure for s in all_scores) / len(all_scores)
        curr_r2 = sum(s["rouge2"].fmeasure for s in all_scores) / len(all_scores)
        curr_rL = sum(s["rougeL"].fmeasure for s in all_scores) / len(all_scores)

        pbar.set_description(f"r1={curr_r1}/r2={curr_r2}/rL={curr_rL}")
        pbar.update(1)

    r1 = sum(s["rouge1"].fmeasure for s in all_scores) / len(all_scores)
    r2 = sum(s["rouge2"].fmeasure for s in all_scores) / len(all_scores)
    rL = sum(s["rougeL"].fmeasure for s in all_scores) / len(all_scores)

    print("Mean r1 =", r1)
    print("Mean r2 =", r2)
    print("Mean rL =", rL)

    # Save
    # with open(save_path, 'w') as f:
    #     json.dump(new_data, f)


if __name__ == "__main__":
    main()
