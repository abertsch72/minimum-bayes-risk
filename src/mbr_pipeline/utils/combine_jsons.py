# temp sampling util: given a list of folders, combine files across them into a single jsonlines


import os
import json
import jsonlines


def sync_files(folderlist, filestart, fileend, outfile):
    combined = []
    for prefix in range(filestart, fileend):
        raw = []
        for folder in folderlist:
            with open(f"{folder}/{prefix}.json") as f:
                raw.append(json.load(f))
        data = {"document": raw[0]["document"], "gold": raw[0]["gold"]}
        joined = raw[0]["all_50"]
        for i in range(1, len(raw)):
            joined.extend(raw[i]["all_50"])
        data["all_50"] = joined
        data["num_unique"] = len(set(joined))
        combined.append(data)

    with jsonlines.open(outfile, "w") as f:
        f.write_all(combined)


sync_files(["temp-runs25", "temp-runs25-1"], 3000, 3250, "topk-outputs/temp07.jsonl")
