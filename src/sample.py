from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import torch 
from tqdm import tqdm
import jsonlines
from typing import Sequence

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", choices=['xsum', 'cnndm'], help="dataset to generate over", required=True)
parser.add_argument("--start", type=int, help="first (ordered) example to generate", required=True)
parser.add_argument("--end", type=int, help="example (ordered) to stop generating at", required=True)
parser.add_argument("--sample_strat", choices=['beam_search', 'temp_sample', 'nucleus_sample'], required=True)
parser.add_argument("--name", default="out", help="any additional info for filename")
parser.add_argument("--modelname", default="facebook/bart-large-xsum", help="model to run")
parser.add_argument("--split", default="validation", help="split to use")
parser.add_argument("--start", type=int, help="first (ordered) example to generate")
parser.add_argument("--num_seqs", type=int, help="number of sequences to generate")
parser.add_argument("--max_length", type=int, default=50, help="generation max len")
parser.add_argument("--num_beams", type=int, help="number of beams to use for beam search")
parser.add_argument("--temperature", type=float, help="temperature for temperature sampling")
parser.add_argument("--nucleus", type=float, help="nucleus sampling parameter")



def beam_search(input_ids, model, num_seqs=50, max_length=50, num_beams=50):
    return model.generate(input_ids, do_sample=False, max_length=max_length, num_beams=num_beams, num_return_sequences=num_seqs)

def temp_sample(input_ids, model, num_seqs=50, max_length=50, temperature=0.7):
    return model.generate(input_ids, do_sample=True, max_length=max_length, temperature=temperature, num_return_sequences=num_seqs)

def nucl_sample(input_ids, model, num_seqs=50, max_length=50, nucleus=0.6):
    return model.generate(input_ids, do_sample=True, max_length=max_length, top_p=nucleus, num_return_sequences=num_seqs)


def sample(dataset, start, end, sample_strat, save_partials=False, name="out", modelname="facebook/bart-large-xsum", split="validation", **kwargs):
    strategy_lookup = {
        "beam_search": beam_search,
        "temp_sample": temp_sample,
        "nucleus_sample": nucl_sample,
    }
    if dataset=="cnndm": # use the version number
        dataset = load_dataset(dataset, "3.0.0", split=split)
    else:
        dataset = load_dataset(dataset, split=split)

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = BartForConditionalGeneration.from_pretrained(modelname)

    for arg in kwargs:
        name += f"_{arg}={kwargs[arg]}"

    with jsonlines.open(f"{sample_strat}_{dataset}_{name}.jsonl", 'a') as f:
        for i in tqdm(range(start, end)):
            dp = dataset["document"][i]
            input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=1024)

            outputs = strategy_lookup[sample_strat](input_ids, model, **kwargs) #
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            outputs= {"document": dp, "gold": dataset["summary"][i], "id": dataset["id"][i], \
                "all_50": outputs, "num_unique": len(set(outputs))}

            f.write(outputs)

if __name__ == "__main__":
    parser.parse_args()
    sample(**parser._get_args())