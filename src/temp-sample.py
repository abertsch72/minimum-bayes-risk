from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import torch 
from tqdm import tqdm
import jsonlines
import json
import sys

dir = "xsum-new-temp1"
dataset = load_dataset("xsum")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum").to('cuda')
input_name = "document"
output_name = "summary"

"""
dataset = load_dataset("cnn_dailymail", "3.0.0")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to('cuda')
input_name = "article"
output_name = "highlights"
"""

start = int(sys.argv[1]) #3200
end = int(sys.argv[2]) #3250
split = "validation"
temp = 0.1

all_outputs = []
with jsonlines.open(f"xsum-temp-samp-{temp}.jsonl", 'a') as f:
    for i in tqdm(range(start, end)):
        dp = dataset[split][input_name][i]
        input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=1024).cuda()

        outputs = model.generate(input_ids, do_sample=True, max_length=70, num_beams=1, temperature=temp, num_return_sequences=50)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print(outputs)
        o = {"document": dp, "gold": dataset[split][output_name][i], "id": dataset[split]["id"][i], \
            "all_50": outputs, "num_unique": len(set(outputs))}
        f.write(o)



