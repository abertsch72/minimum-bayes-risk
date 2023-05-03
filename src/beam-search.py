from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import torch 
from tqdm import tqdm
import jsonlines
import sys 

dataset = load_dataset("xsum")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum").to('cuda')
input_name = "document"
output_name = "summary"
name = "xsum"

"""
dataset = load_dataset("cnn_dailymail", "3.0.0")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to('cuda')
input_name = "article"
output_name = "highlights"
"""


start = int(sys.argv[1])
end = int(sys.argv[2])
split = "validation"
num = int(sys.argv[3])

all_outputs = []
with jsonlines.open(f"{num}-{name}-beam-search.jsonl", 'a') as f:
    for i in tqdm(range(start, end)):
        dp = dataset[split][input_name][i]
        #print(dp)
        input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=1024).cuda()

        outputs = model.generate(input_ids, do_sample=False, max_length=70, num_beams=num, num_return_sequences=num)
        outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.append({"document": dp, "gold": dataset[split][output_name][i], "id": dataset[split]["id"][i], \
            "all_50": outputs_decoded, "num_unique": len(set(outputs_decoded))})

        f.write(all_outputs[-1])


