from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import torch 
from tqdm import tqdm
import jsonlines
import json
import sys

dataset = load_dataset("cnn_dailymail", "3.0.0")
print(dataset)


dir = "cnndm-temp1"
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum")

start = int(sys.argv[1]) #3200
end = int(sys.argv[2]) #3250
split = "validation"
temp = 0.1

all_outputs = []
with jsonlines.open(f"temp-samp-{temp}-new.jsonl", 'a') as f:
    for i in tqdm(range(start, end)):
        dp = dataset[split]["article"][i]
        input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=1024)

        outputs = model.generate(input_ids, do_sample=True, max_length=50, temperature=temp, num_return_sequences=25)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print(outputs)
        o = {"document": dp, "gold": dataset[split]["highlights"][i], "id": dataset[split]["id"][i], \
            "all_50": outputs, "num_unique": len(set(outputs))}
        f.write(o)
        #print(o)
        with open(dir + "/" + str(i) + ".json", 'w') as g:
            json.dump(obj=o, fp=g)

