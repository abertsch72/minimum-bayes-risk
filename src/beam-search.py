from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import torch 
from tqdm import tqdm
import jsonlines

dataset = load_dataset("xsum")
print(dataset)


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum")

start = 3000
end = 3250
split = "validation"

all_outputs = []
with jsonlines.open(f"beam-search.jsonl", 'a') as f:
    for i in tqdm(range(start, end)):
        dp = dataset[split]["document"][i]
        #print(dp)
        input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=1024)

        outputs = model.generate(input_ids, do_sample=False, max_length=50, num_beams=50, num_return_sequences=50)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print(outputs)
        all_outputs.append({"document": dp, "gold": dataset[split]["summary"][i], "id": dataset[split]["id"][i], \
            "all_50": outputs, "num_unique": len(set(outputs))})
        #print(all_outputs)

        f.write(all_outputs[-1])
