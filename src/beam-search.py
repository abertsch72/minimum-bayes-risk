from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import torch 
from tqdm import tqdm
import jsonlines

all_outputs = []

def run_beam_search(args):
    bs_start = args.start_idx
    bs_end = args.end_idx
    bs_split = args.split

    dataset = load_dataset("xsum")
    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    model = BartForConditionalGeneration.from_pretrained(args.hf_model_name)

    for i in tqdm(range(bs_start, bs_end)):
        dp = dataset[bs_split]["document"][i]
        input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=1024).cuda()

        outputs = model.generate(input_ids, do_sample=False, max_length=70, num_beams=num, num_return_sequences=num)
        outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        all_outputs.append({
            "document": dp, 
            "gold": dataset[bs_split]["summary"][i], 
            "id": dataset[bs_split]["id"][i],
            "hypos": outputs_decoded, 
            "num_unique": len(set(outputs_decoded))
        })
        #print(all_outputs)
    with jsonlines.open(args.outfile, "w") as f:
        f.write_all(all_outputs)


if __name__ == '__main__':
    run_beam_search()
