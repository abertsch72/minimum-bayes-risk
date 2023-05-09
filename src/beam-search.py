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
        #print(dp)
        input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=1024)

        outputs = model.generate(input_ids, do_sample=False, max_length=50, num_beams=50, num_return_sequences=50)
        outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #print(outputs)
        if len(set(outputs_decoded)) < 50:
            print("!!!")
            dup = set([i for i in outputs_decoded if outputs_decoded.count(i) > 1])
            for repeat in dup:
                print(repeat)
                indices = [i for i in range(len(outputs_decoded)) if outputs_decoded[i] == repeat]
                for ind in indices:
                    print(outputs[ind])
            print(len(set(outputs_decoded)))

            print(len(set(outputs)))
            #print(outputs)
        all_outputs.append({"document": dp, "gold": dataset[bs_split]["summary"][i], "id": dataset[bs_split]["id"][i], \
            "hypos": outputs, "num_unique": len(set(outputs))})
        #print(all_outputs)

#        f.write(all_outputs[-1])

if __name__ == '__main__':
    run_beam_search()
