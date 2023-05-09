from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import jsonlines
from typing import Sequence, Text, Union
from enum import Enum
from dataclasses import dataclass

class SamplingMethods:
    def beam_search(input_ids, model, num_seqs=50, max_length=50, num_beams=50):
        return model.generate(input_ids, do_sample=False, max_length=max_length, num_beams=num_beams, num_return_sequences=num_seqs)

    def temp_sample(input_ids, model, num_seqs=50, max_length=50, temperature=0.7):
        return model.generate(input_ids, do_sample=True, max_length=max_length, temperature=temperature, num_return_sequences=num_seqs)

    def nucl_sample(input_ids, model, num_seqs=50, max_length=50, nucleus=0.6):
        return model.generate(input_ids, do_sample=True, max_length=max_length, top_p=nucleus, num_return_sequences=num_seqs)

class SamplingStrategy(Enum):
    beam = SamplingMethods.beam_search
    temp = SamplingMethods.temp_sample
    nucleus = SamplingMethods.nucl_sample
    greedy = 3

@dataclass
class SamplingArgs:
    sampling_strategy: Sequence[SamplingStrategy]
    number_outputs: int
    max_length: int
    num_beams: int
    temperature: float
    top_p: float
    top_k: float

def sample(args: SamplingArgs, model, tokenizer, dataset):
    for i in range(len(dataset)):
        dp = dataset["document"][i]
        input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=1024)
        outputs = args.sampling_strategy(input_ids, model) #
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs= {"document": dp, "gold": dataset["summary"][i], "id": dataset["id"][i], \
            "all_samples": outputs, "num_unique": len(set(outputs))}

