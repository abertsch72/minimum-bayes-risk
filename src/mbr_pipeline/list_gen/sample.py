from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import jsonlines
from typing import Sequence, Text, Union, Callable
from enum import Enum
from dataclasses import dataclass

class SamplingMethods:
    @staticmethod
    def beam_search(input_ids, model, num_seqs, max_length, num_beams):
        return model.generate(input_ids, do_sample=False, max_length=max_length, num_beams=num_beams, num_return_sequences=num_seqs)

    @staticmethod
    def temp_sample(input_ids, model, num_seqs, max_length, temperature):
        return model.generate(input_ids, do_sample=True, max_length=max_length, temperature=temperature, num_return_sequences=num_seqs)

    @staticmethod
    def nucl_sample(input_ids, model, num_seqs, max_length, nucleus):
        return model.generate(input_ids, do_sample=True, max_length=max_length, top_p=nucleus, num_return_sequences=num_seqs)

class SamplingStrategy(Enum):
    beam = SamplingMethods.beam_search
    temp = SamplingMethods.temp_sample
    nucleus = SamplingMethods.nucl_sample
    greedy = 3

def sample(strategy_fn: Callable, model, tokenizer, dataset, strategy_args):
    for i in range(len(dataset)):
        dp = dataset["input"][i]
        input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=1024)
        outputs = strategy_fn(input_ids, model, **strategy_args.__dict__) #
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs= {"document": dp, "gold": dataset["output"][i], "id": dataset["id"][i], \
            "all_samples": outputs, "num_unique": len(set(outputs))}

