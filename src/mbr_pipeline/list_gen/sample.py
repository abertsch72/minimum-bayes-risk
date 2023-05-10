from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import jsonlines
from typing import Sequence, Text, Union, Callable
from enum import Enum
from scipy.special import softmax, logsumexp
from dataclasses import dataclass
import random
import numpy as np 


class SamplingMethods:
    @staticmethod
    def beam_search(input_ids, model, num_seqs, max_length, num_beams):
        return model.generate(input_ids, do_sample=False, max_length=max_length, num_beams=num_beams, num_return_sequences=num_seqs, output_scores=True, return_dict_in_generate=True)

    @staticmethod
    def temp_sample(input_ids, model, num_seqs, max_length, temperature):
        return model.generate(input_ids, do_sample=True, max_length=max_length, temperature=temperature, num_return_sequences=num_seqs, output_scores=True, return_dict_in_generate=True)

    @staticmethod
    def nucl_sample(input_ids, model, num_seqs, max_length, nucleus):
        return model.generate(input_ids, do_sample=True, max_length=max_length, top_p=nucleus, num_return_sequences=num_seqs, output_scores=True, return_dict_in_generate=True)

    @staticmethod
    #TODO: fix args
    def lattice_mbr():
        from src.mbr_pipeline.list_gen.lattice_mbr import run_lattice_mbr
        return run_lattice_mbr()

    @staticmethod
    #TODO: fix linking
    def lattice_sampling(input_ids, latticedir, tokenizer, num_seqs, k, sample_uniform, max_len, no_repeats, lattice_score_temp):
        from src.mbr_pipeline.list_gen.lattice_sample import run_lattice_sampling
        return run_lattice_sampling(input_ids, latticedir, tokenizer, num_seqs, k, sample_uniform, max_len, no_repeats, lattice_score_temp)

class SamplingStrategy(Enum):
    beam = SamplingMethods.beam_search
    temp = SamplingMethods.temp_sample
    nucleus = SamplingMethods.nucl_sample
    greedy = 3

def listgen_lattice(strategy_fn: Callable, model, tokenizer, dataset, strategy_args):
    model.eval()
    all_hypos = strategy_fn(model, tokenizer, dataset, **strategy_args)

    

def listgen(strategy_fn: Callable, model, tokenizer, dataset, device, num_seqs, max_length, strategy_args):
    all_hypos = []
    model.eval()

    for i in tqdm(range(len(dataset))):
        dp = dataset["input"][i]
        input_ids = tokenizer.encode(dp, return_tensors='pt', truncation=True, max_length=model.config.max_position_embeddings).to(device)
        outputs = strategy_fn(input_ids, model, num_seqs=num_seqs, max_length=max_length, **strategy_args) #

        # get sequence scores by summing generated token scores and applying length penality
        # Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. 
        scores_on_cpu = tuple(score.cpu() for score in outputs.scores)
        transition_scores = model.compute_transition_scores(
            outputs.sequences.cpu(), scores_on_cpu, outputs.beam_indices.cpu(), normalize_logits=False).numpy()

        output_length = input_ids.shape[0] + np.sum(transition_scores < 0, axis=1)
        length_penalty = model.generation_config.length_penalty
        reconstructed_scores = (np.sum(transition_scores, axis=1) / (output_length**length_penalty)).tolist()

        outputs_decoded = tokenizer.batch_decode(outputs.sequences)
        outputs= {"document": dp, "gold": dataset["output"][i], "id": dataset["id"][i], \
            "all_samples": outputs_decoded, "num_unique": len(set(outputs)), "lprobs": reconstructed_scores} #TODO: add lprobs in
        all_hypos.append(outputs)
    
    return all_hypos
