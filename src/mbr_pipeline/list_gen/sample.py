from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import jsonlines
from typing import Sequence, Text, Union, Callable
from enum import Enum
from dataclasses import dataclass
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
    def run_lattice_sampling(input_ids, model, tokenizer, num_seqs, k, sample_uniform, max_len, no_repeats, lattice_score_temp):
        
        all_hypos = []
        for lattice, output in Lattice.load_lattices(args.lattice_dir, no_tqdm=args.no_tqdm):
            topk_hypos = sample_k(
                lattice, tokenizer, 
                k=k, 
                uniform=sample_uniform,
                max_len=max_len,
                no_repeats=no_repeats,
                temp=lattice_score_temp
            )
            hypos = [v[0] for v in topk_hypos]
            lprobs = [v[1] for v in topk_hypos]
            # sort from greatest to least
            sorted_order = np.argsort(lprobs)[::-1]
            hypos = [hypos[i] for i in sorted_order]
            lprobs = [lprobs[i] for i in sorted_order]
            all_hypos.append({
                "hypos": hypos,
                "lprobs": lprobs, # LOG probabilities
                "gold": output.reference,
                "doc_id": output.doc_id
            })

        with jsonlines.open(args.outfile, "w") as f:
            f.write_all(all_hypos)

        if args.wandb:
            wandb.log({"topk": all_hypos})


class SamplingStrategy(Enum):
    beam = SamplingMethods.beam_search
    temp = SamplingMethods.temp_sample
    nucleus = SamplingMethods.nucl_sample
    greedy = 3

def listgen_lattice(strategy_fn: Callable, model, tokenizer, dataset, strategy_args):
    pass

def listgen(strategy_fn: Callable, model, tokenizer, dataset, device, outfile, num_seqs, max_length, strategy_args):
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

    with jsonlines.open(outfile, 'w') as f:
        f.write_all(all_hypos)
    