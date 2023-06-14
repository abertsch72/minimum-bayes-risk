import warnings
from typing import Any, Callable, Generator, Tuple

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedModel

from src.mbr_pipeline.list_gen.lattice import Lattice


class SamplingMethods:
    @staticmethod
    def beam_search(
        input_ids,
        model,
        num_seqs,
        max_length,
        num_beams,
        num_beam_groups=1,
        diversity_penalty=0.0,
        do_sample=False,
    ):
        return model.generate(
            input_ids,
            do_sample=do_sample,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_seqs,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            output_scores=True,
            return_dict_in_generate=True,
        )

    @staticmethod
    def model_sample(input_ids, model, num_seqs, max_length, temp, top_p, sample_unique_seqs=False):
        return model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            top_p=top_p,
            temperature=temp,
            num_beams=1,
            num_return_sequences=num_seqs,
            num_beam_groups=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

    @staticmethod
    # TODO: fix args
    def lattice_mbr():
        from src.mbr_pipeline.list_gen.lattice_mbr import run_lattice_mbr

        return run_lattice_mbr()


def listgen(
    strategy_fn: Callable,
    tokenizer,
    dataset,
    device,
    num_seqs,
    max_length,
    unique_k,
    strategy_args,
    model: PreTrainedModel = None,
    lattices: Generator[Tuple[Lattice, Any], None, None] = None,
):
    all_hypos = []

    # TODO: should this be a single argument with a union of types instead? probably
    try:
        assert model or lattices
    except AssertionError as e:
        raise ValueError(
            "Must pass either model or lattices to list generation method!"
        )

    if lattices:
        for lattice, output in lattices:
            #  k, output, uniform=False, max_len=float('inf'), no_repeats=False, temp=1.0):
            all_hypos.append(
                strategy_fn(
                    lattice=lattice,
                    tokenizer=tokenizer,
                    num_seqs=num_seqs,
                    output=output,
                    max_length=max_length,
                    **strategy_args
                )
            )
    else:
        assert model
        model.eval()
        for i in tqdm(range(len(dataset))):
            dp = dataset["input"][i]
            max_source_len = 0
            try:
                max_source_len = model.config.max_position_embeddings
            except:
                try:
                    max_source_len = model.config.n_positions
                except: 
                    max_source_len = model.config.d_model

            input_ids = tokenizer.encode(
                dp,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_len,
            ).to(device)

            outputs_tokens = {}

            def continue_list_gen(outputs={}):
                model_outputs = strategy_fn(
                    input_ids,
                    model,
                    num_seqs=num_seqs,
                    max_length=max_length,
                    **strategy_args
                )  #

                # get sequence scores by summing generated token scores and applying length penality
                # Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`.
                scores_on_cpu = tuple(score.cpu() for score in model_outputs.scores)
                try:
                    transition_scores = model.compute_transition_scores(
                        model_outputs.sequences.cpu(),
                        scores_on_cpu,
                        model_outputs.beam_indices.cpu(),
                        normalize_logits=False,
                    ).numpy()
                except:
                    transition_scores = model.compute_transition_scores(
                        model_outputs.sequences.cpu(), scores_on_cpu, normalize_logits=False
                    ).numpy()

                output_length = input_ids.shape[0] + np.sum(transition_scores < 0, axis=1)
                length_penalty = model.generation_config.length_penalty
                reconstructed_scores = (
                    np.sum(transition_scores, axis=1) / (output_length**length_penalty)
                ).tolist()

                # todo: hash outputs.sequences
                outputs_decoded = tokenizer.batch_decode(
                    model_outputs.sequences, skip_special_tokens=True
                )
                hashes = [hash(tuple(seq)) for seq in model_outputs.sequences]
                saved_out = zip(outputs_decoded, reconstructed_scores)
                these_outputs = list(zip(hashes, saved_out))
                outputs_tokens.update(these_outputs)

                del model_outputs
                
                if unique_k:
                    outputs["num_samples"] = outputs.get("num_samples", 0) + num_seqs
                    #print(outputs_tokens)
                    print(f"The total number of unique hypotheses is {len(outputs_tokens.keys())} out of {outputs['num_samples']} total sampled!")
                    outputs_decoded = [output[0] for output in list(outputs_tokens.values())]
                    reconstructed_scores = [output[1] for output in list(outputs_tokens.values())]

                outputs["hypos"] = outputs_decoded
                outputs["lprobs"] = reconstructed_scores

                return outputs
            
            outputs = {
                "document": dp,
                "gold": dataset["output"][i],
                "id": dataset["id"][i],
                "hypos": [],
                "lprobs": [],
                }  


            while len(outputs["hypos"]) < num_seqs:
                outputs = continue_list_gen(outputs)

            outputs["num_unique"] = len(set(outputs["hypos"]))
            all_hypos.append(outputs)

    return all_hypos
