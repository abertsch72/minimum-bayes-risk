import warnings
from typing import Any, Callable, Generator, Tuple

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
from transformers import BartForConditionalGeneration, PreTrainedModel

from mbr_pipeline.list_gen.lattice import Lattice

NUM_GEN_AT_ONCE = 100


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
        stochastic=False,
        memoryless=False,
        beam_temp=1.0,
        beam_top_p=1.0,
    ):
        return model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_seqs,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            output_scores=True,
            return_dict_in_generate=True,
        )

    @staticmethod
    def stochastic_beam_search(
        input_ids,
        model: BartForConditionalGeneration,
        num_seqs,
        max_length,
        num_beams,
        num_beam_groups=1,
        diversity_penalty=0.0,
        stochastic=True,
        memoryless=False,
        beam_temp=1.0,
        beam_top_p=1.0,
    ):
        return model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_seqs,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            top_p=beam_top_p,
            temperature=beam_temp,
            output_scores=True,
            return_dict_in_generate=True,
        )

    @staticmethod
    @torch.no_grad()
    def model_sample(
        input_ids, model, num_seqs, max_length, temp, top_p, sample_unique_seqs=False
    ):
        if num_seqs <= NUM_GEN_AT_ONCE:
            return model.generate(
                input_ids,
                do_sample=True,
                max_length=max_length,
                num_beams=1,
                num_return_sequences=num_seqs,
                num_beam_groups=1,
                top_p=top_p,
                temperature=temp,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            num_generated = 0
            outputs = []
            max_len = -1
            while num_generated < num_seqs:
                curr_num_seqs = min(NUM_GEN_AT_ONCE, num_seqs - num_generated)
                partial_output = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=max_length,
                    num_beams=1,
                    num_return_sequences=curr_num_seqs,
                    num_beam_groups=1,
                    top_p=top_p,
                    temperature=temp,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                # for key in partial_output.keys():
                #     t = partial_output[key]
                #     if isinstance(t, torch.Tensor):
                #         partial_output[key] = t.cpu()
                outputs.append(partial_output)
                num_generated += curr_num_seqs
                max_len = max(max_len, len(partial_output.scores))
            full_sequences = torch.full(
                (num_seqs, max_len + 1), model.config.pad_token_id
            )
            start_idx = 0
            for out in outputs:
                curr_sequences = out.sequences
                curr_bs, curr_len = curr_sequences.shape
                full_sequences[
                    start_idx : start_idx + curr_bs, :curr_len
                ] = curr_sequences
                start_idx += curr_bs
            # beam_indices = torch.cat([out.beam_indices for out in outputs], dim=0)
            scores = []
            for i in range(max_len):
                scores.append(
                    torch.cat(
                        [
                            out.scores[i]
                            if i < len(out.scores)
                            else torch.full(
                                out.scores[0].shape,
                                -float("inf"),
                                device=out.scores[0].device,
                            )
                            for out in outputs
                        ],
                        dim=0,
                    )
                )
            output = partial_output[0]
            output.sequences = full_sequences
            # output.beam_indices = beam_indices
            output.scores = tuple(scores)

            return output


def listgen(
    strategy_fn: Callable,
    tokenizer,
    dataset,
    device,
    num_seqs,
    max_length,
    strategy_args,
    model: PreTrainedModel = None,
    lattices: Generator[Tuple[Lattice, Any], None, None] = None,
    sample_unique_list=False,
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
                    **strategy_args,
                )
            )
    else:
        assert model
        model.eval()
        for i in tqdm(range(len(dataset))):
            dp = dataset["input"][i]
            max_source_len = 0

            max_source_len = getattr(model.config, "max_position_embeddings", None)
            if max_source_len is None:
                max_source_len = getattr(model.config, "n_positions", None)
            if max_source_len is None:
                max_source_len = getattr(model.config, "max_length", None)
            assert max_source_len is not None

            input_ids = tokenizer.encode(
                dp,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_len,
            ).to(device)

            outputs_tokens = {}

            def continue_list_gen(prev_unique={}):
                outputs = strategy_fn(
                    input_ids,
                    model,
                    num_seqs=num_seqs,
                    max_length=max_length,
                    **strategy_args,
                )  #

                # get sequence scores by summing generated token scores and applying length penality
                # Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`.
                scores_on_cpu = tuple(score.cpu() for score in outputs.scores)
                if hasattr(outputs, "beam_indices"):  # beam search
                    transition_scores = model.compute_transition_scores(
                        outputs.sequences.cpu(),
                        scores_on_cpu,
                        outputs.beam_indices.cpu(),
                        normalize_logits=False,
                    ).numpy()

                    # this way of recomputing scores only works for beam search
                    output_length = input_ids.shape[0] + np.sum(
                        transition_scores < 0, axis=1
                    )
                    length_penalty = model.generation_config.length_penalty
                    reconstructed_scores = (
                        np.sum(transition_scores, axis=1)
                        / (output_length**length_penalty)
                    ).tolist()
                else:  # not beam search
                    transition_scores = model.compute_transition_scores(
                        outputs.sequences.cpu(), scores_on_cpu, normalize_logits=True
                    ).numpy()

                    transition_scores[np.isinf(transition_scores)] = 0.0
                    output_length = input_ids.shape[0] + np.sum(
                        1 - np.isinf(transition_scores), axis=1
                    )
                    length_penalty = model.generation_config.length_penalty
                    reconstructed_scores = (
                        np.sum(transition_scores, axis=1)
                        / (output_length**length_penalty)
                    ).tolist()

                # todo: hash outputs.sequences
                outputs_decoded = tokenizer.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                )

                if sample_unique_list:
                    # TODO: track total number of samples
                    # TODO: dedup at token level instead
                    scores_for_unq = dict(zip(outputs_decoded, reconstructed_scores))
                    previous_unq = dict(
                        zip(outputs["hypos"], outputs["reconstructed_scores"])
                    )
                    scores_for_unq.update(
                        previous_unq
                    )  # overrides any duplicates here with the older version
                    # debug
                    print(
                        f"The total number of new hypotheses is {len(scores_for_unq) - len(previous_unq)} for a total of {len(scores_for_unq)} generated!"
                    )
                    outputs_decoded = scores_for_unq.keys()
                    reconstructed_scores = [
                        scores_for_unq[output] for output in outputs_decoded
                    ]

                outputs["hypos"] = outputs["hypos"].extend(outputs_decoded)
                outputs["lprobs"] = outputs["lprobs"].extend(reconstructed_scores)

                return outputs

            while len(outputs["hypos"]) < num_seqs:
                outputs = continue_list_gen(outputs)

            outputs["num_unique"] = len(set(outputs["hypos"]))
            all_hypos.append(outputs)

    return all_hypos
