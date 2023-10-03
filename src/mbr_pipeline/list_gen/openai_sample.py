import random
import time
from typing import Callable

import jsonlines
import openai
from tqdm import tqdm

CNNDM_PROMPT = """\
Given a news article, write a short summary of the article in 2-3 sentence. 

Article: {article1}
Q: Summarize the above article briefly in 2-3 sentences.
A: {summary1}

Article: {article2}
Q: Summarize the above article briefly in 2-3 sentences.
A:"""

OPENAI_MODELS = {
    "gpt-4-0613",
    "gpt-4-0314",
    "text-davinci-001",
    "text-search-curie-query-001",
    "davinci",
    "text-davinci-insert-002",
    "gpt-3.5-turbo-16k-0613",
    "text-babbage-001",
    "curie-instruct-beta",
    "text-davinci-003",
    "davinci-similarity",
    "code-davinci-edit-001",
    "text-davinci-insert-001",
    "text-similarity-curie-001",
    "text-embedding-ada-002",
    "ada-code-search-text",
    "text-search-ada-query-001",
    "babbage-search-query",
    "ada-similarity",
    "gpt-3.5-turbo",
    "text-search-ada-doc-001",
    "text-search-babbage-query-001",
    "code-search-ada-code-001",
    "curie-search-document",
    "text-search-davinci-query-001",
    "text-search-curie-doc-001",
    "gpt-3.5-turbo-0301",
    "babbage-search-document",
    "gpt-4",
    "babbage-code-search-text",
    "whisper-1",
    "gpt-3.5-turbo-16k",
    "davinci-instruct-beta",
    "davinci-search-query",
    "text-similarity-babbage-001",
    "text-davinci-002",
    "code-search-babbage-text-001",
    "babbage",
    "text-search-davinci-doc-001",
    "code-search-ada-text-001",
    "ada-search-query",
    "text-similarity-ada-001",
    "ada-code-search-code",
    "ada",
    "text-davinci-edit-001",
    "davinci-search-document",
    "curie-search-query",
    "babbage-similarity",
    "ada-search-document",
    "text-ada-001",
    "text-similarity-davinci-001",
    "curie",
    "curie-similarity",
    "code-davinci-002",
    "gpt-3.5-turbo-0613",
    "babbage-code-search-code",
    "code-search-babbage-code-001",
    "text-search-babbage-doc-001",
    "text-curie-001",
    "code-cushman-001",
}


def make_cnndm_chat_prompt(document, fewshot_examples):
    history = [
        {
            "role": "system",
            "content": "Given a news article, write a short summary of the article in 2-3 sentence.",
        },
    ]
    for article, summary in fewshot_examples:
        history.extend(
            [
                {
                    "role": "user",
                    "content": f"Article: {article}\nQuestion: Summarize the above article briefly in 2-3 sentences.\nAnswer:",
                },
                {"role": "assistant", "content": f"{summary}"},
            ]
        )

    history.append(
        {
            "role": "user",
            "content": f"Article: {document}\nQuestion: Summarize the above article briefly in 2-3 sentences.\nAnswer:",
        }
    )

    return history


def get_chat_completion(max_retries=9999, **kwargs):
    backoff_time = 3
    time.sleep(backoff_time)
    failures = 0
    while True:
        try:
            return openai.ChatCompletion.create(**kwargs)
        except openai.error.OpenAIError:
            import traceback

            traceback.print_exc()
            if failures >= max_retries:
                raise Exception("Reached max number of retries.")
            time.sleep(backoff_time)
            backoff_time *= 1.5
            failures += 1


def openai_listgen(
    dataset,
    num_seqs,
    max_length,
    unique_k,
    strategy_args,
    model: str,
):
    all_hypos = []

    input_lengths = [len(dataset[idx]["input"]) for idx in range(len(dataset))]
    input_lengths.sort()
    max_example_length = input_lengths[int(len(input_lengths) // 4)]

    for i in tqdm(range(len(dataset))):
        dp = dataset["input"][i]

        fewshot_idxs = [
            idx
            for idx in range(len(dataset))
            if idx != i and len(dataset[idx]["input"]) <= max_example_length
        ]
        ex1, ex2 = random.sample(fewshot_idxs, 2)
        # prompt = CNNDM_PROMPT.format(
        #     article1=dataset['input'][ex1],
        #     summary1=dataset['output'][ex1],
        #     # article2=dataset['input'][ex2],
        #     # summary2=dataset['output'][ex2],
        #     article2=dp
        # )
        messages = make_cnndm_chat_prompt(
            dp, [(dataset["input"][ex1], dataset["output"][ex1])]
        )

        outputs = {
            "document": dp,
            "gold": dataset["output"][i],
            "id": dataset["id"][i],
            "hypos": [],
            # "lprobs": [sum(choice['logprobs']['token_logprobs']) for choice in response['choices']],
            "lprobs": None,
        }

        max_chunk = 25

        import math

        num_chunks = int(math.ceil(num_seqs / max_chunk))

        while len(outputs["hypos"]) < num_seqs:
            chunk_size = min(max_chunk, num_seqs - len(outputs["hypos"]))

            response = get_chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_length,
                temperature=strategy_args.get("temp", 0.0),
                top_p=strategy_args.get("top_p", 1.0),
                n=chunk_size,
                # logprobs=5,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            outputs["hypos"].extend(
                [choice["message"]["content"].strip() for choice in response["choices"]]
            )

        from pprint import pprint

        pprint(outputs)

        all_hypos.append(outputs)

        with jsonlines.open("buffer.jsonl", "w") as f:
            f.write_all(all_hypos)

    return all_hypos
