import os
import random
import time
from typing import Callable

import jsonlines
import openai
import requests
from tqdm import tqdm

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", None)

CNNDM_PROMPT = """\
Given a news article, write a short summary of the article in 2-3 sentence. 

Article: {article1}
Q: Summarize the above article briefly in 2-3 sentences.
A: {summary1}

Article: {article2}
Q: Summarize the above article briefly in 2-3 sentences.
A:"""

RO_EN_PROMPT = """\
Translate the following sentences from Romanian to English.

Q: Fostul șef al cabinetului prezidențial brazilian este adus în fața instanței
A: Brazil's Former Presidential Chief-of-Staff to Stand Trial

Q: Marți, un judecător federal a acceptat acuzațiile aduse împotriva fostului șef al cabinetului prezidențial brazilian pentru presupusa implicare a acestuia într-o schemă masivă de corupție privind compania petrolieră de stat Petrobras.
A: A federal judge on Tuesday accepted the charges filed against Brazil's former presidential chief of staff for his alleged involvement in a massive corruption scheme at state-owned oil company Petrobras.

Q: Biroul procurorului federal a declarat că Jose Dirceu va fi trimis în judecată pentru acuzațiile de corupție, înșelătorie și spălare de bani aduse în această lună.
A: The federal prosecutor's office said Jose Dirceu will face trial on the corruption, racketeering and money laundering charges filed earlier this month.

Q: Alte paisprezece persoane vor fi judecate, printre acestea numărându-se Joao Vaccari Neto, fostul trezorier al Partidului Muncitorilor, aflat la putere în Brazilia, și Renato de Souza Duque, fostul președinte al serviciilor pentru întreprinderi ale Petrobras.
A: Fourteen other people will also be tried, including Joao Vaccari Neto, the former treasurer of Brazil's governing Workers' Party and Renato de Souza Duque, Petrobras' former head of corporate services.

Q: Dirceu este cel mai vechi membru al Partidului Muncitorilor aflat la guvernare luat în custodie pentru legăturile cu această schemă.
A: Dirceu is the most senior member of the ruling Workers' Party to be taken into custody in connection with the scheme.

Q: {src_sentence}
A:"""

ro_en_fewshot_examples = [
    (
        "Fostul șef al cabinetului prezidențial brazilian este adus în fața instanței",
        "Brazil's Former Presidential Chief-of-Staff to Stand Trial",
    ),
    (
        "Marți, un judecător federal a acceptat acuzațiile aduse împotriva fostului șef al cabinetului prezidențial brazilian pentru presupusa implicare a acestuia într-o schemă masivă de corupție privind compania petrolieră de stat Petrobras.",
        "A federal judge on Tuesday accepted the charges filed against Brazil's former presidential chief of staff for his alleged involvement in a massive corruption scheme at state-owned oil company Petrobras.",
    ),
    # (
    #     "Biroul procurorului federal a declarat că Jose Dirceu va fi trimis în judecată pentru acuzațiile de corupție, înșelătorie și spălare de bani aduse în această lună.",
    #     "The federal prosecutor's office said Jose Dirceu will face trial on the corruption, racketeering and money laundering charges filed earlier this month."
    # )
    (
        "Alte paisprezece persoane vor fi judecate, printre acestea numărându-se Joao Vaccari Neto, fostul trezorier al Partidului Muncitorilor, aflat la putere în Brazilia, și Renato de Souza Duque, fostul președinte al serviciilor pentru întreprinderi ale Petrobras.",
        "Fourteen other people will also be tried, including Joao Vaccari Neto, the former treasurer of Brazil's governing Workers' Party and Renato de Souza Duque, Petrobras' former head of corporate services.",
    )
    # (
    #     "Dirceu este cel mai vechi membru al Partidului Muncitorilor aflat la guvernare luat în custodie pentru legăturile cu această schemă.",
    #     "Dirceu is the most senior member of the ruling Workers' Party to be taken into custody in connection with the scheme."
    # )
]

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


def make_translation_chat_prompt(src_sentence, fewshot_examples, src_lang, tgt_lang):
    history = [
        {
            "role": "system",
            "content": f"Translate the following sentences from {src_lang} to {tgt_lang}.",
        },
    ]
    for src_text, tgt_text in fewshot_examples:
        history.extend(
            [
                {
                    "role": "user",
                    "content": f"{src_lang}: {src_text}\n{tgt_lang}:",
                },
                {"role": "assistant", "content": f"{tgt_text}"},
            ]
        )

    history.append(
        {
            "role": "user",
            "content": f"{src_lang}: {src_sentence}\n{tgt_lang}:",
        }
    )

    return history


def get_chat_completion(max_retries=9999, **kwargs):
    backoff_time = 3
    time.sleep(4)
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


def get_completion(max_retries=9999, **kwargs):
    backoff_time = 3
    failures = 0
    while True:
        try:
            return openai.Completion.create(**kwargs)
        except openai.error.OpenAIError:
            import traceback

            traceback.print_exc()
            if failures >= max_retries:
                raise Exception("Reached max number of retries.")
            time.sleep(backoff_time)
            backoff_time *= 1.5
            failures += 1


endpoint = "https://api.together.xyz/inference"


def get_completion_together(**kwargs):
    return requests.post(
        endpoint,
        json=kwargs,
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            # "User-Agent": "<YOUR_APP_NAME>"
        },
    )


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

        # fewshot_idxs = [
        #     idx
        #     for idx in range(len(dataset))
        #     if idx != i and len(dataset[idx]["input"]) <= max_example_length
        # ]
        # ex1, ex2 = random.sample(fewshot_idxs, 2)
        # prompt = CNNDM_PROMPT.format(
        #     article1=dataset['input'][ex1],
        #     summary1=dataset['output'][ex1],
        #     # article2=dataset['input'][ex2],
        #     # summary2=dataset['output'][ex2],
        #     article2=dp
        # )
        # prompt = RO_EN_PROMPT.format(src_sentence=dp)
        # messages = make_cnndm_chat_prompt(
        #     dp, [(dataset["input"][ex1], dataset["output"][ex1])]
        # )
        messages = make_translation_chat_prompt(
            dp, ro_en_fewshot_examples, "Romanian", "English"
        )

        outputs = {
            "document": dp,
            "gold": dataset["output"][i],
            "id": dataset["id"][i],
            "hypos": [],
            # "lprobs": [sum(choice['logprobs']['token_logprobs']) for choice in response['choices']],
            "lprobs": [],
        }

        max_chunk = 50

        while len(outputs["hypos"]) < num_seqs:
            chunk_size = min(max_chunk, num_seqs - len(outputs["hypos"]))

            response = get_chat_completion(
                # response = get_completion(
                model=model,
                messages=messages,
                # prompt=prompt,
                max_tokens=max_length,
                temperature=strategy_args.get("temp", 0.0),
                top_p=strategy_args.get("top_p", 1.0),
                n=chunk_size,
                # logprobs=5,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            # outputs["hypos"].extend(
            #     [choice["text"].strip() for choice in response["choices"]]
            # )
            outputs["hypos"].extend(
                [choice["message"]["content"].strip() for choice in response["choices"]]
            )
            # outputs['lprobs'].extend(
            #     [sum(choice['logprobs']['token_logprobs']) for choice in response['choices']]
            # )

        from pprint import pprint

        pprint(outputs)

        all_hypos.append(outputs)

        with jsonlines.open("ro-en-buffer.jsonl", "w") as f:
            f.write_all(all_hypos)

    return all_hypos
