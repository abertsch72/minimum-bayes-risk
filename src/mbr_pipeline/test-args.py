from args import Args, load_args, save_args

args = Args(
    pipeline=Args.PipelineArgs(hf_model_name="facebook/bart-base"),
    dataset=Args.DatasetArgs(
        dataset=Args.DatasetArgs.SupportedDataset.samsum,
        split=Args.DatasetArgs.DataSplit.val,
    ),
    gen=Args.ListGenArgs(
        method_args=Args.ListGenArgs.BeamSearchArgs(num_beams=10), max_length=50
    ),
    rerank=Args.ListRerankArgs(),
    eval=Args.EvalArgs(),
)

# example of dumping args to file
save_args(args, "configs/config-test.json")

# example of loading args from file
load_args("configs/config-test.json")


# make a large set of args
from entrypoint import pipeline

len_by_dataset = {"cnndm": 70, "samsum": 60}
args = Args(
    pipeline=Args.PipelineArgs(hf_model_name="lidiya/bart-base-samsum"),
    dataset=Args.DatasetArgs(dataset=dataset, split=Args.DatasetArgs.DataSplit.val),
    gen=Args.ListGenArgs(method_args=gen_strat, max_length=length, k=50),
    rerank=Args.ListRerankArgs(rerank_metric=rerank, rerank_geo_mean=geo),
    eval=Args.EvalArgs(eval_metrics="rouge1,rouge2,rougeL,chrf,bertscore"),
)
pipeline(args)
"""
count = 0
models_by_dataset = {
    "cnndm": ["facebook/bart-large-cnn", "ainize/bart-base-cnn"],
    "samsum": [
        "lidiya/bart-base-samsum",
        "linydub/bart-large-samsum",
        "Chikashi/t5-small-finetuned-cnndm",
        "navjordj/t5-base-cnndm",
        "navjordj/t5-large-cnndm",
    ],
}
len_by_dataset = {"cnndm": 70, "samsum": 60}
for dataset in ["cnndm", "samsum"]:
    length = len_by_dataset[dataset]
    for model in models_by_dataset[dataset]:
        for gen_strat in [
            Args.ListGenArgs.ModelSamplingArgs(temp=0.1),
            Args.ListGenArgs.ModelSamplingArgs(temp=0.7),
            Args.ListGenArgs.ModelSamplingArgs(top_p=0.3),
            Args.ListGenArgs.ModelSamplingArgs(top_p=0.5),
            Args.ListGenArgs.ModelSamplingArgs(temp=0.5),
        ]:
            for rerank in [
                "rouge1",
                "rouge2",
                "rouge3",
                "rouge4",
                "rouge5",
                "rouge6",
                "bertscore",
                "bartscore",
            ]:
                for geo in [True, False]:
                    if geo and "rouge" not in rerank:
                        continue  # skip this case
                    args = Args(
                        pipeline=Args.PipelineArgs(hf_model_name=model),
                        dataset=Args.DatasetArgs(
                            dataset=dataset, split=Args.DatasetArgs.DataSplit.val
                        ),
                        gen=Args.ListGenArgs(
                            method_args=gen_strat, max_length=length, k=50
                        ),
                        rerank=Args.ListRerankArgs(
                            rerank_metric=rerank, rerank_geo_mean=geo
                        ),
                        eval=Args.EvalArgs(
                            eval_metrics="rouge1,rouge2,rougeL,chrf,bertscore"
                        ),
                    )
                    pipeline(args)
"""
