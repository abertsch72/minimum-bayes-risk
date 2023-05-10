from args import Args, load_args, save_args

args = Args(
    pipeline= Args.PipelineArgs(hf_model_name="facebook/bart-base"),
    dataset = Args.DatasetArgs(dataset=Args.DatasetArgs.SupportedDataset.samsum, split=Args.DatasetArgs.DataSplit.val),
    gen = Args.ListGenArgs(method_args=Args.ListGenArgs.BeamSearchArgs(num_beams=10), max_length=50),
    rerank = Args.ListRerankArgs(),
    eval = Args.EvalArgs(),
)

# example of dumping args to file
save_args(args, "configs/config-test.json")

# example of loading args from file
load_args("configs/config-test.json")

