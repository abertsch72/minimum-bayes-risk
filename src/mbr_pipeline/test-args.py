from args import Args, load_args

args = Args(
    pipeline= Args.PipelineArgs(hf_model_name="facebook/bart-base"),
    dataset = Args.DatasetArgs(dataset=Args.DatasetArgs.SupportedDataset.samsum, split=Args.DatasetArgs.DataSplit.val),
    gen = Args.ListGenArgs(method_args=Args.ListGenArgs.BeamSearchArgs(beam_width=10)),
    rerank = Args.ListRerankArgs(),
    eval = Args.EvalArgs(),
)

# example of dumping args to file
import json
print(args.to_dict())
with open("../../configs/config-test.json", 'w') as f:
    json.dump(args.to_dict(), f, indent=4)

# example of loading args from file
#load_args("config-test.json")

