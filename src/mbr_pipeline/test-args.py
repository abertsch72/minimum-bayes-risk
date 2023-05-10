from args import Args
from dataclasses_json import dataclass_json
args = Args(
    pipeline= Args.PipelineArgs(),
    gen = Args.ListGenArgs(method_args=Args.ListGenArgs.BeamSearchArgs(beam_width=10)),
    rerank = Args.ListRerankArgs(),
    eval = Args.EvalArgs(),
)

# example of dumping args to file
import json
print(args.to_dict())
with open("config-test.json", 'w') as f:
    json.dump(args.to_dict(), f, indent=4)

# example of loading args from file
def load_args(config_file):
    # json to args with validation
    import json
    config = json.dumps(json.load(open(config_file)))
    args = Args.schema().loads(config)
    print(args)
load_args("config-test.json")