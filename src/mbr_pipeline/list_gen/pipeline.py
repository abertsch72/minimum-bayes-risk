import os
import sys
from mbr_pipeline.args import get_parser, get_args

assert __name__ == "__main__"  # file should be run as main script

# Toss in all the usual flags for the other files,
# then add --type {top-k list method}
method = sys.argv[sys.argv.index("--type") + 1]

if method == "latticesamp":
    gen_parser = get_parser(latticesamp=True)
    from lattice_sample import run_lattice_sampling as gen_fn
elif method == "latticembr":
    gen_parser = get_parser(latticembr=True)
    from lattice_mbr import run_lattice_mbr as gen_fn
elif method == "beamsearch":
    raise NotImplementedError
elif method == "tempsamp":
    raise NotImplementedError
else:
    raise Exception(f"Unknown decoding method: {method}")

gen_args = get_args(gen_parser)
print("Generation args:", gen_args)

if not os.path.isfile(gen_args.outfile):
    gen_fn(gen_args)
else:
    print(f"Outfile {gen_args.outfile} already exists, skipping to evaluation.")

from rerank import run_rerank

rerank_parser = get_parser(rerank=True)
rerank_args = get_args(rerank_parser)
rerank_configs = getattr(rerank_args, "rerank_configs")
if rerank_configs is not None:
    for rerank_config in rerank_configs:
        for k, v in rerank_config.items():
            setattr(rerank_args, k, v)
        print("Reranking args:", rerank_args)
        run_rerank(rerank_args)
else:
    run_rerank(rerank_args)

from evaluate import run_eval

eval_parser = get_parser()
eval_args = get_args(eval_parser)
print("Evaluation args:", eval_args)
# import cProfile
# cProfile.run('run_eval(eval_args)', "all_stats.txt")
run_eval(eval_args)
