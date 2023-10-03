import random
import sys

import numpy as np
import torch

is_run_pipeline = any("run_pipeline.py" in arg for arg in sys.argv)

from src.recom_search.model.setup import (
    process_arg,
    render_address,
    setup_logger,
    setup_model,
)

args, grouped_args = None, None
tokenizer = None
model = None
dataset = None
dec_prefix = None
logger = None

if is_run_pipeline:  # only setup in if we're running the main pipeline script
    args, grouped_args = process_arg()

    # fix the seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    dict_io = render_address(root=args.path_output)
    setup_logger(name=f"{args.task}_{args.model}_{args.dataset}")
    print(args)
    print("Running run_pipeline.py")
    tokenizer, model, dataset, dec_prefix = setup_model(
        task=args.task,
        dataset=args.dataset,
        model_name=args.hf_model_name,
        device_name=args.device,
    )
