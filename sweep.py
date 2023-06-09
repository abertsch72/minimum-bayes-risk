"""
Quick and dirty code to run a sweep over a set of lattice hyperparameters
"""
import itertools
import json
import subprocess

# choices = dict(
#     # mbr args
#     lattice_metric = ['match1', 'match2'],
#     uniform_match = [True],
#     target_evidence_length = [0],
#     evidence_length_deviation = [10, 100],
#     target_candidate_length = [0],
#     candidate_length_deviation = [5, 10, 15],
#     lattice_score_temp = [0.3, 1.0, 2.0],
#     count_aware = [True, False],
#     k_per_node = [5, 25],
#     # rerank args
#     rerank_metric = ["rouge1", "rouge2", "rouge6"],
#     rerank_geo_mean = [True]
# )
choices = dict(k=[50, 5, 10, 25, 75, 100])
# uniform
# length_alpha
#


def make_config_file(
    base_config_file,
    temp_config_file,
    method_args=None,
    rerank_args=None,
    modify_fn=None,
):
    with open(base_config_file, "r") as f:
        config_dict = json.load(f)

    if method_args is not None:
        config_dict["gen"]["method_args"] = method_args
    if rerank_args is not None:
        config_dict["rerank"] = rerank_args
    if modify_fn is not None:
        config_dict = modify_fn(config_dict)

    with open(temp_config_file, "w+") as f:
        json.dump(config_dict, f)


keys = list(choices.keys())
values = [choices[k] for k in keys]

BASE_CONFIG_FILE = "configs/xsum-beamsamp-large.json"
# BASE_CONFIG_FILE = "configs/cnndm-lattmbr.json"
# BASE_CONFIG_FILE = "configs/xsum-lattmbr.json"
TEMP_CONFIG_FILE = "configs/tempfile-xsum-beamsamp-large.json"
# TEMP_CONFIG_FILE = "configs/cnndm_temp_config.json"
# TEMP_CONFIG_FILE = "configs/xsum_temp_config.json"


def main():
    print("Base config file:", BASE_CONFIG_FILE)
    print("Temp config file:", TEMP_CONFIG_FILE)

    for config in itertools.product(*values):
        # if config[-2] == 1 and config[-1] != 1:
        #    continue
        # method_args = {k: v for k, v in zip(keys[:-2], config[:-2])}
        # rerank_args = {k: v for k, v in zip(keys[-2:], config[-2:])}
        # make_config_file(BASE_CONFIG_FILE, TEMP_CONFIG_FILE, method_args, rerank_args)
        k = config[0]

        def edit_fn(k):
            def fn(config_dict):
                config_dict["gen"]["method_args"]["num_beams"] = k
                config_dict["gen"]["k"] = k
                return config_dict

            return fn

        make_config_file(BASE_CONFIG_FILE, TEMP_CONFIG_FILE, modify_fn=edit_fn(k))

        subprocess.check_call(["bash", "run_entrypoint.sh", TEMP_CONFIG_FILE])


if __name__ == "__main__":
    main()
