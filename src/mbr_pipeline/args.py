from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import marshmallow as mm
from dataclasses_json import config, dataclass_json


@dataclass_json
@dataclass
class Args:
    @dataclass_json
    @dataclass
    class PipelineArgs:
        """
        General arguments
        """

        hf_model_name: str = field()
        hf_tokenizer_name: Optional[str] = field(default=None)
        wandb: Optional[bool] = field(default=True)
        wandb_group: Optional[str] = field(default=None)
        run_name: Optional[str] = field(default=None)
        seed: Optional[int] = field(default=101)
        no_tqdm: Optional[bool] = field(default=False)
        no_gpu: Optional[bool] = field(default=False)
        lattice_dir: Optional[str] = field(
            default=None, metadata={"help": "directory where lattices are stored"}
        )

    @dataclass_json
    @dataclass
    class DatasetArgs:
        class SupportedDataset(Enum):
            samsum = ["samsum"]
            cnndm = ["ccdv/cnn_dailymail", "3.0.0"]
            xsum = ["xsum"]
            flores = ["facebook/flores", "nob_Latn-eng_Latn"]
            flores_isl = ["facebook/flores", "isl_Latn-eng_Latn"]

            def __str__(e):
                return e.name

        class DataSplit(str, Enum):
            val = "validation"
            test = "test"
            train = "train"
            # for flores:
            dev = "dev"
            devtest = "devtest"

            def __str__(e):
                return e.name

        dataset: Union[str, SupportedDataset] = field(
            metadata=config(
                encoder=lambda d: d.name
                if isinstance(d, Args.DatasetArgs.SupportedDataset)
                else d,
                decoder=lambda d: d
                if isinstance(d, Args.DatasetArgs.SupportedDataset)
                else Args.DatasetArgs.SupportedDataset[d],
            )
        )
        split: Union[str, DataSplit] = field(
            metadata=config(
                encoder=lambda d: d.name
                if isinstance(d, Args.DatasetArgs.DataSplit)
                else d,
                decoder=lambda d: d
                if isinstance(d, Args.DatasetArgs.DataSplit)
                else Args.DatasetArgs.DataSplit[d],
            )
        )
        start_index: int = 0
        end_index: int = -1
        shuffle: Optional[bool] = field(default=False)

    @dataclass_json
    @dataclass
    class ListGenArgs:
        """
        General arguments for top-k list generation
        """

        @dataclass_json
        @dataclass
        class BeamSearchArgs:
            """
            Arguments pertaining to beam search top-k list generation
            """

            num_beams: Optional[int] = field(default=50)
            num_beam_groups: Optional[int] = field(default=1)
            diversity_penalty: Optional[float] = field(default=0.0)
            stochastic: Optional[bool] = field(default=False)
            memoryless: Optional[bool] = field(default=False)
            beam_temp: Optional[float] = field(default=1.0)
            beam_top_p: Optional[float] = field(default=1.0)

        @dataclass_json
        @dataclass
        class ModelSamplingArgs:
            """
            Arguments for all model-based sampling baselines
            """

            temp: Optional[float] = field(default=1.0)
            top_p: Optional[float] = field(default=1.0)

        @dataclass_json
        @dataclass
        class LatticeMBRArgs:
            """
            Arguments pertaining to MBR over lattice for top-k list generation
            """

            lattice_metric: str = field(
                default="rouge1",
                metadata={
                    "help": "options are 'rouge1', 'rouge2', 'match1', 'match2', 'exact_rouge1'"
                },
            )
            uniform_length: Optional[bool] = field(
                default=False,
                metadata={
                    "help": "use uniform scoring for mean length calculation (i.e. count-based)"
                },
            )
            length_temp: Optional[float] = field(
                default=1.0,
                metadata={
                    "help": "temperature when calculating non-uniform mean length"
                },
            )
            uniform_match: Optional[bool] = field(
                default=False,
                metadata={"help": "use uniform scoring for expected match calculation"},
            )
            match_temp: Optional[float] = field(
                default=1.0,
                metadata={
                    "help": "temperature when calculating non-uniform expected match"
                },
            )
            target_evidence_length: Optional[int] = field(
                default=-1,
                metadata={
                    "help": "Target length for evidence set. -1 denotes no target length, "
                    + "0 denotes target length = mean length, and anything >0 will "
                    + "be used directly as target length"
                },
            )

            # min/max deviation of evidence set length from target length
            evidence_length_deviation: Optional[int] = field(
                default=float("inf"),
                metadata=config(
                    mm_field=mm.fields.Float(allow_nan=True),
                ),
            )
            target_candidate_length: Optional[int] = field(
                default=-1,
                metadata={
                    "help": "Target length for candidate set. -1 denotes no target length, "
                    + "0 denotes target length = mean length, and anything >0 will "
                    + "be used directly as target length"
                },
            )

            # min/max deviation of candidate set length from target length
            candidate_length_deviation: Optional[int] = field(
                default=float("inf"),
                metadata=config(mm_field=mm.fields.Float(allow_nan=True)),
            )

            mean_override: Optional[int] = field(
                default=-1,
                metadata={
                    "help": "value to override the mean with (-1 means use the actual mean)"
                },
            )
            lattice_score_temp: Optional[float] = field(
                default=1.0,
                metadata={"help": "temperature on lattice edge probabilities"},
            )
            count_aware: Optional[bool] = field(
                default=False,
                metadata={"help": "use count awareness in rouge(-1) approximation"},
            )
            k_per_node: Optional[int] = field(
                default=0,
                metadata={
                    "help": "number of candidates to track per node in the lattice"
                },
            )

        @dataclass_json
        @dataclass
        class LatticeSamplingArgs:
            """
            Arguments pertaining to topk-list generation via direct sampling from the lattice
            """

            sample_uniform: Optional[bool] = field(
                default=False,
                metadata={"help": "whether to ignore edge probs when sampling"},
            )
            no_repeats: Optional[bool] = field(
                default=False,
                metadata={"help": "Whether to disregard repeats when sampling"},
            )
            lattice_score_temp: Optional[float] = field(
                default=1.0,
                metadata={"help": "temperature on lattice edge probabilities"},
            )

        subclasses = [
            BeamSearchArgs,
            ModelSamplingArgs,
            LatticeMBRArgs,
            LatticeSamplingArgs,
        ]

        @classmethod
        def decode(cls, d: dict):
            if isinstance(d, tuple(cls.subclasses)):
                return d

            for subdataclass in cls.subclasses:
                if len(
                    [k for k in d.keys() if k in list(subdataclass.__dict__.keys())]
                ) == len(d.keys()):
                    return subdataclass.from_dict(d)
            raise AttributeError(f"Could not parse ListGenArgs method_args: {str(d)}")

        method_args: Union[
            BeamSearchArgs, ModelSamplingArgs, LatticeMBRArgs, LatticeSamplingArgs, dict
        ] = field(metadata=config(decoder=lambda d: Args.ListGenArgs.decode(d)))
        max_length: int
        k: Optional[int] = field(
            default=50,
            metadata={"help": "Length of top-k list"},
        )
        unique_k: Optional[bool] = field(
            default=False,
            metadata={"help": "whether to get <k> unique outputs"}
        )
        outfile: Optional[str] = field(default=None)

    @dataclass_json
    @dataclass
    class ListRerankArgs:
        """
        Arguments pertaining to topk-list reranking
        """

        rerank_metric: Optional[
            str
            # Literal['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rouge5', 'rouge6',
            #         'bertscore', 'bartscore']
        ] = field(default=None)

        rerank_temp: Optional[float] = field(
            metadata=config(mm_field=mm.fields.Float(allow_nan=True)),
            default=float("inf"),
        )

        rerank_geo_mean: Optional[bool] = field(
            default=False,
            metadata={"help": "use geo mean of rouges up to specified order"},
        )

        rank_by_freq: Optional[bool] = field(
            default=False,
            metadata={"help": "use frequency of hypotheses instead of lprobs to rank"},
        )

        evidence_set_file: Optional[str] = field(
            default=None,
            metadata={
                "help": "file containing hypotheses for evidence set; if null, hypothesis set used as evidence set"
            },
        )

    @dataclass_json
    @dataclass
    class EvalArgs:
        """
        Arguments pertaining to topk-list evaluation
        """

        eval_metrics: str = field(
            default=None, #"rouge1,rouge2,rougeL,chrf",
            metadata={"help": "comma-separated list of metrics to evaluate hypos on"},
        )
        outfile: Optional[str] = field(default=None)

    pipeline: PipelineArgs
    dataset: DatasetArgs
    gen: Optional[ListGenArgs]
    rerank: Optional[ListRerankArgs]
    eval: Optional[EvalArgs] = field(default=None)


def load_args(config_file: str) -> Args:
    # json to args with validation
    import json

    config = json.dumps(json.load(open(config_file)))
    args = Args.schema().loads(config)
    return args


def save_args(args: Args, config_file: str) -> None:
    # args to json
    import json

    with open(config_file, "w") as f:
        json.dump(args.to_dict(), f, indent=4)
