import json
from transformers import HfArgumentParser
from typing import Optional, Literal, Union, get_type_hints
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
import marshmallow as mm

@dataclass_json
@dataclass
class Args:
    @dataclass_json
    @dataclass
    class PipelineArgs:
        """
        General arguments
        """
        hf_model_name: Optional[str] = field(default='facebook/bart-large-xsum')
        wandb: Optional[bool] = field(default=False)
        wandb_group: Optional[str] = field(default=None)
        run_name: Optional[str] = field(default=None)
        seed: Optional[int] = field(default=None)
        no_tqdm: Optional[bool] = field(default=False)
        config: Optional[str] = field(
            default=None,
            metadata={"help": "File containing command line flags in json format."}
        )

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
            beam_width: Optional[int] = field(default=50)

        @dataclass_json
        @dataclass
        class ModelSamplingArgs:
            """
            Arguments for all model-based sampling baselines
            """
            
        @dataclass_json
        @dataclass
        class LatticeMBRArgs:
            """
            Arguments pertaining to MBR over lattice for top-k list generation
            """
            lattice_dir: str = field(
                default='',
                metadata={'help': 'directory where lattices are stored'}
            )
            lattice_metric: str = field(
                default="rouge1",
                metadata={'help': "options are 'rouge1', 'rouge2', 'match1', 'match2', 'exact_rouge1'"}
            )
            uniform_length: Optional[bool] = field(
                default=False,
                metadata={'help': 'use uniform scoring for mean length calculation (i.e. count-based)'}
            )
            length_temp: Optional[float] = field(
                default=1.0,
                metadata={"help": "temperature when calculating non-uniform mean length"}
            )
            uniform_match: Optional[bool] = field(
                default=False,
                metadata={'help': 'use uniform scoring for expected match calculation'}
            )
            match_temp: Optional[float] = field(
                default=1.0,
                metadata={"help": "temperature when calculating non-uniform expected match"}
            )
            target_evidence_length: Optional[int] = field(
                default=-1,
                metadata={"help": "Target length for evidence set. -1 denotes no target length, " +
                                "0 denotes target length = mean length, and anything >0 will " +
                                "be used directly as target length"}
            )
            
            # min/max deviation of evidence set length from target length
            evidence_length_deviation: Optional[int] = field(
                default=float('inf'),
                metadata = config(
                    mm_field = mm.fields.Float(allow_nan=True),
                )
            )
            target_candidate_length: Optional[int] = field(
                default=-1,
                metadata={"help": "Target length for candidate set. -1 denotes no target length, " +
                                "0 denotes target length = mean length, and anything >0 will " +
                                "be used directly as target length"}
            )

            # min/max deviation of candidate set length from target length
            candidate_length_deviation: Optional[int] = field(
                default=float('inf'),
                metadata=config(
                    mm_field = mm.fields.Float(allow_nan=True)
                )
            )

            mean_override: Optional[int] = field(
                default=-1,
                metadata={'help': 'value to override the mean with (-1 means use the actual mean)'},
            )
            lattice_score_temp: Optional[float] = field(
                default=1.0,
                metadata={'help': 'temperature on lattice edge probabilities'},
            )
            count_aware: Optional[bool] = field(
                default=False,
                metadata={"help": "use count awareness in rouge(-1) approximation"}
            )
            k_per_node: Optional[int] = field(
                default=0,
                metadata={"help": "number of candidates to track per node in the lattice"}
            )

        @dataclass_json
        @dataclass
        class LatticeSamplingArgs:
            """
            Arguments pertaining to topk-list generation via direct sampling from the lattice
            """
            lattice_dir: str = field(
                default='',
                metadata={'help': 'directory where lattices are stored'}
            )
            sample_uniform: Optional[bool] = field(
                default=False,
                metadata={"help": "whether to ignore edge probs when sampling"}
            )
            no_repeats: Optional[bool] = field(
                default=False,
                metadata={'help': 'Whether to disregard repeats when sampling'}
            )
            max_len: Optional[int] = field(
                default=1_000_000,
                metadata={'help': 'Max len of a sample from the lattice'}
            )
            lattice_score_temp: Optional[float] = field(
                default=1.0,
                metadata={'help': 'temperature on lattice edge probabilities'},
            )

        subclasses = [BeamSearchArgs, ModelSamplingArgs, LatticeMBRArgs, LatticeSamplingArgs]
        @classmethod
        def decode(cls, d: dict):
            if isinstance(d, tuple(cls.subclasses)):
                return d

            for subdataclass in cls.subclasses:
                if len([k for k in d.keys() if k in list(subdataclass.__dict__.keys())]) == len(d.keys()):
                    return subdataclass.from_dict(d)
            raise AttributeError(f"Could not parse ListGenArgs method_args: {str(d)}")

        method_args: Union[BeamSearchArgs, ModelSamplingArgs, LatticeMBRArgs, LatticeSamplingArgs, dict] = field(
                metadata=config(
                    decoder= lambda d: Args.ListGenArgs.decode(d)
                )
            )
        outfile: Optional[str] = field(default=None)
        k: Optional[int] = field(
            default=50,
            metadata={"help": "Length of top-k list"},
        )

    @dataclass_json
    @dataclass
    class ListRerankArgs:
        """
        Arguments pertaining to topk-list reranking
        """
        rerank_metric: Optional[str
            # Literal['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rouge5', 'rouge6', 
            #         'bertscore', 'bartscore']
        ] = field(default=None)

        rerank_temp: Optional[float] = field(
            metadata=config(
                mm_field=mm.fields.Float(allow_nan=True)
            ), 
            default=float('inf')
        )
            
        rerank_geo_mean: Optional[bool] = field(
            default=False,
            metadata={"help": "use geo mean of rouges up to specified order"}
        )

    @dataclass_json
    @dataclass
    class EvalArgs:
        """
        Arguments pertaining to topk-list evaluation
        """
        eval_metrics: str = field(
            default='rouge1,rouge2,rougeL,chrf',
            metadata={"help": "comma-separated list of metrics to evaluate hypos on"}
        )
        outfile: Optional[str] = field(default=None)


    pipeline: Optional[PipelineArgs]
    gen: Optional[ListGenArgs]
    rerank: Optional[ListRerankArgs]
    eval: Optional[EvalArgs]

def load_args(config_file, wandb=None):
    # json to args with validation
    import json
    config = json.dumps(json.load(open(config_file)))
    args = Args.schema().loads(config)
    
    if wandb:
        wandb.config.update(config)

    return args