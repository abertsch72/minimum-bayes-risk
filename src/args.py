from transformers import HfArgumentParser
from typing import Optional, Literal
from dataclasses import dataclass, field


@dataclass
class Args:
    """
    General arguments
    """
    hf_model_name: Optional[str] = field(default='facebook/bart-large-xsum')
    wandb: Optional[bool] = field(default=False)
    wandb_group: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)
    seed: Optional[int] = field(default=None)
    no_tqdm: Optional[bool] = field(default=False)


@dataclass
class ListGenArgs:
    """
    General arguments for top-k list generation
    """
    outfile: Optional[str] = field(default=None)
    k: Optional[int] = field(
        default=50,
        metadata={"help": "Length of top-k list"},
    )


@dataclass
class BeamSearchArgs:
    """
    Arguments pertaining to beam search top-k list generation
    """
    beam_width: Optional[int] = field(default=50)


@dataclass
class NucleusSamplingArgs:
    pass


@dataclass
class LatticeMBRArgs:
    """
    Arguments pertaining to MBR over lattice for top-k list generation
    """
    lattice_dir: str = field(
        default='',
        metadata={'help': 'directory where lattices are stored'}
    )
    lattice_metric: Literal['rouge1', 'rouge2', 'match1', 
                            'match2', 'exact_rouge1'] = field(
        default="rouge1"
    )
    uniform_match: Optional[bool] = field(
        default=False,
        metadata={'help': 'use uniform scoring for expected match calculation only'}
    )
    candidate_length_deviation: Optional[int] = field(
        default=float('inf'),
        metadata={"help": "Min/max deviation of candidate set length from mean"},
    )
    evidence_length_deviation: Optional[int] = field(
        default=float('inf'),
        metadata={"help": "Min/max deviation of evidence set length from mean"},
    )
    mean_override: Optional[int] = field(
        default=-1,
        metadata={'help': 'value to override the mean with (-1 means use the actual mean)'},
    )
    lattice_score_temp: Optional[float] = field(
        default=1.0,
        metadata={'help': 'temperature on lattice edge probabilities'},
    )


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


@dataclass
class ListRerankArgs:
    """
    Arguments pertaining to topk-list reranking
    """
    rerank_metric: Optional[str
        # Literal['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rouge5', 'rouge6', 
        #         'bertscore', 'bartscore']
    ] = field(default=None)
    rerank_temp: Optional[float] = field(default=float('inf'))


@dataclass
class EvalArgs:
    """
    Arguments pertaining to topk-list evaluation
    """
    eval_metrics: str = field(
        default='rouge1,rouge2,rougeL',
        metadata={"help": "comma-separated list of metrics to evaluate hypos on"}
    )
    outfile: Optional[str] = field(default=None)


def get_parser(
    beamsearch=False, 
    latticembr=False, 
    latticesamp=False, 
    nuclsamp=False,
):
    dataclass_types = [Args]
    if any([beamsearch, latticembr, latticesamp, nuclsamp]):
        dataclass_types.append(ListGenArgs)
        for should_add, data_cls in zip(
            [beamsearch,     latticembr,     latticesamp,         nuclsamp],
            [BeamSearchArgs, LatticeMBRArgs, LatticeSamplingArgs, NucleusSamplingArgs]
        ):
            if should_add:
                dataclass_types.append(data_cls)
    else:
        dataclass_types.append(ListRerankArgs)
        dataclass_types.append(EvalArgs)

    parser = HfArgumentParser(dataclass_types)
    return parser
