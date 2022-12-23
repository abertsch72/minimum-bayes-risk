# MBR Decoding Update

To run MBR decoding, assuming you have already generated lattice results, you can run
```
python3 src/mbr_rouge.py [args]
```
Note that outputs for BFS+recomb are already stored in `output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9`. See the 'mbr' argument group in `setup.py` for a list of valid arguments. As an example, for MBR with ROUGE-1, count awareness, and length bounding of 4, run:
```
python3 src/mbr_rouge.py --rouge 1 --count_aware --d_length 4
```

# [Massive-scale Decoding for Text Generation using Lattices](https://arxiv.org/abs/2112.07660)
[Jiacheng Xu](https://jiacheng-xu.github.io/), Sid Reddy, [Greg Durrett](https://www.cs.utexas.edu/~gdurrett/)

TL;DR: a new search algorithm to construct lattices encoding many generation options; 
two key technical contributions: (1) best-first search, (2) path recombination.

## Visualization
We provide a few examples in the ```vis``` folder and on [my homepage](https://www.cs.utexas.edu/~jcxu/data/summarization/). You need to download the html files to view and **interact** with the model outputs.

The complete set of outputs are available on [Box](https://utexas.box.com/s/wmvhg8lol3kvgirizqyiyiblbn6ogj1a).

## Getting started


- ```model``` contains all of the methods, including baselines like beam search, nucleus sampling, and our methods.
- ```evaluation``` contains scripts for evaluation.
- ```command``` are the prompts and shells we use to run the experiment. 

Beam Search:
```
PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 100  -ngram_suffix 4  -beam_size 16 -min_len 10 -max_len 35   -model bs 
```

Best-first Search:
```
PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 100  -ngram_suffix 4  -beam_size 16 -min_len 10 -max_len 35   -model astar_baseline
```

Best-first Search with Recomb:
```
PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 100  -ngram_suffix 4 -beam_size 16 -min_len 10 -max_len 35 -model bfs_recom -merge rcb  -avg_score 0.75  -dfs_expand 
```

Best-first Search with Zip:
```
PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 100  -ngram_suffix 4 -beam_size 16 -min_len 10 -max_len 35 -model bfs_recom -merge zip  -avg_score 0.75  -dfs_expand 
```
More detailed instructions coming soon!

## Citation
```
@misc{xu-etal-2022-massive,
    title={Massive-scale Decoding for Text Generation using Lattices},
    author = {Xu, Jiacheng and Jonnalagadda, Siddhartha Reddy and Durrett, Greg},
    year={2022},
    eprint={2112.07660},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Contact

jcxu@utexas.edu 