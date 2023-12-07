# MBR Decoding Update

To run MBR decoding, assuming you have already generated lattice results, you can run
```
python3 src/mbr_rouge.py [args]
```
Note that outputs for BFS+recomb can be pulled from DVC; after pulling them, they can be found in `output/data/sum_xsum_bfs_recom_16_35_False_0.4_True_False_4_5_zip_0.75_0.0_0.9`.

See the 'mbr' argument group in `setup.py` for a list of valid arguments. As an example, for MBR with ROUGE-1 lattice approximation, count awareness, and length bounding of 4, run:
```
python3 src/mbr_rouge.py --lattice_metric rouge1 --count_aware --d_length 4
```
To run our current best method (43.2 R1 / 20.5 R2 / 37.2 RL; MBR with unigram match lattice approximation, length bounding of 4, second-stage reranking over k=10 hypotheses, geometric mean of ROUGE1-6 as the reranking metric, and uniform scoring for unigram match):
```
python3 -i src/mbr_rouge.py --lattice_metric match1 --d_length 4 --lattice_topk 10 --rerank_rouge 6 --match_uniform
```

**This repo was built on the code from [Massive-scale Decoding for Text Generation using Lattices](https://arxiv.org/abs/2112.07660)
[Jiacheng Xu](https://jiacheng-xu.github.io/), Sid Reddy, [Greg Durrett](https://www.cs.utexas.edu/~gdurrett/)! If you use this code, please also cite their paper:**
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
