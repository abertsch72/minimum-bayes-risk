#!/bin/bash
#SBATCH --job-name=gen-zip-2
#SBATCH --output=/projects/tir4/users/abertsch/lattice-search/logs/%x.out
#SBATCH --err=/projects/tir4/users/abertsch/lattice-search/logs/%x.err 
#SBATCH --time=60:00:00
#SBATCH	--mem=32Gb
#SBATCH --gres=gpu:1
#SBATCH --exclude=tir-0-[3,11,32,36],tir-1-[13,18]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lattice_search 
cd /projects/tir4/users/abertsch/lattice-search/ 

PYTHONPATH=./ python src/recom_search/scripts/run_pipeline.py -nexample 5000  -ngram_suffix 4 -beam_size 16 -min_len 10 -max_len 35 -model bfs_recom -merge zip  -avg_score 0.75  -dfs_expand -device cuda:0 -startexample 6000


