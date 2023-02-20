#!/bin/bash
#SBATCH --job-name=sweep-match
#SBATCH --output=/projects/tir4/users/abertsch/huggingface-knnlm-port/logs/%x.out
#SBATCH --err=/projects/tir4/users/abertsch/huggingface-knnlm-port/logs/%x.err
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=tir-0-[3,11,32,36],tir-1-[13,18]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lattice_search
cd /projects/tir4/users/abertsch/lattice-search/
export PYTHONPATH="${PYTHONPATH}:."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/abertsch/miniconda3/lib/
python3 sweep.py 

