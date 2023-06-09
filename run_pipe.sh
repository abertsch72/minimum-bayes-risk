#!/bin/bash
#SBATCH --job-name=mbr-pipe
#SBATCH --output=/projects/tir4/users/abertsch/lattice-search/logs/%x-%j.out
#SBATCH --err=/projects/tir4/users/abertsch/lattice-search/logs/%x-%j.err
#SBATCH --time=6:00:00
#SBATCH	--mem=16Gb
#SBATCH --gres=gpu:1
#SBATCH --exclude=tir-0-[3,11,32,36],tir-1-[13,18]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lattice_search
cd /projects/tir4/users/abertsch/lattice-search/

PYTHONPATH=./ python src/mbr_pipeline/entrypoint.py --config_file $1
