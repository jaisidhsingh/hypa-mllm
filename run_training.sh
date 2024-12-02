#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -c 8
#SBATCH --job-name=vitb32_llama3.2_lin_pt
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=/home/mila/s/sparsha.mishra/scratch/hyperalignment/slurm_logs/logs-%j.out
#SBATCH --error=/home/mila/s/sparsha.mishra/scratch/hyperalignment/slurm_errors/error-%j.err

pyfile="/home/mila/s/sparsha.mishra/projects/hypa-mllm/main.py"

module load anaconda/3

conda activate /home/mila/s/sparsha.mishra/.conda/envs/sparse

ulimit -Sn $(ulimit -Hn)

python3 $pyfile \
    --random_seed=0 \
    --batch_size=32 \
    --learning_rate=1e-3 \
    --num_workers=8 \
    --use_wandb=True;
