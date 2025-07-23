#!/bin/bash
#SBATCH --job-name=wav2vec_all
#SBATCH --output=logs/wav2vec_all.out
#SBATCH --error=logs/wav2vec_all.err
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

echo "Starting at $(date) on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asr_env
export HF_HUB_READ_TIMEOUT=60

echo "Running Wav2Vec2 on CLEAN data..."
python scripts/wav2vec_infer.py --split clean

echo "Running Wav2Vec2 on NOISY data..."
python scripts/wav2vec_infer.py --split noisy

echo "Finished at $(date)"
