#!/bin/bash
#SBATCH --job-name=train-pro6000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=94G
#SBATCH --time=1-24:00:00
#SBATCH --output=logs/%x-%j.out

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
nvidia-smi

# ---- FIX CONDA IN NON-INTERACTIVE SLURM SHELL ----
source /home/ilinca/miniforge3/etc/profile.d/conda.sh
conda activate thesis-env

# ---- ENV VARS ----
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# ---- RUN TRAINING ----
cd /home/flaviu/Local/SSLFoundationModel_LymphNodesSegmentation_BreastCancer

python -m src.VocoLarge.segmentation.multiclass_segmentation.after_ssl_domain_train.main