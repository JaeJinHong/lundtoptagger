#!/bin/bash

# to submit this script, do sbatch submit_slurm.sh

# job name
#SBATCH --job-name=lundtoptagger_full_8CPUs

# choose the GPU queue
#SBATCH -p GPU

# request one node
#SBATCH -N1
# do not share nodes with other running jobs
# #SBATCH --exclusive
# exclude the node with 40 GB A100 GPUs - this ensures that if A100 GPUs are requested, the ones with 80 GB are used
# #SBATCH --exclude=compute-gpu-0-3

# keep environment variables
#SBATCH --export=ALL

# request CPUs
#SBATCH -n8

# request GPUs
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|l40s|v100'

# request enough memory
#SBATCH --mem=50G

# email notifications
#SBATCH --mail-user=toni.mlinarevic.20@ucl.ac.uk
#SBATCH --mail-type=ALL

# change log names; %j gives job id, %x gives job name
#SBATCH --output=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/slurm-%j.%x.out
# optional separate error output file
# #SBATCH --error=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/slurm-%j.%x.err

# speedup trick
# export OMP_NUM_THREADS=1

cd ~/Lund_tagging/lundtoptagger
echo "Moved dir, now in:"
pwd

echo "Hostname:"
hostname

echo "Activating environment"
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126
echo $CONDA_DEFAULT_ENV

echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

echo "Running training script..."
echo ""
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml