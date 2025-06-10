#!/bin/bash

# to submit this script, do sbatch submit_slurm.sh

# job name
#SBATCH --job-name=make_data

# choose the RCIF queue
#SBATCH -p RCIF

# request one node
#SBATCH -N1
# do not share nodes with other running jobs
# #SBATCH --exclusive

# keep environment variables
#SBATCH --export=ALL

# request CPUs
#SBATCH -n4

# request enough memory - probaby don't need this much
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

cd ~/Lund_tagging/GN2X
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
python Make_data.py configs/config_make_data.yaml