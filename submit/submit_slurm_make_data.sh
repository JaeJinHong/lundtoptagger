#!/bin/bash

# to submit this script, do sbatch submit_slurm_make_data.sh

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

# SLURM array: one job per input/id pair (0-6 for 7 pairs, only run 4 simultaneously)
#SBATCH --array=0-6%4

# email notifications
#SBATCH --mail-user=toni.mlinarevic.20@ucl.ac.uk
#SBATCH --mail-type=ALL

# change log names; %j gives job id, %x gives job name, %a gives array index
#SBATCH --output=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/slurm-%j.%a.%x.out
# optional separate error output file
# #SBATCH --error=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/slurm-%j.%a.%x.err

# speedup trick
# export OMP_NUM_THREADS=1

input_paths=( \
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364703.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root" \
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364704.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root" \
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364705.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root" \
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364706.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root" \
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364707.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root" \
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.426345.e6880_s3681_r13144_p5981.FTAG1_TV3_ANALYSIS.root/*.root" \
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.801859.e8482_s3681_r13144_p6781.FTAG1_TV3_ANALYSIS.root/*.root" \
)
ids=( \
    QCD_364703 \
    QCD_364704 \
    QCD_364705 \
    QCD_364706 \
    QCD_364707 \
    Zprime_tt_426345 \
    W_flat_pt_801859_801859 \
)

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

# Select the current pair based on SLURM_ARRAY_TASK_ID
path_to_rootfiles="${input_paths[$SLURM_ARRAY_TASK_ID]}"
id="${ids[$SLURM_ARRAY_TASK_ID]}"
echo ""
echo "path_to_rootfiles: $path_to_rootfiles"
echo "id: $id"

echo "Running training script..."
echo ""
python Make_data.py configs/config_make_data.yaml --override path_to_rootfiles="$path_to_rootfiles" id="$id"