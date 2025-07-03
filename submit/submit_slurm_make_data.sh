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
#SBATCH --mem=35G

# SLURM array: one job per input/id/signal set and event fraction
# only run up to 10 simultaneously
# number of elements should be equal to NUM_INPUTS * NUM_EVENT_FRACTIONS
# last index is included in the array
#SBATCH --array=0-174%10

# email notifications
#SBATCH --mail-user=toni.mlinarevic.20@ucl.ac.uk
#SBATCH --mail-type=ALL

# change log names; %j gives job id, %x gives job name, %a gives array index
#SBATCH --output=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/make_data/slurm-%j.%a.%x.out
# optional separate error output file
# #SBATCH --error=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/make_data/slurm-%j.%a.%x.err

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
    # "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_old/mc20_13TeV.801859.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT.deriv.DAOD_JETM2.e8482_s3681_r13145_p5548_files_3-4.root" \
)
ids=( \
    QCD_364703 \
    QCD_364704 \
    QCD_364705 \
    QCD_364706 \
    QCD_364707 \
    Zprime_tt_426345 \
    W_flat_pt_801859 \
)
signals=( \
    top \
    top \
    top \
    top \
    top \
    top \
    W \
)

NUM_INPUTS=7
NUM_EVENT_FRACTIONS=25

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

# Compute indices for event fraction and input set
event_fraction_idx=$(( SLURM_ARRAY_TASK_ID / NUM_INPUTS ))
input_set_idx=$(( SLURM_ARRAY_TASK_ID % NUM_INPUTS ))

path_to_rootfiles="${input_paths[$input_set_idx]}"
id="${ids[$input_set_idx]}"
signal="${signals[$input_set_idx]}"

echo ""
echo "path_to_rootfiles: $path_to_rootfiles"
echo "id: $id"
echo "signal: $signal"
echo "event_fraction_idx: $event_fraction_idx"

echo "Running training script..."
echo ""
python Make_data.py configs/config_make_data.yaml --override \
    out_dir="/share/lustre/tmlinare/Lund_tagging/graphs/v2.1.8_GN2X_m40-inf_pt200-3100/data{frac}" \
    path_to_rootfiles="$path_to_rootfiles" \
    id="$id" \
    signal="$signal" \
    event_fraction_idx="$event_fraction_idx"