#!/bin/bash

# to submit this script, do sbatch submit_slurm_make_histos.sh

# job name
#SBATCH --job-name=make_histos

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

# SLURM array: one job per input/id/signal set (0-10 for 11 sets, only run up to 8 simultaneously)
#SBATCH --array=0-10%8

# email notifications
#SBATCH --mail-user=toni.mlinarevic.20@ucl.ac.uk
#SBATCH --mail-type=ALL

# change log names; %j gives job id, %x gives job name, %a gives array index
#SBATCH --output=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/slurm-%j.%a.%x.out
# optional separate error output file
# #SBATCH --error=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/slurm-%j.%a.%x.err

# speedup trick
# export OMP_NUM_THREADS=1

# Space-separated string of patterns
FTAG1_QCD_P8=(
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364703.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root \
     /share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364704.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root \
     /share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364705.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root \
     /share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364706.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root \
     /share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364707.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
)
JETM2_W_flatmass=(
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_old/mc20_13TeV.802017.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT_wideWmass.deriv.DAOD_JETM2.e8482_s3797_r13145_p5548_files_1-50.root
     /share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_old/mc20_13TeV.802017.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT_wideWmass.deriv.DAOD_JETM2.e8482_s3797_r13145_p5548_files_51-99.root"
)

# Array of input pattern lists, each element is a space-separated string of patterns
infile_lists=(
    # FTAG1 Zprime ttbar sample
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.426345.e6880_s3681_r13144_p5981.FTAG1_TV3_ANALYSIS.root/*.root"
    # FTAG1 QCD samples, to be used with top selection
    "$FTAG1_QCD_P8"
    # alternative QCD samples, to be used with top selection
    "/share/lustreslow/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_JeanPierre/Herwing_dipole/*.root"
    "/share/lustreslow/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_JeanPierre/Sherpa_Cluster/*.root"
    "/share/lustreslow/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_JeanPierre/Sherpa_Lund/*.root"
    # FTAG1 W flat pT sample
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.801859.e8482_s3681_r13144_p6781.FTAG1_TV3_ANALYSIS.root/*.root"
    # FTAG1 QCD samples, to be used with W selection
    "$FTAG1_QCD_P8"
    # alternative QCD samples, to be used with W selection
    "/share/lustreslow/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_JeanPierre/Herwing_dipole/*.root"
    "/share/lustreslow/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_JeanPierre/Sherpa_Cluster/*.root"
    "/share/lustreslow/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_JeanPierre/Sherpa_Lund/*.root"
    # JETM2 W flat mass sample
    "$JETM2_W_flatmass"
)

outfiles=(
    "histos/mass_40-3500_pt_350-3100/topBSMP8_426345.root"
    "histos/mass_40-3500_pt_350-3100/qcdP8.root"
    "histos/mass_40-3500_pt_350-3100/qcdHD.root"
    "histos/mass_40-3500_pt_350-3100/qcdSC.root"
    "histos/mass_40-3500_pt_350-3100/qcdSL.root"
    "histos/mass_40-300_pt_200-3100/WBSMP8.root"
    "histos/mass_40-300_pt_200-3100/qcdP8.root"
    "histos/mass_40-300_pt_200-3100/qcdHD.root"
    "histos/mass_40-300_pt_200-3100/qcdSC.root"
    "histos/mass_40-300_pt_200-3100/qcdSL.root"
    "histos/mass_40-300_pt_200-3100/WBSMP8_flat.root"
)

signals=(
    top
    top
    top
    top
    top
    W
    W
    W
    W
    W
    W
)

cd ~/Lund_tagging/lundtoptagger
echo "Moved dir, now in:"
pwd

echo "Hostname:"
hostname

echo "Activating environment"
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc12-opt/setup.sh

echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

# Check array bounds
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#infile_lists[@]}" ] || \
   [ "$SLURM_ARRAY_TASK_ID" -ge "${#outfiles[@]}" ] || \
   [ "$SLURM_ARRAY_TASK_ID" -ge "${#signals[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds for job arrays."
    exit 1
fi

# Select the current parameters based on SLURM_ARRAY_TASK_ID
infile_list="${infile_lists[$SLURM_ARRAY_TASK_ID]}"
outfile="${outfiles[$SLURM_ARRAY_TASK_ID]}"
signal="${signals[$SLURM_ARRAY_TASK_ID]}"

# Split the selected infile_list string into an array
IFS=' ' read -r -a infile_patterns <<< "$infile_list"

echo ""
echo "infile patterns: ${infile_patterns[@]}"
echo "signal: $signal"

echo "Running training script..."
echo ""

python make_histos.py \
    --signal "$signal" \
    --infile "${infile_patterns[@]}" \
    --outfile "$outfile" \
    --nbins 100