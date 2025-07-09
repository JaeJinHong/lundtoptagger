#!/bin/bash

# to submit this script, do sbatch submit_slurm_scores.sh

# job name
# #SBATCH --job-name=lundtoptagger_full_a100_8CPUs
#SBATCH --job-name=lundtoptagger_eval

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
# #SBATCH --constraint='a100|l40s|v100'
#SBATCH --constraint='v100'

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
python test_make_scores.py configs/config_make_scores.yaml  --override \
    data.sample="v2.1.5_GN2X_m40-inf_pt200-3100_0.25percent" \
    data.paths_to_test_file_root="[ \
        '/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/graphs/{sample}/*W_flat_pt*.root',
        '/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/graphs/{sample}/*QCD*.root'
    ]" \
    data.paths_to_test_file_graphs="[ \
        '/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/graphs/{sample}/graphs*W_flat_pt*',
        '/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/graphs/{sample}/graphs*QCD*'
    ]" \
    data.path_to_outdir="/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/scores/data_{sample}_scores_v2.2.3/" \
    data.output_suffix="_scores" \
    test.path_to_combined_ckpt.null="/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run34_v2.1.5_adversarial_rafael_graphs_SmallerW_Weights/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_comb_e200_0.03981.pt" \
    test.scores_branch_name=fjet_{model}_run34_rafael_graphs_adv_e200