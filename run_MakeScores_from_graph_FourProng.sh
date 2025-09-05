#!/bin/bash

# set -x

# $ eval "$(micromamba shell hook --shell=bash)"

export CONDA_PREFIX=/data/jjhong96/miniconda3
export PATH=${CONDA_PREFIX}/bin/:$PATH
source ${CONDA_PREFIX}/etc/profile.d/conda.sh

conda env list
conda init
conda activate weaver

python3 test_make_scores_FourProng.py /home/jjhong96/FourProngTagger/LundTopTagger/configs_FourProng/config_make_scores_from_graph.yaml