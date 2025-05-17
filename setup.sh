# setup for lxplus-gpu
if hostnamectl | grep -q "Red Hat Enterprise Linux 9"; then
    # set up the LCG release LCG_104cuda if the highest supported CUDA version is >= 11.8
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d"." -f1,2)
    if (( $(echo "$cuda_version >= 11.8" | bc -l) )); then
        echo "sourcing /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-el9-gcc11-opt/setup.sh"
        source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-el9-gcc11-opt/setup.sh
    fi

# setup for CentOS 7 machines with CVMFS access, no GPU
elif hostnamectl | grep -q "CentOS Linux 7"; then
    echo "sourcing /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc12-opt/setup.sh"
    source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc12-opt/setup.sh

# setup for UCL GPU server
elif [ $(hostname) == "gpu02" ]; then
    # set up conda
    __conda_setup="$('/mnt/storage/tmlinare/installs/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/mnt/storage/tmlinare/installs/miniforge3/etc/profile.d/conda.sh" ]; then
            . "/mnt/storage/tmlinare/installs/miniforge3/etc/profile.d/conda.sh"
        else
            export PATH="/mnt/storage/tmlinare/installs/miniforge3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    if [ -f "/mnt/storage/tmlinare/installs/miniforge3/etc/profile.d/mamba.sh" ]; then
        . "/mnt/storage/tmlinare/installs/miniforge3/etc/profile.d/mamba.sh"
    fi

    # activate conda environment
    conda activate /mnt/storage/tmlinare/conda/envs/pytorch_py39_cu102
fi