#!/bin/bash
# srun -p alpha-interactive -N 1 -n 1 -c 96 --exclusive --gres=gpu:1 --hint=multithread --time=16:00:00 --pty bash   #job submission in ml nodes with allocating: 1 node, 1 task per node, 6 CPUs per task, 1 gpu per node, with 32000 mb on 8 hours.

module load modenv/hiera  GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5 PyTorch/1.9.0
module load Miniconda3

conda init

# Create new Conda env by cloning from base env
ENV_NAME="apnea-detection"
ENV_DIR="$HOME/.conda/envs/$ENV_NAME"
if [ ! -d $ENV_DIR ]
then
  echo "--> Creating Conda env now. That might take a while! <--"
  conda env create --name $ENV_NAME --file environment.yml
  conda activate $ENV_NAME
  conda install -y pytorch cudatoolkit=11.1 -c pytorch -c nvidia
fi

conda activate $ENV_NAME
conda deactivate
conda activate $ENV_NAME
