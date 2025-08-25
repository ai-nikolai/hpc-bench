#!/bin/bash
#SBATCH -J vllm_experiment
#SBATCH -N 1
#SBATCH --ntasks-per-node=1 #2
#SBATCH --gpus-per-node=8 #8
#SBATCH -t 01:00:00
#SBATCH --mem=128G #128G is not working #requesting more than 128 leads to an error.

#SBATCH -p faculty
#SBATCH --qos=gtqos    #the other option is stqos

#SBATCH -o ./logs/vllm_experiment/run_%j_%N_%s_%t.log
#SBATCH -e ./logs/vllm_experiment/run_%j_%N_%s_%t.log


# TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S") # Format: YYYY-MM-DD_HH-MM-SS

mkdir -p ./logs/vllm_experiment/

module load apptainer

# export APPTAINER_BINDPATH=//path/to/dataset/://path/to/dataset/,/vast:/vast

cd ~/z_code/hpc-bench/
mkdir -p logs

# srun --ntasks=2 --gpus=4 \
#   apptainer exec --rocm rocm-dev-5.7.sif \
#   python train_ddp.py --data /path/to/dataset/ --out /vast/run_ddp
NUM_GPUS=1

srun --ntasks=1 --gpus=${NUM_GPUS} \
  apptainer exec --rocm ./apptainer_images/vllm-rocm.sif \
  python ./ml-loads/simple_inference.py --num_gpus ${NUM_GPUS}