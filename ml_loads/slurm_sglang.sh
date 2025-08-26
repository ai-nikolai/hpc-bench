#!/bin/bash
#SBATCH -J vllm_experiment
#SBATCH -N 1
#SBATCH --ntasks-per-node=1 #2
#SBATCH --gpus-per-node=4 #8
#SBATCH -t 01:00:00
#SBATCH --mem=128G #128G is not working #requesting more than 128 leads to an error.

#SBATCH -p faculty
#SBATCH --qos=gtqos    #the other option is stqos

#SBATCH -o ./logs/sglang_experiment/run_%j_%N_%s_%t.log
#SBATCH -e ./logs/sglang_experiment/run_%j_%N_%s_%t.log


# TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S") # Format: YYYY-MM-DD_HH-MM-SS

NUM_GPUS=1


mkdir -p ./logs/vllm_experiment/

module load apptainer

# export APPTAINER_BINDPATH=//path/to/dataset/://path/to/dataset/,/vast:/vast

cd ~/z_code/hpc-bench/
mkdir -p logs

source env_vllm/bin/activate


python ./ml_loads/simple_inference_sglang.py --num_gpus ${NUM_GPUS} --model_name "Qwen/Qwen2.5-7B-Instruct"

# Optional command to block nodes.
# SBATCH -x auh7-1b-gpu-188,auh7-1b-gpu-185,auh7-1b-gpu-186,auh7-1b-gpu-303,auh7-1b-gpu-302
