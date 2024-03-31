#!/bin/bash

#SBATCH -A mdredze1_gpu
#SBATCH --partition ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu=60GB
#SBATCH --qos=qos_gpu
#SBATCH --job-name="medalpaca"
#SBATCH --output="./log/log-%a-%j.txt" # Path to store logs


export CUDA_VISIBLE_DEVICES=0,1,2,3
export CONV_RSH=ssh
export HF_HOME=/data/mdredze1/zfang/cache

srun python -m test_medalpaca