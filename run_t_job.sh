#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=60:00:00          # walltime
export CUDA_VISIBLE_DEVICES=0

python train.py --dataset NYUCAD --model_name ResUNet_rgb --fusion late --expr_name tanh_identity --train_batch_size 4 --val_batch_size 2 --base_lr_2d 0.0001 --base_lr_3d 0.01 --decay_2DModel 0.05 --decay_3DModel 0.0005 --epochs 100

