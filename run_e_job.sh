#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00          # walltime

python evaluation.py --dataset NYUCAD --model_name ResUNet_rgb --fusion late --expr_name tanh_identity --weights_2D "./saved_models/2D_pretrained_late_tanh_identity_fold1_2024-01-26.pth" "./saved_models/2D_pretrained_late_tanh_identity_fold2_2024-01-26.pth" "./saved_models/2D_pretrained_late_tanh_identity_fold3_2024-01-27.pth"  --weights_3D "./saved_models/ResUNet_rgb_late_tanh_identity_fold1_2024-01-26.pth" "./saved_models/ResUNet_rgb_late_tanh_identity_fold2_2024-01-26.pth" "./saved_models/ResUNet_rgb_late_tanh_identity_fold3_2024-01-27.pth" 