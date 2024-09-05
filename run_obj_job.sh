#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00          # walltime
python 3d_obj_gen.py --model_name ResUNet_rgb --fusion late --expr_name NYUCAD_tanh_identity_f1 --gt_path './NYUCAD_gt_pred/' --output_path './obj/' --weights_2D "./saved_models/2D_pretrained_late_tanh_identity_fold1_2024-01-26.pth" --weights_3D "./saved_models/ResUNet_rgb_late_tanh_identity_fold1_2024-01-26.pth"