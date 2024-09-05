#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00          # walltime

# to extract 2D semantic labels from nyu_depth_v2_labeled.mat
python extract_nyu_2d_labels.py --dataset NYUCAD
