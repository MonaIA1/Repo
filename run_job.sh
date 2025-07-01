#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00          # walltime

# to preprocess the data 
python preproc_tsdf.py --base_path './data/NYUCADtest' --dest_path './data/NYUCAD_test_preproc'
