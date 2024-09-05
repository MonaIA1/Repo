# Adapted from: https://gitlab.com/UnBVision/spawn/-/tree/main?ref_type=heads
import h5py
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys
from utils import nyu
import argparse

PATH_CONFIG = './data/'
DATASET = 'NYU'
def parse_arguments():
    global PATH_CONFIG,DATASET

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Target dataset", type=str, choices=['NYU', 'NYUCAD'])
    parser.add_argument("--path_config", help="The dir where the NYU data exist. Default: "+PATH_CONFIG, type=str, default=PATH_CONFIG, required=False)
    
    args = parser.parse_args()

    PATH_CONFIG = args.path_config
    DATASET = args.dataset
    
def Run():
    print("\nDataset:", DATASET)
    NYU_V2_PATH = PATH_CONFIG
    matlab_file = os.path.join(NYU_V2_PATH,'nyu_depth_v2_labeled.mat')
    print("Reading NYU Matlab file from ", matlab_file, "...", sep="")
    f = h5py.File(matlab_file, 'r')
    labels = np.array(f['labels'])
    
    class_colors = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, .7, 0],
        [0.66, 0.66, 1],
        [0.8, 0.8, 1],
        [1, 1, 0.0392201],
        [1, 0.490452, 0.0624932],
        [0.657877, 0.0505005, 1],
        [0.0363214, 0.0959549, 0.6],
        [0.316852, 0.548847, 0.186899],
        [.8, .5, .1],
        [1, 0.241096, 0.718126],
      ])
    
    class_colors_bgr = np.flip(class_colors,1)
    
    map894_11 = np.array(nyu.get_894_11_class_mapping())
    
    print("Processing labels...")
    for n, label_data in tqdm(enumerate(labels),total=len(labels), file=sys.stdout):
    
        label_image = class_colors_bgr[map894_11[label_data.T]] * 255
    
        file_name = 'NYU{:04d}'.format(n+1) + '_0000'
        
        if (DATASET == 'NYU'):
          if os.path.isfile(os.path.join(NYU_V2_PATH,'NYUtest',file_name+'.bin')):
              cv2.imwrite(os.path.join(NYU_V2_PATH,'NYUtest',file_name+'_labels_rgb.png'), label_image)# RGB_channel_labels
              cv2.imwrite(os.path.join(NYU_V2_PATH,'NYUtest',file_name+'_labels.png'), map894_11[label_data.T]) # Gray_scale channel labels[(one channel)machine freindly --> use it as 2d GT, 12 classes labels from 0-11]
          elif os.path.isfile(os.path.join(NYU_V2_PATH,'NYUtrain',file_name+'.bin')):
              cv2.imwrite(os.path.join(NYU_V2_PATH,'NYUtrain',file_name+'_labels_rgb.png'), label_image)
              cv2.imwrite(os.path.join(NYU_V2_PATH,'NYUtrain',file_name+'_labels.png'), map894_11[label_data.T])
          else:
            print("Warning: NYU 3D labels not found: ", file_name+".bin")
            print("Have you downloaded it? Check your path.conf file.")
            break
            
        if (DATASET == 'NYUCAD'):
          if os.path.isfile(os.path.join(NYU_V2_PATH,'NYUCADtrain',file_name+'.bin')):
              cv2.imwrite(os.path.join(NYU_V2_PATH,'NYUCADtrain',file_name+'_labels_rgb.png'), label_image)
              cv2.imwrite(os.path.join(NYU_V2_PATH,'NYUCADtrain',file_name+'_labels.png'), map894_11[label_data.T])
          elif os.path.isfile(os.path.join(NYU_V2_PATH,'NYUCADtest',file_name+'.bin')):
              cv2.imwrite(os.path.join(NYU_V2_PATH,'NYUCADtest',file_name+'_labels_rgb.png'), label_image)
              cv2.imwrite(os.path.join(NYU_V2_PATH,'NYUCADtest',file_name+'_labels.png'), map894_11[label_data.T])
          else:
            print("Warning: NYU 3D labels not found: ", file_name+".bin")
            print("Have you downloaded it? Check your path.conf file.")
            break
        
if __name__ == '__main__':
  parse_arguments()
  Run()
