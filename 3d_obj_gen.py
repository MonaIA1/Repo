import os
import pandas as pd
import torch
import torch.nn.functional as F
#import torch.nn as nn
#import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils.data_setup import SSCData
from utils.file_utils import get_all_preprocessed_prefixes
from model.mdbnet import get_res_unet_rgb, get_2Dfeatures, get_activation
#from model.network_2 import get_res_unet, get_res_unet_rgb, get_2Dfeatures, get_activation
from utils.visual_utils import voxel_export, obj_export
import argparse


# data paths
GT_PATH = './NYU_gt_pred/'
OUTPUT_PATH = './obj/'
BATCH_SIZE = 1
EXPR_NAME =''
MODEL_NAME ='ResUNet_rgb'
#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 6
SAVED_WEIGHTES_3D = ''
SAVED_WEIGHTES_2D =''
PRE_TRAINED_MODEL = "./pretrained_segformer_b5_ade-640-640"
IMAGE_SIZE = (480, 640)
FUSION_LOC = 'late'

######################################################################
def parse_arguments():
    global EXPR_NAME, GT_PATH, OUTPUT_PATH, MODEL_NAME, SAVED_WEIGHTES_2D,SAVED_WEIGHTES_3D, FUSION_LOC
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model name", type=str)
    parser.add_argument("--fusion", help="fusion location", type=str, choices=['early', 'mid', 'late'])
    parser.add_argument("--expr_name", help="experiment name", type=str)
    parser.add_argument("--gt_path", help="gt path of the original data. Default: "+GT_PATH, type=str, default=GT_PATH, required=False)
    parser.add_argument("--output_path", help="output obj path. Default: "+OUTPUT_PATH, type=str, default=OUTPUT_PATH, required=False)
    parser.add_argument('--weights_2D', metavar='File', type=str, nargs='+', help='a 2D model weight file to process')
    parser.add_argument('--weights_3D', metavar='File', type=str, nargs='+', help='a 3D model weight file to process')
    args = parser.parse_args()

    #####################################################
    MODEL_NAME = args.model_name
    EXPR_NAME = args.expr_name
    GT_PATH = args.gt_path
    OUTPUT_PATH = args.output_path
    SAVED_WEIGHTES_2D = args.weights_2D
    SAVED_WEIGHTES_3D = args.weights_3D
    FUSION_LOC = args.fusion
##################################################################

def obj_generation():
  print('Obj path', GT_PATH) 
  v_unit = 0.02
  xs, ys, zs = 60, 36, 60
  file_prefixes = get_all_preprocessed_prefixes(GT_PATH, criteria="*.npz")
  
  df = pd.DataFrame(data={"filename": file_prefixes})
  gt_data = SSCData(df['filename'].tolist(), train= False )
  
  gt_loader = DataLoader(gt_data, batch_size= 1, shuffle=False,num_workers=NUM_WORKERS)
  
  
  model2D_, img_processor = get_2Dfeatures(PRE_TRAINED_MODEL) 
  model_ = get_res_unet_rgb(FUSION_LOC) # 3D model
  
  #######################################################
  # make device agnostic code
  dev = 'cuda:0'
  device = torch.device(dev) if torch.cuda.is_available() else torch.device("cpu")
  ######################################################
  
  
  # 1- load the saved state dict
  state_dict_2d = torch.load(f=SAVED_WEIGHTES_2D[0])
  state_dict_3d = torch.load(f=SAVED_WEIGHTES_3D[0])
  
  # 2- create a new state dict in which 'module.' prefix is removed 'if multple GPUs'
  new_state_dict_2d = {k.replace('module.', ''): v for k, v in state_dict_2d.items()}
  new_state_dict_3d = {k.replace('module.', ''): v for k, v in state_dict_3d.items()} 
  
  # 3- load the new state dict to your model      
  model2D_.load_state_dict(new_state_dict_2d)
  model_.load_state_dict(new_state_dict_3d)
  
  # send model to GPU
  model2D_ = model2D_.to(device)
  model_ = model_.to(device)
  
  
  
  for batch_num, sample in enumerate(gt_loader):
      
      rgb = sample['rgb'].to(device)
      x = sample['vox_tsdf'].to(device) # F-TSDF
      y = sample['vox_lbl'].to(device)
      w = sample['vox_weight'].to(device) # weights calculated in preprocessing
      m = sample['vox_mask'].to(device)
      label_2d = sample['label_2d'].to(device)
      depth_3d = sample['vox_depth'].to(device)    
      
      out_file = os.path.basename(file_prefixes[batch_num])
      print(f'file: {out_file}')
      
      y = y.argmax(dim=1)
      label= y[0]*w[0]
      print(f"label {label.shape}")
      
      print(torch.unique(label))
      voxel_export(OUTPUT_PATH + MODEL_NAME + '_'+ FUSION_LOC+'_'+ EXPR_NAME+'_'+ out_file+'.bin', label) 
      obj_export(OUTPUT_PATH + MODEL_NAME + '_'+ FUSION_LOC+'_'+ EXPR_NAME+'_'+ str(out_file)+'_obj', label, (xs,ys,zs), 0, 0, 0, v_unit, include_top=True,triangular=False,inner_faces=True)
      
      # get the predictions
      model_.eval()
      model2D_.eval()
      with torch.inference_mode():
        
        # get the 2d feature maps from pretrained model
        imgs = img_processor(images=rgb, return_tensors="pt").to(device)
        
        hook,activation = get_activation('linear_fuse')
        model2D_.decode_head.register_forward_hook(hook)
        cls = model2D_(**imgs) # predected 2d classes
        classes_logits = cls.logits # logits from classification layer
        feature2d = activation['linear_fuse'] # the activation maps befor the classification layer
        
        # upsample the output to match NYU images size
        feature2d = F.interpolate(feature2d, size=IMAGE_SIZE, mode='bilinear', align_corners=True)
        
        # input the predicted 2D features to the 3D model alongside with depth data
        pred = model_(rgb, x,feature2d, depth_3d, device)

        print(f'pred shape:{pred.shape}')
        pred = pred.argmax(dim=1)
        print(f"predection {pred.shape}")
        label_pred= pred[0]*w[0]
        
        print(f"label_pred shape: {label_pred.shape}")
        print(torch.unique(label_pred))
        
        voxel_export(OUTPUT_PATH + MODEL_NAME + '_'+ FUSION_LOC+'_'+ EXPR_NAME+'_'+"pred_"+out_file+'.bin', label_pred) 
        obj_export(OUTPUT_PATH + MODEL_NAME + '_'+ FUSION_LOC+'_'+ EXPR_NAME+'_'+"pred_"+str(out_file)+'_obj', label_pred, (xs,ys,zs), 0, 0, 0, v_unit, include_top=True,triangular=False,inner_faces=True)
      print(f'-------------------------------------------------------------------------------------------------------') 
# Main Function
def main():

    parse_arguments()
    obj_generation()
    
if __name__ == '__main__':
  main()        

