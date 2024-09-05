import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
import argparse
from tqdm.auto import tqdm

from utils.data_setup import SSCData
from utils.file_utils import get_all_preprocessed_prefixes 
from utils.engin import EarlyStopping, create_writer,save_model, print_train_time 
from model.mdbnet import get_res_unet_rgb, get_2Dfeatures, get_activation 
#from model.network_2 import get_res_unet, get_res_unet_rgb, get_2Dfeatures, get_activation 
#from model.network_2_mid import get_res_unet, get_res_unet_rgb, get_2Dfeatures, get_activation
from model.losses import WeightedCrossEntropyLoss, WCE_k3Clusters
from model.metrics import comp_IoU, m_IoU
from torch.utils.tensorboard import SummaryWriter
import wandb
os.environ["WANDB_API_KEY"] = 'Add your key'
os.environ["WANDB_MODE"] = "offline"


######################################################################
MODEL_NAME ='ResUNet'
BATCH_SIZE = 1
#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 6
FOLD_NUM = 1
SAVED_WEIGHTES_3D = ''
SAVED_WEIGHTES_2D =''
PRE_TRAINED_MODEL = "./pretrained_segformer_b5_ade-640-640"
IMAGE_SIZE = (480, 640)
FUSION_LOC = 'late'

def parse_arguments():
    global DATASET, MODEL_NAME,EXPR_NAME, PREPROC_PATH, SAVED_WEIGHTES_2D,SAVED_WEIGHTES_3D, FUSION_LOC
    print("\n Evaluation Script\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Target dataset", type=str, choices=['NYU', 'NYUCAD'])
    parser.add_argument("--model_name", help="model name", type=str)
    parser.add_argument("--fusion", help="fusion location", type=str, choices=['early', 'mid', 'late'])
    parser.add_argument("--expr_name", help="experiment name", type=str)
    parser.add_argument('--weights_2D', metavar='File', type=str, nargs='+', help='a 2D model weight file to process')
    parser.add_argument('--weights_3D', metavar='File', type=str, nargs='+', help='a 3D model weight file to process')
    
    args = parser.parse_args()
    ####################################################
    DATASET = args.dataset
    MODEL_NAME = args.model_name
    EXPR_NAME = args.expr_name
    SAVED_WEIGHTES_2D = args.weights_2D
    SAVED_WEIGHTES_3D = args.weights_3D
    FUSION_LOC = args.fusion
    if(DATASET == 'NYU')or (DATASET == 'nyu'):  
      PREPROC_PATH = './data/NYU_test_preproc/'
    elif (DATASET == 'NYUCAD') or (DATASET == 'nyucad'):
      PREPROC_PATH = './data/NYUCAD_test_preproc/'
    ###################################################
    
    
def model_evalauteion ():
  ################################### testing data loader ####################################################################
  file_prefixes_test = get_all_preprocessed_prefixes(PREPROC_PATH, criteria='*.npz')
  df_test = pd.DataFrame(data={"filename": file_prefixes_test})
  test_ds = SSCData(df_test['filename'],train =False) 
  test_loader = DataLoader(test_ds, batch_size= BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)
  fold_num = 1
  #########################################
  print("Dataset path:" , PREPROC_PATH)
  print("2D Model weights:", SAVED_WEIGHTES_2D)
  print("3D Model weights:", SAVED_WEIGHTES_3D)
  
  # Make device agnostic code
  dev = 'cuda:0'
  device = torch.device(dev) if torch.cuda.is_available() else torch.device("cpu")
  
  # define loss
  criterion = nn.CrossEntropyLoss(label_smoothing=0.2) # for 2d loss
  
  loss_function = WCE_k3Clusters(device)
  
  
  for w in range(len(SAVED_WEIGHTES_3D)):
    
    
    model2D_eval, img_processor = get_2Dfeatures(PRE_TRAINED_MODEL) 
    model_eval = get_res_unet_rgb(FUSION_LOC) # 3D model
              
        
    # initialize wandb for each weighted fold
    run = wandb.init(project=f'{MODEL_NAME}_{FUSION_LOC}_{EXPR_NAME}_testing', group='k-fold', name=f'fold-{fold_num}', reinit=True)
        
    # 1- load the saved state dict
    state_dict_2d = torch.load(f=SAVED_WEIGHTES_2D[w])
    state_dict_3d = torch.load(f=SAVED_WEIGHTES_3D[w])
    
    # 2- create a new state dict in which 'module.' prefix is removed 'if multiple GPUs used'
    new_state_dict_2d = {k.replace('module.', ''): v for k, v in state_dict_2d.items()}
    new_state_dict_3d = {k.replace('module.', ''): v for k, v in state_dict_3d.items()} 
    
    # 3- load the new state dict to your model      
    model2D_eval.load_state_dict(new_state_dict_2d)
    model_eval.load_state_dict(new_state_dict_3d)
    
    # send model to GPU
    model2D_eval = model2D_eval.to(device)
    model_eval = model_eval.to(device)
    
    model_eval.eval()
    model2D_eval.eval()
    
    with torch.inference_mode():
          total_loss, total_IoU, total_prec, total_recall, total_mIoU = 0,0,0,0,0
          #####################
          ceil_ls = []
          floor_ls = []
          wall_ls = []
          wind_ls = []
          chair_ls = []
          bed_ls = []
          sofa_ls = []
          table_ls = []
          tvs_ls = []
          furn_ls = []
          objs_ls = []
          ####################
          for batch_num, sample in enumerate(test_loader):
              test_loss = 0
              test_IoU = 0
              test_mIoU = 0
              test_prec = 0
              test_recall = 0
              seg_lis =[]
              
              rgb = sample['rgb'].to(device)
              label_2d = sample['label_2d'].to(device)
              inputs = sample['vox_tsdf'].to(device) # F-TSDF
              targets = sample['vox_lbl'].to(device)
              weights = sample['vox_weight'].to(device)
              masks = sample['vox_mask'].to(device)
              depth_3d = sample['vox_depth'].to(device)
              
              # get the 2d feature maps from pretrained model
              imgs = img_processor(images=rgb, return_tensors="pt").to(device)
              
              hook,activation = get_activation('linear_fuse')
              model2D_eval.decode_head.register_forward_hook(hook)
              cls = model2D_eval(**imgs) # predected 2d classes
              classes_logits = cls.logits # logits from classification layer
              feature2d = activation['linear_fuse'] # the activation maps 
              
              # Upsample the output to match NYU images size
              semantics2d = F.interpolate(classes_logits, size=IMAGE_SIZE, mode='bilinear', align_corners=True) 
              feature2d = F.interpolate(feature2d, size=IMAGE_SIZE, mode='bilinear', align_corners=True)
              
              # input the predicted 2D features to the 3D model alongside with depth data
              outputs = model_eval(rgb, inputs,feature2d, depth_3d, device)
              
              label_2d = label_2d.squeeze(1).long() # remove the addiyional dim
              test_loss2d =criterion(semantics2d, label_2d)
              test_loss = loss_function(outputs, targets, weights) + test_loss2d # combine 3d loss and 2d loss
              
              test_IoU, test_prec, test_recall = comp_IoU(outputs, targets, masks)
              test_mIoU, seg_lis = m_IoU(outputs, targets)
              print(f"IoU: {test_IoU} | precision: {test_prec} | recall: {test_recall} | mIoU: {test_mIoU}")
              print(f"*************************************************")
              total_loss += test_loss
              total_IoU += test_IoU
              total_prec += test_prec
              total_recall += test_recall
              
              ######################
              ceil_ls.append(seg_lis[0])
              floor_ls.append( seg_lis[1])
              wall_ls.append(seg_lis[2])
              wind_ls.append(seg_lis[3])
              chair_ls.append(seg_lis[4])
              bed_ls.append(seg_lis[5])
              sofa_ls.append(seg_lis[6])
              table_ls.append(seg_lis[7])
              tvs_ls.append(seg_lis[8])
              furn_ls.append(seg_lis[9])
              objs_ls.append(seg_lis[10])
              #######################
          total_loss /= len(test_loader)
          total_IoU /= len(test_loader)
          total_prec /= len(test_loader)
          total_recall /= len(test_loader)
          
          #############################
          # get avg for each class in NYU dataset
          ceil_ls = sum(ceil_ls)/73
          floor_ls = sum(floor_ls)/639
          wall_ls = sum(wall_ls)/606
          wind_ls = sum(wind_ls)/146
          chair_ls = sum(chair_ls)/267
          bed_ls = sum(bed_ls)/152
          sofa_ls = sum(sofa_ls)/138
          table_ls = sum (table_ls)/296
          tvs_ls = sum(tvs_ls)/37
          furn_ls = sum(furn_ls)/529
          objs_ls = sum(objs_ls)/630
          total_mIoU = (ceil_ls+floor_ls+wall_ls+wind_ls+chair_ls+bed_ls+sofa_ls+table_ls+tvs_ls+furn_ls+objs_ls ) /11
          ###########################
          print(f"Average scores for fold-{fold_num}:")
          print(f"AVG loss: {total_loss} | AVG IoU: {total_IoU} | AVG precision: {total_prec} | AVG recall: {total_recall} | AVG mIoU: {total_mIoU}")
          # log metrics to wandb
          wandb.log({"AVG test Loss": total_loss, "AVG test IoU":total_IoU, "AVG test precision": total_prec, "AVG test recall":total_recall, "AVG test mIoU": total_mIoU})
          wandb.log({"AVG ceil": ceil_ls, "AVG floor":floor_ls, "AVG wall": wall_ls, "AVG wind":wind_ls, "AVG chair": chair_ls, "AVG bed": bed_ls, "AVG sofa" : sofa_ls, "AVG table": table_ls, "AVG tvs":tvs_ls, "AVG furn": furn_ls ,"AVG objs": objs_ls})
    
    
    # Close this run for wandb
    run.finish()
    fold_num+=1
    print(f"----------------------------------------------------------------------------------------------------------------------------------------------------------------")

# Main Function
def main():

    parse_arguments()
    model_evalauteion()
    
if __name__ == '__main__':
  main()        

