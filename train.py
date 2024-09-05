
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
import argparse
from torch.optim.lr_scheduler import OneCycleLR
import random
from timeit import default_timer as timer 
# Import tqdm for progress bar
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.data_setup import SSCData 
from utils.file_utils import get_all_preprocessed_prefixes 
from utils.engin import EarlyStopping, create_writer,save_model, print_train_time 
from model.mdbnet import get_res_unet_rgb, get_2Dfeatures, get_activation
#from model.network_2 import get_res_unet_rgb, get_2Dfeatures, get_activation
#from model.network_2_mid import get_res_unet, get_res_unet_rgb, get_2Dfeatures, get_activation 
from model.losses import WeightedCrossEntropyLoss, WCE_k3Clusters
from model.metrics import comp_IoU, m_IoU
from torchsummary import summary
from utils.cosine_lr import CosineLRScheduler 
from utils.optim import AdamW 


###### set seeds ###################################################
SEED_VAL = 1234
np.random.seed(SEED_VAL)
random.seed(SEED_VAL)
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)
torch.cuda.random.manual_seed_all(SEED_VAL)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False

    
def worker_init_fn(worker_id): 
    np.random.seed(np.random.get_state()[1][0] + worker_id)
####################################################################
########################## wandb login #############################
import wandb
os.environ["WANDB_API_KEY"] = 'Add your key'
os.environ["WANDB_MODE"] = "offline"
######################################################################

MODEL_NAME ='ResUNet'
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 2
BASE_LR_2D= 0.0001
BASE_LR_3D = 0.01
DECAY_2D = 0.05
DECAY_3D=0.0005
EPOCHS = 100
#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 6 # 2 to eliminate DataLoader running slow or even freeze
FOLD_NUM = 1
IMAGE_SIZE = (480, 640)
PRE_TRAINED_MODEL = "./pretrained_segformer_b5_ade-640-640"
FUSION_LOC = 'late'
######################################################################

def parse_arguments():
    global DATASET, EXPR_NAME, MODEL_NAME, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, BASE_LR_2D,BASE_LR_3D, DECAY_2D, DECAY_3D, EPOCHS, PREPROC_PATH,PRE_TRAINED_MODEL, FUSION_LOC
    print("\n Training Script\n")

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", help="Target dataset", type=str, choices=['NYU', 'NYUCAD'])
    parser.add_argument("--model_name", help="Model name", type=str, choices=['ResUNet', 'ResUNet_rgb'])
    parser.add_argument("--fusion", help="fusion location", type=str, choices=['early', 'mid', 'late'])
    parser.add_argument("--expr_name", help="Experiment name", type=str)
    parser.add_argument("--train_batch_size", help="Trainig batch size. Default: "+str(TRAIN_BATCH_SIZE),type=int, default=TRAIN_BATCH_SIZE, required=False)
    parser.add_argument("--val_batch_size",  help="Val batch size. Default: "+str(VAL_BATCH_SIZE),type=int, default=VAL_BATCH_SIZE, required=False)
    parser.add_argument("--base_lr_2d", help="Base LR for 2D input. Default " + str(BASE_LR_2D),type=float, default=BASE_LR_2D, required=False)
    parser.add_argument("--base_lr_3d", help="Base LR for 3D input. Default " + str(BASE_LR_3D),type=float, default=BASE_LR_3D, required=False)
    parser.add_argument("--decay_2DModel", help="Weight decay. Default: " + str(DECAY_2D),type=float, default=DECAY_2D, required=False)
    parser.add_argument("--decay_3DModel", help="Weight decay. Default: " + str(DECAY_3D),type=float, default=DECAY_3D, required=False)
    parser.add_argument("--epochs", help="How many epochs? Default: " + str(EPOCHS),type=int, default=EPOCHS, required=False)
    #parser.add_argument("--pretrained_model", help="Pre-trained model path:", "Default: " + str(PRE_TRAINED_MODEL), required=False)
    args = parser.parse_args()


    ####################################################
    DATASET = args.dataset
    EXPR_NAME = args.expr_name
    MODEL_NAME = args.model_name
    FUSION_LOC = args.fusion
    EPOCHS= args.epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    BASE_LR_2D= args.base_lr_2d
    BASE_LR_3D= args.base_lr_3d
    DECAY_2D = args.decay_2DModel
    DECAY_3D = args.decay_3DModel
    #PRE_TRAINED_MODEL = args.pretrained_model
    
    if(DATASET == 'NYU')or (DATASET == 'nyu'):  # add these paths to file_prefixes_train and file_prefixes_val
      PREPROC_PATH = './data/NYU_train_preproc/'
    elif (DATASET == 'NYUCAD') or (DATASET == 'nyucad'):
      PREPROC_PATH = './data/NYUCAD_train_preproc/'
    ###################################################
   

def train():
  global FOLD_NUM  
    

  ######################################################
  print("Model: "+ MODEL_NAME+"_"+FUSION_LOC+"_"+EXPR_NAME)
  print("Dataset path: "+ PREPROC_PATH)
  print(f"BASE_LR_3D:" + str(BASE_LR_3D) + "  BASE_LR_2D:" + str(BASE_LR_2D)) 
  # get file prefixes
  file_prefixes = get_all_preprocessed_prefixes(PREPROC_PATH, criteria='*.npz')
  df = pd.DataFrame(data={"filename": file_prefixes})
  
  
  ####################################################################################################################
  kf = KFold(n_splits=3, random_state=SEED_VAL, shuffle=True)
  
  for fold, (train_ids, val_ids) in enumerate(kf.split(df)):
      
      train_df= df.iloc[train_ids]
      val_df= df.iloc[val_ids]
      train_ds = SSCData(train_df['filename'].tolist(),train = True)
      val_ds = SSCData(val_df['filename'].tolist(), train = False)
      
      
      print(f'Items in train_dataset ',len(train_ds), ' |  Items in val_dataset ',len(val_ds))
  
      train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS,worker_init_fn=worker_init_fn, pin_memory=True) 
      val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS,worker_init_fn=worker_init_fn, pin_memory=True)
 
      print(f"Length of train dataloader: {len(train_loader)} batches ({len(train_ds)}/{TRAIN_BATCH_SIZE})")
      print(f"Length of val dataloader: {len(val_loader)} batches ({len(val_ds)}/{VAL_BATCH_SIZE})")
  
      # model initialization for each fold
      model2D, img_processor = get_2Dfeatures(PRE_TRAINED_MODEL)  
      model = get_res_unet_rgb(FUSION_LOC) # 3D model
            
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
      # If there are multiple GPUs, wrap the model with nn.DataParallel
      if torch.cuda.device_count() > 1:
          print("Using", torch.cuda.device_count(), "GPUs!")
          print ('os.environ["CUDA_VISIBLE_DEVICES"] ',os.environ["CUDA_VISIBLE_DEVICES"])
          model = nn.DataParallel(model, device_ids= list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(','))) )
          model2D = nn.DataParallel(model2D, device_ids= list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(','))) )
      
      #move 3D model to device
      model.to(device) 
      model2D.to(device)
      ############################################################################################
      ############################################################################################
      # define loss
      criterion = nn.CrossEntropyLoss(label_smoothing=0.2) # for 2d loss
      loss_function = WCE_k3Clusters(device)# k=3
     
      print("3D Loss function: "+ str(loss_function) + "2D Loss function: "+ str(criterion))
      print("-------------------------------------------------------------")
      ############################################################################################
      ############################################################################################
      
      # define the optimizer for model
      optimizer_3d = optim.SGD(model.parameters(), lr=BASE_LR_3D, weight_decay=DECAY_3D, momentum=0.9)
      optimizer_2d = AdamW(filter(lambda p: p.requires_grad, model2D.parameters()), lr=BASE_LR_2D, weight_decay=DECAY_2D)
     
      # Set up the scheduler
      scheduler_3d = OneCycleLR(optimizer_3d, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=EPOCHS)
      scheduler_2d = CosineLRScheduler(optimizer_2d, t_initial=EPOCHS, lr_min=0.0000001, warmup_lr_init=1.0e-6, warmup_t=0, cycle_decay= 0.1 )
      
      # initialize the timer
      train_time_start = timer()
      
      # initialize wandb for each fold
      run = wandb.init(project=f'{MODEL_NAME}_{FUSION_LOC}_{EXPR_NAME}', group='k-fold', name=f'fold-{FOLD_NUM}', reinit=True)
      
      # initialize the early_stopping object
      early_stopping = EarlyStopping(patience=15, verbose=True)
      
      print(f"Processing FOLD NUM: {FOLD_NUM}")
      print(f'Number of EPOCHS in this fold: {EPOCHS}') 
      
      for epoch in tqdm(range(EPOCHS)):
          np.random.seed(NUM_WORKERS*epoch)
          ### Training
          model2D.train()
          model.train()
          
          train_loss, train_IoU, train_prec, train_recall, train_mIoU = 0,0,0,0,0
          
          ####################
         
          # Add a loop to loop through training batches
          for batch_num, sample in enumerate(train_loader):
              loss, t_IoU, t_prec, t_recall, t_mIoU = 0,0,0,0,0
              loss2d = 0
              loss3d = 0
              seg_list=[]
              
              rgb = sample['rgb'].to(device)
              label_2d = sample['label_2d'].to(device)
              inputs = sample['vox_tsdf'].to(device) # F-TSDF
              targets = sample['vox_lbl'].to(device)
              weights = sample['vox_weight'].to(device)
              masks = sample['vox_mask'].to(device)
              depth_3d = sample['vox_depth'].to(device)
              
              
              # 1. forward pass
              ##########################
              
              # get the 2d feature maps from pretrained model
              imgs = img_processor(images=rgb, return_tensors="pt").to(device)
                
              hook,activation = get_activation('linear_fuse')
              model2D.decode_head.register_forward_hook(hook)
              cls = model2D(**imgs) # predected 2d classes
              classes_logits = cls.logits # logits from classification layer
              feature2d = activation['linear_fuse'] # the activation maps befor the classification layer
                
              # upsample the output to match NYU images size
              semantics2d = F.interpolate(classes_logits, size=IMAGE_SIZE, mode='bilinear', align_corners=True) 
              feature2d = F.interpolate(feature2d, size=IMAGE_SIZE, mode='bilinear', align_corners=True)
                
              # input the predicted 2D features to the 3D model alongside with depth data
              outputs = model(rgb, inputs,feature2d, depth_3d, device)
              ########################### 
              
              # 2. calculate loss (per batch)
              label_2d = label_2d.squeeze(1).long() # remove the addiyional dim
              loss2d =criterion(semantics2d, label_2d)
              #################
              loss3d = loss_function(outputs, targets, weights)
              
              loss = loss2d + loss3d
              train_loss += loss # accumulatively add up the loss  
              
              # calculate IoU and mIoU
              t_IoU, t_prec, t_recall = comp_IoU(outputs, targets, masks)
              t_mIoU,seg_list = m_IoU(outputs, targets)
              train_IoU += t_IoU
              train_prec += t_prec
              train_recall += t_recall
              train_mIoU += t_mIoU 
              
              #######################
              
              # 3. optimizer zero grad
              optimizer_2d.zero_grad()
              optimizer_3d.zero_grad()
              
              # 4. loss backward
              loss.backward()
  
              # 5. optimizer step
              optimizer_2d.step()
              optimizer_3d.step()
              
              # step the scheduler
              scheduler_3d.step()
          
          
          scheduler_2d.step(epoch)
              
          # divide total train loss, train_IoU, and  train mIoU by length of train dataloader
          train_loss /= len(train_loader)
          train_IoU /= len(train_loader) 
          train_prec /= len(train_loader) 
          train_recall /= len(train_loader) 
          train_mIoU /= len(train_loader) 
          ###########################
        ############################## validation #######################################    
          model2D.eval()
          model.eval()
          #model_2D.eval()
          with torch.inference_mode():
              val_loss, val_IoU, val_prec, val_recall, val_mIoU = 0,0,0,0,0
              
              ####################
              for batch_num, sample in enumerate(val_loader):
                  v_IoU, v_prec, v_recall, v_mIoU = 0,0,0,0
                  loss_v = 0
                  loss2d_v = 0
                  loss3d_v = 0
                  seg_lis =[]     
                  
                  
                  rgb = sample['rgb'].to(device)
                  label_2d = sample['label_2d'].to(device)
                  inputs = sample['vox_tsdf'].to(device)
                  targets = sample['vox_lbl'].to(device)
                  weights = sample['vox_weight'].to(device)
                  masks = sample['vox_mask'].to(device)
                  depth_3d = sample['vox_depth'].to(device)
                  
                  ##########################
                  
                  # get the 2d feature maps from pretrained model
                  imgs = img_processor(images=rgb, return_tensors="pt").to(device)
                    
                  hook,activation = get_activation('linear_fuse')
                  model2D.decode_head.register_forward_hook(hook)
                    
                  cls = model2D(**imgs) # predected 2d classes
                  classes_logits = cls.logits # logits from classification layer
                  feature2d = activation['linear_fuse'] # the activation maps befor the classification layer
                    
                  # upsample the output to match NYU images size
                  semantics2d = F.interpolate(classes_logits, size=IMAGE_SIZE, mode='bilinear', align_corners=True) 
                  feature2d = F.interpolate(feature2d, size=IMAGE_SIZE, mode='bilinear', align_corners=True)
                    
                  # input the predicted 2D features to the 3D model alongside with depth data
                  outputs = model(rgb, inputs,feature2d, depth_3d, device)
                 
                  ###########################
                  label_2d = label_2d.squeeze(1).long() # remove the addiyional dim
                  loss2d_v = criterion(semantics2d, label_2d)
                  loss3d_v = loss_function(outputs, targets, weights) 
                  loss_v = loss2d_v  + loss3d_v
                  val_loss += loss_v
                  
                  v_IoU, v_prec, v_recall = comp_IoU(outputs, targets, masks)
                  v_mIoU, seg_lis = m_IoU(outputs, targets)
                   
                  # accumulate the values over patches
                  val_IoU += v_IoU
                  val_prec += v_prec
                  val_recall += v_recall
                  val_mIoU += v_mIoU
                  ######################
              
              # Divide total val loss, val_IoU, val_mIoU by length of val dataloader
              val_loss /= len(val_loader)
              val_IoU /= len(val_loader)
              val_prec /= len(val_loader)
              val_recall /= len(val_loader)
              val_mIoU /= len(val_loader)
              #############################
              expr_name_fold = FUSION_LOC +"_"+ EXPR_NAME + "_fold"+ str(FOLD_NUM)
              
              # Early stopping
              early_stopping(model, MODEL_NAME, expr_name_fold, val_loss, model2D)
              
              if early_stopping.early_stop:
                print("Early stopping")
                expr_name_fold = " "
                break
              
          print(f"Epoch: {epoch}")
          print(f"Loss 2d: {loss2d:.5f} | Loss 3d: {loss3d:.5f} | combined train Loss: {train_loss:.5f} | train IoU: {train_IoU:.2f} |train precision: {train_prec:.2f} | train recall: {train_recall:.2f} | train mIoU: {train_mIoU:.2f}")
          print(f"Loss 2d_val: {loss2d_v:.5f} | Loss 3d_val: {loss3d_v:.5f} | combined val Loss: {val_loss:.5f} | val IoU: {val_IoU:.2f} |val precision: {val_prec:.2f} | val recall: {val_recall:.2f} | val mIoU: {val_mIoU:.2f}")
          print(f"................................................................................................................................................................")
          
          ###########################################################################################################
          # log metrics to wandb
          wandb.log({"Loss2d": loss2d,"Loss3d": loss3d,"compined train Loss": train_loss, "train IoU":train_IoU, "train precision": train_prec, "train recall":train_recall, "train mIoU": train_mIoU, "Loss2d_val": loss2d_v,"Loss3d_val": loss3d_v,"compined val Loss": val_loss, "val IoU":val_IoU, "val precision": val_prec, "val recall":val_recall, "val mIoU": val_mIoU },step=epoch )
        ##############################################################################################################
        
      # Calculate training time for each fold      
      train_time_end = timer()
      total_train_time_model_0 = print_train_time(start=train_time_start, 
                                                end=train_time_end,
                                                device=str(next(model.parameters()).device))
      
      ##########################################
      # Close this run for wandb
      run.finish()
      FOLD_NUM+=1
      print(f"----------------------------------------------------------------------------------------------------------------------------------------------------------------")

# Main Function
def main():
    parse_arguments()
    train()
    
if __name__ == '__main__':
  main()
