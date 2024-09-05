import os
import torch
from torch.utils.tensorboard import SummaryWriter 
from datetime import datetime 
from pathlib import Path 
import numpy as np
import wandb


"""
Early stopping by:
https://github.com/Bjarten/early-stopping-pytorch/tree/master
"""
class EarlyStopping:
    
    def __init__(self, patience=10, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
                            
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    
    def __call__(self, model, model_name, expr_name, val_loss, model2D =None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, model_name, expr_name, val_loss, model2D)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, model_name, expr_name, val_loss, model2D)
            self.counter = 0
            self.early_stop = False  # reset here
            
    def save_checkpoint(self, model, model_name, expr_name, val_loss, model2D):
        # saves model when validation loss decrease.
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)
        save_model(model, model_name, expr_name, model2D)
        self.val_loss_min = val_loss

################################# Model saving #######################################

def save_model(model, model_name, expr_name, model2D = None):
  # create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
  model_path = Path("saved_models")
  model_path.mkdir(parents=True, # create parent directories if needed
                   exist_ok=True # if models directory already exists, don't error
  )

  # create model save path
  timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format
  full_model_name = model_name +"_"+ expr_name + "_" + timestamp +".pth"
  model_save_path = model_path / full_model_name

  # save the model state dict
  print(f"Saving model to: {model_save_path}")
  
  # save the 3D model in saved_models dir
  torch.save(obj=model.state_dict(), # only saving the state_dict() for the learned parameters
             f=model_save_path)
  # save model in wandb dir
  torch.save(model.state_dict(), os.path.join(wandb.run.dir, full_model_name))
  
  if model2D != None:
    full_model_name =  "2D_pretrained_"+ expr_name + "_" + timestamp +".pth"
    model_save_path = model_path / full_model_name
    
    # save pretrained model weights model in saved_models dir
    torch.save(obj=model2D.state_dict(), # only saving the state_dict() for the learned parameters
               f=model_save_path)
    # save pretrained in wandb dir
    torch.save(model2D.state_dict(), os.path.join(wandb.run.dir, full_model_name))
    
  
  loaded_model_size = Path(model_save_path).stat().st_size // (1024*1024)
  print(f"{model_name}_{expr_name} model size: {loaded_model_size} MB")
  
      
                    
##################################### train time cal ##############################
def print_train_time(start: float, end: float, device: torch.device = None):
    
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time    
############################## trainig summary #####################################

def create_writer(experiment_name: str, 
                  model_name: str,
                  fold_num = None) -> torch.utils.tensorboard.writer.SummaryWriter():
    
    # get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if fold_num:
        # create log directory path
        fold_num = "fold" + str(fold_num)
        log_dir = os.path.join("runs", timestamp, model_name, experiment_name, fold_num)
    else:
        log_dir = os.path.join("runs", timestamp, model_name, experiment_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
####################################################################################

