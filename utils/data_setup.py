
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import glob
import cv2
from PIL import Image
import random
from skimage import io
######################################################################################################
######################################################################################################
class SSCData(Dataset):
    def __init__(self, file_prefixes, train= True):

        self.file_prefixes = file_prefixes
        self.train = train
        
    def __len__(self):
        return len(self.file_prefixes)

    def __getitem__(self, index):
        file_prefix = self.file_prefixes[index]
        npz_file = file_prefix + '.npz'
        loaded = np.load(npz_file)

        # turn the 3D data to pytorch tensor format
        x_batch = torch.from_numpy(loaded['tsdf']).reshape( 1, 240, 144, 240)
        y_batch = torch.from_numpy(loaded['lbl']).long()
        y_batch = one_hot(y_batch, num_classes=12).float().permute(3, 0, 1, 2).float()
        w_batch = torch.from_numpy(loaded['weights']).float()
        m_batch = torch.from_numpy(loaded['masks']).float()
        d_batch = torch.from_numpy(loaded['mapping'])
        
        rgb_path = loaded['rgb'].item()
        rgb_path = rgb_path + '.png'
        rgb = io.imread(rgb_path)
        
        lbl2d_path = loaded['label2d'].item()
        lbl2d_path = lbl2d_path + '.png'
        lbl2d = io.imread(lbl2d_path)
        
        #########################################################
        
        # convert to tensor
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255)
        lbl2d = torch.from_numpy(lbl2d.astype(int))
        
        sample = {
            'vox_tsdf': x_batch,
            'vox_lbl': y_batch,
            'vox_weight': w_batch,
            'vox_mask':  m_batch,
            'vox_depth': d_batch,
            'rgb': rgb,
            'label_2d': lbl2d
        }

        return sample


