import os
from fnmatch import fnmatch
import re
  


def get_file_prefixes_from_path(data_path):
    depth_prefixes = {}
    rgb_prefixes = {}
    labels2D_prefixes = {}
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if fnmatch(name, "NYU*_0000.png"): # depth image
                key = name[3:7]
                depth_prefixes[key] = os.path.join(path, name[:-4]) # removes the ".png" from the end of the filename
            
            if fnmatch(name, "NYU*_colors.png"): # raw rgb image
                key = name[3:7]
                rgb_prefixes[key] = os.path.join(path, name[:-4]) # removes the ".png" from the end of the filename
             
            if fnmatch(name, "NYU*_0000_labels.png"): # 2D segmentation map within gray scale 
                key = name[3:7]
                labels2D_prefixes[key] = os.path.join(path, name[:-4]) # removes the ".png" from the end of the filename
    return depth_prefixes, rgb_prefixes, labels2D_prefixes  

#########################
def get_all_preprocessed_prefixes(data_path, criteria="*.npz"):
    prefixes = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if fnmatch(name, criteria):
                prefixes.append(os.path.join(path, name)[:-4])# removes the last 4 characters from the string (".npz" for a matching file)
                
    return prefixes