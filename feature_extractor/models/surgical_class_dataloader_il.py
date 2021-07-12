'''
Project         : Learning Domain Generaliazation with Graph Neural Network for Surgical Scene Understanding.
Lab             : MMLAB, National University of Singapore
contributors    : Lalith, Mobarak
Note            : Dataloader for incremental learning for feature extraction
'''

import os
import sys
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

class SurgicalClassDataset18_incremental(Dataset):
    def __init__(self, filenames, fine_tune_size = None, is_train=None):
        
        self.is_train = is_train
        self.img_list = []
        
        # Using readlines() 
        for i, txt_file in enumerate(filenames):
            curr_file = open((txt_file), 'r') 
            Lines = curr_file.readlines()  
            if (fine_tune_size is not None) and (i == len(filenames)-1):
                indices = np.random.permutation(len(Lines))
                Lines = [Lines[i] for i in indices[0:fine_tune_size]]
            for line in Lines: self.img_list.append(line. rstrip())
            #print(self.img_list)
            curr_file.close()
        
    def __len__(self): return len(self.img_list)

    def __getitem__(self, index):
        _img_dir = self.img_list[index]
        #print(_img_dir)
        _img = Image.open(_img_dir).convert('RGB')
        _target = int(_img_dir[:-4].split('_')[-1:][0])
        _img = np.asarray(_img, np.float32) / 255
        _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1,)).float()
        _target = torch.from_numpy(np.array(_target)).long()
        return _img, _target
