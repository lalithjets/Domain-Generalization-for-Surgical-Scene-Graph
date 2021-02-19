import sys
import random

import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
from glob import glob
    
class SurgicalSceneDataset(Dataset):
    '''
    '''
    def __init__(self, seq_set, dataconst, feature_extractor):
        self.dataconst = dataconst
        self.dir_root_gt = 'datasets/instruments18/seq_'
        self.feature_extractor = feature_extractor
        
        self.xml_dir_list = []
        for i in seq_set:
            xml_dir_temp = self.dir_root_gt + str(i) + '/xml/'
            self.xml_dir_list = self.xml_dir_list + glob(xml_dir_temp + '/*.xml')
        
        self.word2vec = h5py.File('datasets/surgicalscene_word2vec.hdf5', 'r')
    
    # word2vec
    def _get_word2vec(self,node_ids):
        word2vec = np.empty((0,300))
        for node_id in node_ids:
            vec = self.word2vec[self.dataconst.instrument_classses[node_id]]
            word2vec = np.vstack((word2vec, vec))
        return word2vec

    def __len__(self):
        return len(self.xml_dir_list)

    def __getitem__(self, idx):
    
        file_name = os.path.splitext(os.path.basename(self.xml_dir_list[idx]))[0]
        file_root = os.path.dirname(os.path.dirname(self.xml_dir_list[idx]))
        _img_loc = os.path.join(file_root+'/left_frames/'+ file_name + '.png')
        
        frame_data = h5py.File(os.path.join(file_root+'/vsgat/'+self.feature_extractor+'/'+ file_name + '_features.hdf5'), 'r')    
        data = {}
        data['img_name'] = frame_data['img_name'].value[:] + '.jpg'
        
        data['node_num'] = frame_data['node_num'].value
        data['roi_labels'] = frame_data['classes'][:]
        data['det_boxes'] = frame_data['boxes'][:]
        
        
        data['edge_labels'] = frame_data['edge_labels'][:]
        data['edge_num'] = data['edge_labels'].shape[0]
        
        data['features'] = frame_data['node_features'][:]
        data['spatial_feat'] = frame_data['spatial_features'][:]
        data['word2vec'] = self._get_word2vec(data['roi_labels'])
        return data

# for DatasetLoader
def collate_fn(batch):
    '''
        Default collate_fn(): https://github.com/pytorch/pytorch/blob/1d53d0756668ce641e4f109200d9c65b003d05fa/torch/utils/data/_utils/collate.py#L43
    '''
    batch_data = {}
    batch_data['img_name'] = []
    batch_data['node_num'] = []
    batch_data['roi_labels'] = []
    batch_data['det_boxes'] = []
    batch_data['edge_labels'] = []
    batch_data['edge_num'] = []
    batch_data['features'] = []
    batch_data['spatial_feat'] = []
    batch_data['word2vec'] = []
    
    for data in batch:
        batch_data['img_name'].append(data['img_name'])
        batch_data['node_num'].append(data['node_num'])
        batch_data['roi_labels'].append(data['roi_labels'])
        batch_data['det_boxes'].append(data['det_boxes'])
        batch_data['edge_labels'].append(data['edge_labels'])
        batch_data['edge_num'].append(data['edge_num'])
        batch_data['features'].append(data['features'])
        batch_data['spatial_feat'].append(data['spatial_feat'])
        batch_data['word2vec'].append(data['word2vec'])
        
    batch_data['edge_labels'] = torch.FloatTensor(np.concatenate(batch_data['edge_labels'], axis=0))
    batch_data['features'] = torch.FloatTensor(np.concatenate(batch_data['features'], axis=0))
    batch_data['spatial_feat'] = torch.FloatTensor(np.concatenate(batch_data['spatial_feat'], axis=0))
    batch_data['word2vec'] = torch.FloatTensor(np.concatenate(batch_data['word2vec'], axis=0))
    
    return batch_data