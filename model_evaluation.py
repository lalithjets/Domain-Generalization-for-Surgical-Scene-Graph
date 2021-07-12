'''
Project         : Learning Domain Generaliazation with Graph Neural Network for Surgical Scene Understanding.
Lab             : MMLAB, National University of Singapore
contributors    : Lalith, Mobarak 
Note            : Code adopted and modified from Visual-Semantic Graph Attention Networks and end-to-end incremental learning.
                        Visual-Semantic Graph Network:
                        @article{liang2020visual,
                          title={Visual-Semantic Graph Attention Networks for Human-Object Interaction Detection},
                          author={Liang, Zhijun and Rojas, Juan and Liu, Junfa and Guan, Yisheng},
                          journal={arXiv preprint arXiv:2001.02302},
                          year={2020}
                        }
                        Incremental Learning:
                        @inproceedings{castro2018end,
                            title={End-to-end incremental learning},
                            author={Castro, Francisco M and Mar{\'\i}n-Jim{\'e}nez, Manuel J and Guil, Nicol{\'a}s and Schmid, Cordelia and Alahari, Karteek},
                            booktitle={Proceedings of the European conference on computer vision (ECCV)},
                            pages={233--248},
                            year={2018}
                        }
'''

from __future__ import print_function

import os
import copy
import time
import argparse

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils.io as io
from utils.vis_tool import *
from model.agrnn_network import *
from model.surgicalscenedataloader import *

import utils.io as io
from utils.evals import *


def seed_everything(seed=27):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_cls_freq(dataloader, num_classes):
    cls_freq = np.zeros((num_classes,1))
    for data in dataloader:
        edge_labels = data['edge_labels']
        edge_labels = np.argmax(edge_labels.cpu().data.numpy(), axis=-1)    
        for i in edge_labels:
            cls_freq[i] += 1
    return cls_freq


def evaluate(args, data_const, model, seq, device, dname, rdname, plot_name = 'graph',plot = False):
    '''

    '''
    train_dataset = SurgicalSceneDataset(seq_set = seq['train_seq'], data_dir = seq['data_dir'], \
                            img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                            feature_extractor = args.feature_extractor, reduce_size = False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle= True, \
                            collate_fn=collate_fn)
    
    val_dataset = SurgicalSceneDataset(seq_set = seq['val_seq'], data_dir = seq['data_dir'], \
                            img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                            feature_extractor = args.feature_extractor, reduce_size = False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= 1, shuffle= True, \
                            collate_fn=collate_fn)
    
    # model evaluate
    model.eval()
    
    # criterion and scheduler
    criterion = nn.MultiLabelSoftMarginLoss()

    # each epoch has a training and validation step                   
    edge_count = 0
    total_acc = 0.0
    total_loss = 0.0
    logits_list = []
    labels_list = []
    start_time = time.time()
    
    if args.use_cda_t:
        cls_freq = calculate_cls_freq(train_dataloader, len(data_const.action_classes))
        cls_freq_log = torch.tensor(cls_freq).log()
        cls_freq_log_norm = cls_freq_log/cls_freq_log.max()
        temp = args.t_scale - (1-cls_freq_log_norm.view(-1))
        #print(temp)
        temp = temp.to(device)
    
    for data in val_dataloader:
        train_data = data
        img_name = train_data['img_name']
        img_loc = train_data['img_loc']
        node_num = train_data['node_num']
        roi_labels = train_data['roi_labels']
        det_boxes = train_data['det_boxes']
        edge_labels = train_data['edge_labels']
        edge_num = train_data['edge_num']
        features = train_data['features']
        spatial_feat = train_data['spatial_feat']
        word2vec = train_data['word2vec']
        features, spatial_feat, word2vec, edge_labels = features.to(device), spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)    
            
        #if img_name[0] == 'frame123.jpg':
            #print(edge_labels)
        with torch.no_grad():
            outputs = model(node_num, features, spatial_feat, word2vec, roi_labels, validation=True)
            
            if args.use_t: outputs/args.t_scale
            elif args.use_cda_t: outputs = outputs/temp

            logits_list.append(outputs)
            labels_list.append(edge_labels)       
            
            # loss and accuracy
            loss = criterion(outputs, edge_labels.float())
            acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
            
            #if img_name[0] == 'frame123.jpg' or img_name[0] == 'frame0450.jpg':
            #    print(np.argmax(edge_labels.cpu().data.numpy(), axis=-1))
            #    print(np.argmax(outputs.cpu().data.numpy(), axis=-1))
        # accumulate loss and accuracy of the batch
        total_loss += loss.item() * edge_labels.shape[0]
        total_acc  += acc
        edge_count += edge_labels.shape[0]
    
    logits_all = torch.cat(logits_list).cuda()
    labels_all = torch.cat(labels_list).cuda()
    
    # calculate the loss and accuracy
    total_acc = total_acc / edge_count
    total_loss = total_loss / len(val_dataloader)
    
    end_time = time.time()
    
    logits_all = F.softmax(logits_all, dim=1)
    map_value, recall, ece, sce, tace, brier, uce = calibration_metrics(logits_all, labels_all, rdname, plot=plot, model_name=plot_name)
    print('acc: %0.6f, map: %0.6f, recall: %0.6f, loss: %0.6f, ece:%0.6f, sce:%0.6f, tace:%0.6f, brier:%.6f, uce:%.6f' %(total_acc, map_value, recall, total_loss, ece, sce, tace, brier, uce.item()) )


if __name__ == "__main__":

    # Version and feature extraction
    ver = 'N_d2g_ecbs_t_resnet18_11_cbs_ts'
    f_e = 'resnet18_11_cbs_ts'

    parser = argparse.ArgumentParser(description='domain_generalization in scene understanding')
    # Hyperparams
    parser.add_argument('--lr',                type=float,   default=0.00001,                     help='0.00001')
    parser.add_argument('--epoch',             type=int,     default=251,                         help='251')
    parser.add_argument('--ft_epoch',          type=int,     default=81,                          help='81')
    parser.add_argument('--start_epoch',       type=int,     default=0,                           help='0')
    parser.add_argument('--batch_size',        type=int,     default=32,                          help='32')
    parser.add_argument('--train_model',       type=str,     default='epoch',                     help='epoch')
    # network
    parser.add_argument('--layers',            type=int,     default = 1,                         help='1') 
    parser.add_argument('--bn',                type=bool,    default = False,                     help='pass empty string for false') 
    parser.add_argument('--drop_prob',         type=float,   default = 0.3,                       help='0.3') 
    parser.add_argument('--bias',              type=bool,    default = True,                      help='pass empty string for false') 
    parser.add_argument('--multi_attn',        type=bool,    default = False,                     help='pass empty string for false') 
    parser.add_argument('--diff_edge',         type=bool,    default = False,                     help='pass empty string for false') 
    # CBS
    parser.add_argument('--use_cbs',           type=bool,     default = True,                     help='pass empty string for false')
    # temperature_scaling
    parser.add_argument('--use_t',             type=bool,     default = True,                     help='pass empty string for false')
    parser.add_argument('--t_scale',           type=float,     default = 1.5,                     help='1.5')
    # optimizer
    parser.add_argument('--optim',             type=str,     default ='adam',                     help='sgd / adam')
    # GPU
    parser.add_argument('--gpu',               type=bool,    default=True,                        help='pass empty string for false')
    # file locations
    parser.add_argument('--exp_ver',           type=str,     default=ver,                         help='version_name')
    parser.add_argument('--log_dir',           type=str,     default = './log/' + ver,            help='log_dir')
    parser.add_argument('--save_dir',          type=str,     default = './checkpoints/' + ver,    help='save_dir')
    parser.add_argument('--output_img_dir',    type=str,     default = './results/' + ver,        help='epoch')
    parser.add_argument('--save_every',        type=int,     default = 10,                        help='10')
    parser.add_argument('--pretrained',        type=str,     default = None,                      help='pretrained_loc')
    # data_processing
    parser.add_argument('--sampler',           type=int,      default = 0,                        help='0')
    parser.add_argument('--data_aug',          type=bool,     default = False,                    help='pass empty string for false')
    parser.add_argument('--feature_extractor', type=str,      default = f_e,                      help='feature_extractor')
    # print every
    parser.add_argument('--print_every',       type=int,     default=10,                          help='10')
    args = parser.parse_args()

    
    seed_everything()
    data_const = SurgicalSceneConstants()
    
    for domain in args.testset:
        # val dataset
        if domain == 1:
            train_seq = [[2,3,4,6,7,9,10,11,12,14,15]]
            val_seq = [[1,5,16]]
            data_dir = ['datasets/instruments18/seq_']
            img_dir = ['/left_frames/']
            dset = [0]
            seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 'img_dir':img_dir, 'dset': dset}

        elif domain == 2:
            train_seq = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
            val_seq = [[16,17,18,19,20,21,22]]
            data_dir = ['datasets/SGH_dataset_2020/']
            img_dir = ['/resized_frames/']
            dset = [1]
            seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 'img_dir':img_dir, 'dset': dset}

        elif domain == 12:
            train_seq = [[2,3,4,6,7,9,10,11,12,14,15], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
            val_seq = [[1,5,16],[16,17,18,19,20,21,22]]
            #val_seq = [[5],[22]]
            data_dir = ['datasets/instruments18/seq_', 'datasets/SGH_dataset_2020/']
            img_dir = ['/left_frames/', '/resized_frames/']
            dset = [0, 1]
            seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 'img_dir':img_dir, 'dset': dset}

        # model
        model = AGRNN(bias=args.bias, bn=args.bn, dropout=args.drop_prob, multi_attn=args.multi_attn, layer=args.layers, diff_edge=args.diff_edge, use_cbs = args.use_cbs)
        if args.use_cbs: model.grnn1.gnn.apply_h_h_edge.get_new_kernels(0)

#         for i in [150,160,170,180,190,200,210,220,230,240,250]:
        # for i in [190]: # 190,190,190,220,190,220
#         for i in [20,30,40,50,60,70,80]:
        for i in [80]: #50,20,40,80
            pretrain_model = 'checkpoints/'+ver+'/'+ver+'/'+'epoch_train/checkpoint_D2F'+str(i)+'_epoch.pth'
            # pretrain_model = 'checkpoints/'+ver+'/'+ver+'/'+'epoch_train/checkpoint_D1'+str(i)+'_epoch.pth'
#             pretrain_model = 'checkpoints/'+ver+'/'+ver+'/'+'epoch_train/checkpoint_D2'+str(i)+'_epoch.pth'
            checkpoints = torch.load(pretrain_model)
            model.load_state_dict(checkpoints['state_dict'])

            # use cpu or cuda
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            print(i)
            evaluate(args, data_const, model,seq, device, "D1", str(i),plot=False)  

