'''
Predictor 
    init    : config
    forward : edge

AGRNN
    init    : bias, bn, dropout, multi_attn, layer, diff_edge
    forward : node_num, feat, spatial_feat, word2vec, roi_label, validation, choose_nodes, remove_nodes
'''

import dgl
import ipdb
import numpy as np

import torch
import torch.nn as nn

from model.grnn import GRNN
from model.modules import MLP
from model.config import CONFIGURATION

class Predictor(nn.Module):
    '''
    init    : config
    forward : edge
    '''
    def __init__(self, CONFIG):
        super(Predictor, self).__init__()
        self.classifier = MLP(CONFIG.G_ER_L_S, CONFIG.G_ER_A, CONFIG.G_ER_B, CONFIG.G_ER_BN, CONFIG.G_ER_D)
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge):
        feat = torch.cat([edge.dst['new_n_f'], edge.dst['new_n_f_lang'], edge.data['s_f'], edge.src['new_n_f_lang'], edge.src['new_n_f']], dim=1)
        pred = self.classifier(feat)
        # if the criterion is BCELoss, you need to uncomment the following code
        # output = self.sigmoid(output)
        return {'pred': pred}

class AGRNN(nn.Module):
    '''
    init    : 
        feature_type, bias, bn, dropout, multi_attn, layer, diff_edge
        
    forward : 
        node_num, features, spatial_features, word2vec, roi_label,
        validation, choose_nodes, remove_nodes
    '''
    def __init__(self, bias=True, bn=True, dropout=None, multi_attn=False, layer=1, diff_edge=True):
        super(AGRNN, self).__init__()
 
        self.multi_attn = multi_attn # false
        self.layer = layer           # 1 layer
        self.diff_edge = diff_edge   # false
        
        self.CONFIG1 = CONFIGURATION(layer=1, bias=bias, bn=bn, dropout=dropout, multi_attn=multi_attn)

        self.grnn1 = GRNN(self.CONFIG1, multi_attn=multi_attn, diff_edge=diff_edge)
        self.edge_readout = Predictor(self.CONFIG1)
        
    def _collect_edge(self, node_num, roi_label, node_space, diff_edge):
        '''
        arguments: node_num, roi_label, node_space, diff_edge
        '''
        
        # get human nodes && object nodes
        h_node_list = np.where(roi_label == 1)[0]
        obj_node_list = np.where(roi_label != 1)[0]
        edge_list = []
        
        h_h_e_list = []
        o_o_e_list = []
        h_o_e_list = []
        
        readout_edge_list = []
        readout_h_h_e_list = []
        readout_h_o_e_list = []
        
        # get all edge in the fully-connected graph, edge_list, For node_num = 2, edge_list = [(0, 1), (1, 0)]
        for src in range(node_num):
            for dst in range(node_num):
                if src == dst:
                    continue
                else:
                    edge_list.append((src, dst))
        
        # readout_edge_list, get corresponding readout edge in the graph
        src_box_list = np.arange(roi_label.shape[0])
        for dst in h_node_list:
            if dst == roi_label.shape[0]-1:
                continue
            src_box_list = src_box_list[1:]
            for src in src_box_list:
                readout_edge_list.append((src, dst))
        
        # readout h_h_e_list, get corresponding readout h_h edges && h_o edges
        temp_h_node_list = h_node_list[:]
        for dst in h_node_list:
            if dst == h_node_list.shape[0]-1:
                continue
            temp_h_node_list = temp_h_node_list[1:]
            for src in temp_h_node_list:
                if src == dst: continue
                readout_h_h_e_list.append((src, dst))

        # readout h_o_e_list
        readout_h_o_e_list = [x for x in readout_edge_list if x not in readout_h_h_e_list]

        # add node space to match the batch graph
        h_node_list = (np.array(h_node_list)+node_space).tolist()
        obj_node_list = (np.array(obj_node_list)+node_space).tolist()
        
        h_h_e_list = (np.array(h_h_e_list)+node_space).tolist() #empty no diff_edge
        o_o_e_list = (np.array(o_o_e_list)+node_space).tolist() #empty no diff_edge
        h_o_e_list = (np.array(h_o_e_list)+node_space).tolist() #empty no diff_edge

        readout_h_h_e_list = (np.array(readout_h_h_e_list)+node_space).tolist()
        readout_h_o_e_list = (np.array(readout_h_o_e_list)+node_space).tolist()   
        readout_edge_list = (np.array(readout_edge_list)+node_space).tolist()

        return edge_list, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list
    
    def _build_graph(self, node_num, roi_label, node_space, diff_edge):
        '''
        Declare graph, add_nodes, collect edges, add_edges
        '''
        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)

        edge_list, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list = self._collect_edge(node_num, roi_label, node_space, diff_edge)
        src, dst = tuple(zip(*edge_list))
        graph.add_edges(src, dst)   # make the graph bi-directional

        return graph, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list

    def forward(self, node_num=None, feat=None, spatial_feat=None, word2vec=None, roi_label=None, validation=False, choose_nodes=None, remove_nodes=None):
        
        batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, batch_readout_edge_list, batch_readout_h_h_e_list, batch_readout_h_o_e_list = [], [], [], [], [], [], [], [], []
        node_num_cum = np.cumsum(node_num) # !IMPORTANT
        
        for i in range(len(node_num)):
            # set node space
            node_space = 0
            if i != 0:
                node_space = node_num_cum[i-1]
            graph, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list = self._build_graph(node_num[i], roi_label[i], node_space, diff_edge=self.diff_edge)
            
            # updata batch
            batch_graph.append(graph)
            batch_h_node_list += h_node_list
            batch_obj_node_list += obj_node_list
            
            batch_h_h_e_list += h_h_e_list
            batch_o_o_e_list += o_o_e_list
            batch_h_o_e_list += h_o_e_list
            
            batch_readout_edge_list += readout_edge_list
            batch_readout_h_h_e_list += readout_h_h_e_list
            batch_readout_h_o_e_list += readout_h_o_e_list
        
        batch_graph = dgl.batch(batch_graph)
        
        # GRNN
        self.grnn1(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, validation, initial_feat=True)
        batch_graph.apply_edges(self.edge_readout, tuple(zip(*(batch_readout_h_o_e_list+batch_readout_h_h_e_list))))
        
        if self.training or validation:
            # !NOTE: cannot use "batch_readout_h_o_e_list+batch_readout_h_h_e_list" because of the wrong order
            return batch_graph.edges[tuple(zip(*batch_readout_edge_list))].data['pred']
        else:
            return batch_graph.edges[tuple(zip(*batch_readout_edge_list))].data['pred'], \
                   batch_graph.nodes[batch_h_node_list].data['alpha'], \
                   batch_graph.nodes[batch_h_node_list].data['alpha_lang'] 
