'''
H_H_EdgeApplyModule
    init    : config, multi_attn 
    forward : edge
    
H_NodeApplyModule
    init    : config
    forward : node
    
E_AttentionModule1
    init    : config
    forward : edge
    
GNN
    init    : config, multi_attn, diff_edge
    forward : g, h_node, o_node, h_h_e_list, o_o_e_list, h_o_e_list, pop_features
    
GRNN
    init    : config, multi_attn, diff_edge
    forward : b_graph, b_h_node_list, b_o_node_list, b_h_h_e_list, b_o_o_e_list, b_h_o_e_list, features, spatial_features, word2vec, valid, pop_features, initial_features
'''

import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import MLP

class H_H_EdgeApplyModule(nn.Module): #human to human edge
    '''
        init    : config, multi_attn 
        forward : edge
    '''
    def __init__(self, CONFIG, multi_attn=False):
        super(H_H_EdgeApplyModule, self).__init__()
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        self.edge_fc_lang = MLP(CONFIG.G_E_L_S2, CONFIG.G_E_A2, CONFIG.G_E_B2, CONFIG.G_E_BN2, CONFIG.G_E_D2)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.data['s_f'], edge.dst['n_f']], dim=1)
        feat_lang = torch.cat([edge.src['word2vec'], edge.dst['word2vec']], dim=1)
        e_feat = self.edge_fc(feat)
        e_feat_lang = self.edge_fc_lang(feat_lang)
  
        return {'e_f': e_feat, 'e_f_lang': e_feat_lang}

class H_NodeApplyModule(nn.Module): #human node
    '''
        init    : config
        forward : node
    '''
    def __init__(self, CONFIG):
        super(H_NodeApplyModule, self).__init__()
        self.node_fc = MLP(CONFIG.G_N_L_S, CONFIG.G_N_A, CONFIG.G_N_B, CONFIG.G_N_BN, CONFIG.G_N_D)
        self.node_fc_lang = MLP(CONFIG.G_N_L_S2, CONFIG.G_N_A2, CONFIG.G_N_B2, CONFIG.G_N_BN2, CONFIG.G_N_D2)
    
    def forward(self, node):
        # import ipdb; ipdb.set_trace()
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        feat_lang = torch.cat([node.data['word2vec'], node.data['z_f_lang']], dim=1)
        n_feat = self.node_fc(feat)
        n_feat_lang = self.node_fc_lang(feat_lang)

        return {'new_n_f': n_feat, 'new_n_f_lang': n_feat_lang}

class E_AttentionModule1(nn.Module): #edge attention
    '''
        init    : config
        forward : edge
    '''
    def __init__(self, CONFIG):
        super(E_AttentionModule1, self).__init__()
        self.attn_fc = MLP(CONFIG.G_A_L_S, CONFIG.G_A_A, CONFIG.G_A_B, CONFIG.G_A_BN, CONFIG.G_A_D)
        self.attn_fc_lang = MLP(CONFIG.G_A_L_S2, CONFIG.G_A_A2, CONFIG.G_A_B2, CONFIG.G_A_BN2, CONFIG.G_A_D2)

    def forward(self, edge):
        a_feat = self.attn_fc(edge.data['e_f'])
        a_feat_lang = self.attn_fc_lang(edge.data['e_f_lang'])
        return {'a_feat': a_feat, 'a_feat_lang': a_feat_lang}

class GNN(nn.Module):
    '''
        init    : config, multi_attn, diff_edge
        forward : g, h_node, o_node, h_h_e_list, o_o_e_list, h_o_e_list, pop_features
    '''
    def __init__(self, CONFIG, multi_attn=False, diff_edge=True):
        super(GNN, self).__init__()
        self.diff_edge = diff_edge # false
        self.apply_h_h_edge = H_H_EdgeApplyModule(CONFIG, multi_attn)
        self.apply_edge_attn1 = E_AttentionModule1(CONFIG)  
        self.apply_h_node = H_NodeApplyModule(CONFIG)

    def _message_func(self, edges):
        return {'nei_n_f': edges.src['n_f'], 'nei_n_w': edges.src['word2vec'], 'e_f': edges.data['e_f'], 'e_f_lang': edges.data['e_f_lang'], 'a_feat': edges.data['a_feat'], 'a_feat_lang': edges.data['a_feat_lang']}

    def _reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['a_feat'], dim=1)
        alpha_lang = F.softmax(nodes.mailbox['a_feat_lang'], dim=1)

        z_raw_f = nodes.mailbox['nei_n_f']+nodes.mailbox['e_f']
        z_f = torch.sum( alpha * z_raw_f, dim=1)

        z_raw_f_lang = nodes.mailbox['nei_n_w']
        z_f_lang = torch.sum(alpha_lang * z_raw_f_lang, dim=1)
         
        # we cannot return 'alpha' for the different dimension 
        if self.training or validation: return {'z_f': z_f, 'z_f_lang': z_f_lang}
        else: return {'z_f': z_f, 'z_f_lang': z_f_lang, 'alpha': alpha, 'alpha_lang': alpha_lang}

    def forward(self, g, h_node, o_node, h_h_e_list, o_o_e_list, h_o_e_list, pop_feat=False):
        
        g.apply_edges(self.apply_h_h_edge, g.edges())
        g.apply_edges(self.apply_edge_attn1)
        g.update_all(self._message_func, self._reduce_func)
        g.apply_nodes(self.apply_h_node, h_node+o_node)

        # !NOTE:PAY ATTENTION WHEN ADDING MORE FEATURE
        g.ndata.pop('n_f')
        g.ndata.pop('word2vec')

        g.ndata.pop('z_f')
        g.edata.pop('e_f')
        g.edata.pop('a_feat')

        g.ndata.pop('z_f_lang')
        g.edata.pop('e_f_lang')
        g.edata.pop('a_feat_lang')

class GRNN(nn.Module):
    '''
    init: 
        config, multi_attn, diff_edge
    forward: 
        batch_graph, batch_h_node_list, batch_obj_node_list,
        batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list,
        features, spatial_features, word2vec,
        valid, pop_features, initial_features
    '''
    def __init__(self, CONFIG, multi_attn=False, diff_edge=True):
        super(GRNN, self).__init__()
        self.multi_attn = multi_attn #false
        self.gnn = GNN(CONFIG, multi_attn, diff_edge)

    def forward(self, batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, valid=False, pop_feat=False, initial_feat=False):
        
        # !NOTE: if node_num==1, there will be something wrong to forward the attention mechanism
        global validation 
        validation = valid

        # initialize the graph with some datas
        batch_graph.ndata['n_f'] = feat           # node: features 
        batch_graph.ndata['word2vec'] = word2vec  # node: words
        batch_graph.edata['s_f'] = spatial_feat   # edge: spatial features

        try:
            self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list)
        except Exception as e:
            print(e)
            ipdb.set_trace()
        