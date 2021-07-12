'''
Project         : Learning Domain Generaliazation with Graph Neural Network for Surgical Scene Understanding.
Lab             : MMLAB, National University of Singapore
contributors    : Lalith, Mobarak 
Note            : Code adopted and modified from Visual-Semantic Graph Attention Networks.
                        @article{liang2020visual,
                          title={Visual-Semantic Graph Attention Networks for Human-Object Interaction Detection},
                          author={Liang, Zhijun and Rojas, Juan and Liu, Junfa and Guan, Yisheng},
                          journal={arXiv preprint arXiv:2001.02302},
                          year={2020}
                        }
Contains        : Primary activation and MLP layer
                    acivation:
                        Identity
                        ReLU
                        LeakyReLU
                    MLP:
                        init: layer size, activation, bias, use_BN, dropout_probability
                        forward: x
'''

import torch.nn as nn
from collections import OrderedDict

class Identity(nn.Module):
    '''
    Identity class activation layer
    x = x
    '''
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self, x):
        return x

def get_activation(name):
    '''
    get_activation sub-function
    argument: activatoin name (eg. ReLU, Identity, LeakyReLU)
    '''
    if name=='ReLU': return nn.ReLU(inplace=True)
    elif name=='Identity': return Identity()
    elif name=='LeakyReLU': return nn.LeakyReLU(0.2,inplace=True)
    else: assert(False), 'Not Implemented'
    #elif name=='Tanh': return nn.Tanh()
    #elif name=='Sigmoid': return nn.Sigmoid()

class MLP(nn.Module):
    '''
    Args:
        layer_sizes: a list, [1024,1024,...]
        activation: a list, ['ReLU', 'Tanh',...]
        bias : bool
        use_bn: bool
        drop_prob: default is None, use drop out layer or not
    '''
    def __init__(self, layer_sizes, activation, bias=True, use_bn=False, drop_prob=None):
        super(MLP, self).__init__()
        self.bn = use_bn
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias)
            activate = get_activation(activation[i])
            block = nn.Sequential(OrderedDict([(f'L{i}', layer), ]))
            
            # !NOTE:# Actually, it is inappropriate to use batch-normalization here
            if use_bn:                                  
                bn = nn.BatchNorm1d(layer_sizes[i+1])
                block.add_module(f'B{i}', bn)
            
            # batch normalization is put before activation function 
            block.add_module(f'A{i}', activate)

            # dropout probablility
            if drop_prob:
                block.add_module(f'D{i}', nn.Dropout(drop_prob))
            
            self.layers.append(block)
    
    def forward(self, x):
        for layer in self.layers:
            # !NOTE: sometime the shape of x will be [1,N], and we cannot use batch-normailzation in that situation
            if self.bn and x.shape[0]==1:
                x = layer[0](x)
                x = layer[:-1](x)
            else:
                x = layer(x)
        return x
