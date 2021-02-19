'''
configurations of the network
    
    readout: G_ER_L_S = [1024+300+16+300+1024,  1024, 117]

    node_func: G_N_L_S = [1024+1024, 1024]
    node_lang_func: G_N_L_S2 = [300+300+300]
    
    edge_func : G_E_L_S = [1024*2+16, 1024]
    edge_lang_func: [300*2, 1024]
    
    attn: [1024, 1]
    attn_lang: [1024, 1]
'''

class CONFIGURATION(object):
    '''
    Configuration arguments: feature type, layer, bias, batch normalization, dropout, multi-attn
    
    readout           : fc_size, activation, bias, bn, droupout
    gnn_node          : fc_size, activation, bias, bn, droupout
    gnn_node_for_lang : fc_size, activation, bias, bn, droupout
    gnn_edge          : fc_size, activation, bias, bn, droupout
    gnn_edge_for_lang : fc_size, activation, bias, bn, droupout
    gnn_attn          : fc_size, activation, bias, bn, droupout
    gnn_attn_for_lang : fc_size, activation, bias, bn, droupout
    '''
    def __init__(self, layer=1, bias=True, bn=False, dropout=0.2, multi_attn=False):
        
        # if multi_attn:
        if True:
            if layer==1:
                    
                # readout
                self.G_ER_L_S = [1024+300+16+300+1024, 1024, 13]
                self.G_ER_A   = ['ReLU', 'Identity']
                self.G_ER_B   = bias    #true
                self.G_ER_BN  = bn      #false
                self.G_ER_D   = dropout #0.3
                # self.G_ER_GRU = 1024

                # # gnn node function
                self.G_N_L_S = [1024+1024, 1024]
                self.G_N_A   = ['ReLU']
                self.G_N_B   = bias #true
                self.G_N_BN  = bn      #false
                self.G_N_D   = dropout #0.3
                # self.G_N_GRU = 1024

                # # gnn node function for language
                self.G_N_L_S2 = [300+300, 300]
                self.G_N_A2   = ['ReLU']
                self.G_N_B2   = bias    #true
                self.G_N_BN2  = bn      #false
                self.G_N_D2   = dropout #0.3
                # self.G_N_GRU2 = 1024

                # gnn edge function1
                self.G_E_L_S = [1024*2+16, 1024]
                self.G_E_A   = ['ReLU']
                self.G_E_B   = bias     #true
                self.G_E_BN  = bn       #false
                self.G_E_D   = dropout  #0.3

                # gnn edge function2 for language
                self.G_E_L_S2 = [300*2, 1024]
                self.G_E_A2   = ['ReLU']
                self.G_E_B2   = bias     #true
                self.G_E_BN2  = bn       #false
                self.G_E_D2   = dropout  #0.3

                # gnn attention mechanism
                self.G_A_L_S = [1024, 1]
                self.G_A_A   = ['LeakyReLU']
                self.G_A_B   = bias     #true
                self.G_A_BN  = bn       #false
                self.G_A_D   = dropout  #0.3

                # gnn attention mechanism2 for language
                self.G_A_L_S2 = [1024, 1]
                self.G_A_A2   = ['LeakyReLU']
                self.G_A_B2   = bias    #true
                self.G_A_BN2  = bn      #false
                self.G_A_D2   = dropout #0.3
                    
    def save_config(self):
        model_config = {'graph_head':{}, 'graph_node':{}, 'graph_edge':{}, 'graph_attn':{}}
        CONFIG=self.__dict__
        for k, v in CONFIG.items():
            if 'G_H' in k:
                model_config['graph_head'][k]=v
            elif 'G_N' in k:
                model_config['graph_node'][k]=v
            elif 'G_E' in k:
                model_config['graph_edge'][k]=v
            elif 'G_A' in k:
                model_config['graph_attn'][k]=v
            else:
                model_config[k]=v
        
        return model_config