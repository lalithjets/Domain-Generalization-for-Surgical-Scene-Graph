'''
Project         : Learning Domain Generaliazation with Graph Neural Network for Surgical Scene Understanding.
Lab             : MMLAB, National University of Singapore
contributors    : Lalith, Mobarak 
'''

import os
import utils.io as io

class SurgicalSceneConstants():
    def __init__( self):
        self.instrument_classses = ( '', 'kidney', 'bipolar_forceps', 'fenestrated_bipolar', 
                                     'prograsp_forceps', 'large_needle_driver', 'vessel_sealer',
                                     'grasping_retractor', 'monopolar_curved_scissors', 
                                     'ultrasound_probe', 'suction', 'clip_applier', 'stapler')
        self.action_classes = ( 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation', 
                                'Tool_Manipulation', 'Cutting', 'Cauterization', 
                                'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple', 
                                'Ultrasound_Sensing')
        self.xml_data_dir = 'datasets/instruments18/seq_'
        self.word2vec_loc = 'datasets/surgicalscene_word2vec.hdf5'