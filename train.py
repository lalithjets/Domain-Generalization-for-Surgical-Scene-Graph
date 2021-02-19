from __future__ import print_function

import os
import time

import numpy as np
from tqdm import tqdm
from PIL import Image
import utils.io as io
#from utils.vis_tool import vis_img

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

def run_model(args, data_const):
    '''
    a) set dataset and dataloader for train and validate set
    b) set model
    c) optimizer
    d) loss
    e) learning rate scheduler
    '''

    # set up dataset variable
    train_dataset = SurgicalSceneDataset(seq_set = [2,4,6,7,9,10,11,12,14,15,16], dataconst = data_const, feature_extractor = args.feature_extractor)
    val_dataset = SurgicalSceneDataset(seq_set= [1,3,5], dataconst = data_const, feature_extractor = args.feature_extractor)
    dataset = {'train': train_dataset, 'val': val_dataset}
    print('set up dataset variable successfully')
   
    # use default DataLoader() to load the data. 
    train_dataloader = DataLoader(dataset=dataset['train'], batch_size=args.batch_size, shuffle= True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=dataset['val'], batch_size=args.batch_size, shuffle= True, collate_fn=collate_fn)
    dataloader = {'train': train_dataloader, 'val': val_dataloader}

    print('set up dataloader successfully')

    # use cpu or cuda
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print('training on {}...'.format(device))

    # model
    model = AGRNN(bias=args.bias, bn=args.bn, dropout=args.drop_prob, multi_attn=args.multi_attn, layer=args.layers, diff_edge=args.diff_edge)

    # calculate the amount of all the learned parameters
    parameter_num = 0
    for param in model.parameters(): parameter_num += param.numel()
    print(f'The parameters number of the model is {parameter_num / 1e6} million')

    # load pretrained model
    if args.pretrained:
        print(f"loading pretrained model {args.pretrained}")
        checkpoints = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoints['state_dict'])
    model.to(device)
    
    # build optimizer  
    if args.optim == 'sgd': optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    else: optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    
    # criterion and scheduler
    criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.3) #the scheduler divides the lr by 10 every 150 epochs

    # get the configuration of the model and save some key configurations
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver), recursive=True)
    for i in range(args.layers):
        if i==0:
            model_config = model.CONFIG1.save_config()
            model_config['lr'] = args.lr
            model_config['bs'] = args.batch_size
            model_config['layers'] = args.layers
            model_config['multi_attn'] = args.multi_attn
            model_config['data_aug'] = args.data_aug
            model_config['drop_out'] = args.drop_prob
            model_config['optimizer'] = args.optim
            model_config['diff_edge'] = args.diff_edge
            model_config['model_parameters'] = parameter_num
            io.dump_json_object(model_config, os.path.join(args.save_dir, args.exp_ver, 'l1_config.json'))
    print('save key configurations successfully...')

    epoch_train(model, dataloader, dataset, criterion, optimizer, scheduler, device, data_const)

def epoch_train(model, dataloader, dataset, criterion, optimizer, scheduler, device, data_const):
    '''
    input: model, dataloader, dataset, criterain, optimizer, scheduler, device, data_const
    data: 
        img_name, node_num, roi_labels, det_boxes, edge_labels,
        edge_num, features, spatial_features, word2vec
    '''
    print('epoch training...')
    
    # set visualization and create folder to save checkpoints
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.exp_ver + '/' + 'epoch_train')
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver, 'epoch_train'), recursive=True)

    for epoch in range(args.start_epoch, args.epoch):
        # each epoch has a training and validation step
        epoch_loss = 0
        for phase in ['train', 'val']:
            start_time = time.time()
            running_loss = 0.0
            idx = 0
            
            for data in tqdm(dataloader[phase]):
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
                
                #if idx == 2: break
                    
                if phase == 'train':
                    model.train()
                    model.zero_grad()
                    outputs = model(node_num, features, spatial_feat, word2vec, roi_labels)
                    loss = criterion(outputs, edge_labels.float())
                    loss.backward()
                    optimizer.step()

                else:
                    model.eval()
                    # turn off the gradients for validation, save memory and computations
                    with torch.no_grad():
                        outputs = model(node_num, features, spatial_feat, word2vec, roi_labels, validation=True)
                        loss = criterion(outputs, edge_labels.float())
                    
                        # print result every 1000 iteration during validation
                        if idx == 10:
                            #print(img_loc[0])
                            io.mkdir_if_not_exists(os.path.join(args.output_img_dir, ('epoch_'+str(epoch))), recursive=True)
                            image = Image.open(img_loc[0]).convert('RGB')
                            det_actions = nn.Sigmoid()(outputs[0:int(edge_num[0])])
                            det_actions = det_actions.cpu().detach().numpy()
                            action_img = vis_img(image, roi_labels[0], det_boxes[0],  det_actions, score_thresh = 0.7)
                            image = image.save(os.path.join(args.output_img_dir, ('epoch_'+str(epoch)),img_name[0]))

                idx+=1
                # accumulate loss of each batch
                running_loss += loss.item() * edge_labels.shape[0]
            
            # calculate the loss and accuracy of each epoch
            epoch_loss = running_loss / len(dataset[phase])
            
            # import ipdb; ipdb.set_trace()
            # log trainval datas, and visualize them in the same graph
            if phase == 'train':
                train_loss = epoch_loss 
            else:
                writer.add_scalars('trainval_loss_epoch', {'train': train_loss, 'val': epoch_loss}, epoch)
            
            # print data
            if (epoch % args.print_every) == 0:
                end_time = time.time()
                print("[{}] Epoch: {}/{} Loss: {} Execution time: {}".format(\
                        phase, epoch+1, args.epoch, epoch_loss, (end_time-start_time)))
                        
        # scheduler.step()
        # save model
        if epoch_loss<0.0405 or epoch % args.save_every == (args.save_every - 1) and epoch >= (200-1):
            checkpoint = { 
                            'lr': args.lr,
                           'b_s': args.batch_size,
                          'bias': args.bias, 
                            'bn': args.bn, 
                       'dropout': args.drop_prob,
                        'layers': args.layers,
                    'multi_head': args.multi_attn,
                     'diff_edge': args.diff_edge,
                    'state_dict': model.state_dict()
            }
            save_name = "checkpoint_" + str(epoch+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, args.exp_ver, 'epoch_train', save_name))

    writer.close()
    print('Finishing training!')
