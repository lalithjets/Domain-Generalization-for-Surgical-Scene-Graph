'''
Project         : Learning Domain Generaliazation with Graph Neural Network for Surgical Scene Understanding.
Lab             : MMLAB, National University of Singapore
contributors    : Lalith, Mobarak
Note            : Incremental learning for surgical instrument classification and feature extraction
                  This work is extended from Incremental learning and Curriculum by smoothing.
                  This work includes:
                      a) Naive implementation of incremental learning
                      b) Improve curriculum based learning by introduced Laplacian of Gaussian based filters.
                  
                  Incremental Learning:
                    @inproceedings{castro2018end,
                        title={End-to-end incremental learning},
                        author={Castro, Francisco M and Mar{\'\i}n-Jim{\'e}nez, Manuel J and Guil, Nicol{\'a}s and Schmid, Cordelia and Alahari, Karteek},
                        booktitle={Proceedings of the European conference on computer vision (ECCV)},
                        pages={233--248},
                        year={2018}
                    }
                  
                  Curriculum by smoothing:
                    @article{sinha2020curriculum,
                        title={Curriculum by smoothing},
                        author={Sinha, Samarth and Garg, Animesh and Larochelle, Hugo},
                        journal={arXiv preprint arXiv:2003.01367},
                        year={2020}
                    }

'''

import os
import copy
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.resnet import *
from models.cel_ls import *
from models.surgical_class_dataloader_il import *


def seed_everything(seed=27):
    '''
    Fixing the random seeds
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train (args, period, model, model_old, train_loader, loss_criterion, optimizer, class_old, class_novel, finetune):
    '''
    arguments: args, period, model, model_old, train_loader, loss_criterion, optimizer, class_old, class_novel, finetune
    returns: tcost, loss_avg, acc_avg
    Code adopted from incremental learning
    '''

    acc_avg = 0
    num_exp = 0
    loss_avg = 0
    loss_cls_avg = 0
    loss_dist_avg = 0
    tstart = time.clock()

    # set net to train mode
    model.train()
    model_old.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        #optimizer.zero_grad()

        # prepare target_onehot
        bs = np.shape(target)[0]
        target_onehot = np.zeros(shape = (bs, args.num_classes), dtype=np.int)
        for i in range(bs): target_onehot[i,target[i]] = 1
        target_onehot = torch.from_numpy(target_onehot)
        target_onehot = target_onehot.float()

        # indices for combined classes
        class_indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
        
        # send data to cuda
        if args.cuda:
            data = data.cuda()
            target_onehot = target_onehot.cuda()
            class_indices = class_indices.cuda()

        # predict output
        output = model(data)

        # loss for network
        output_new_onehot = torch.index_select(output, 1, class_indices)
        target_onehot = torch.index_select(target_onehot, 1, class_indices)
        combined_loss = loss_criterion(output_new_onehot, target_onehot)

        ''' ===== Distillation loss based on old net ====='''
        if (period > 0):

            # distillation loss activation
            if args.dist_loss_act == 'softmax': dist_loss_act = nn.Softmax(dim=1)
            if args.cuda: dist_loss_act = dist_loss_act.cuda()
            
            # indices of old class
            if not finetune:
                class_indices = torch.LongTensor(class_old)
                if args.cuda: class_indices = class_indices.cuda()
                    
            # current_network output
            dist = torch.index_select(output, 1, class_indices)
            if args.use_tnorm: dist = dist/args.tnorm

            with torch.no_grad():
                # old network output
                output_old = model_old(data)
                output_old = torch.index_select(output_old, 1, class_indices)
            
            target_dist = Variable(output_old)
            if args.use_tnorm: target_dist = target_dist/args.tnorm
            #loss_dist = loss_criterion(dist, loss_activation(target_dist))
            
            if(args.dist_loss == 'ce'):
                loss_dist = F.binary_cross_entropy(dist_loss_act(dist), dist_loss_act(target_dist))
            else: loss_dist = 0.0
            
        else: loss_dist = 0.0
        '''----------------------------------------------'''

        # loss calculatoin
        loss = combined_loss + args.dist_ratio*loss_dist
        loss_avg += loss.item()
        loss_cls_avg += combined_loss.item()
        if period == 0: loss_dist_avg += 0
        else:loss_dist_avg += loss_dist.item()

        acc = np.sum(np.equal(np.argmax(output_new_onehot.cpu().data.numpy(), axis=-1), np.argmax(target_onehot.cpu().data.numpy(), axis=-1)))
        acc_avg += acc
        num_exp += np.shape(target)[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # average calculation
    loss_avg /= num_exp
    loss_cls_avg /= num_exp
    loss_dist_avg /= num_exp
        
    # average calculation
    acc_avg /= num_exp

    # time calculation
    tend = time.clock()
    tcost = tend - tstart

    return(tcost, loss_avg, acc_avg)


def test(args, model, test_loader, class_old, class_novel):
    '''
    arguments: args, model, test_loader, class_old, class_novel
    return: tcost, acc_avg
    Code adopted from incremental learning
    '''

    acc_avg = 0
    num_exp = 0
    tstart = time.clock()

    # set net to eval
    model.eval()
    
    # loss
    if args.dist_loss_act == 'softmax': 
        dist_loss_act = nn.Softmax(dim=1)
    else:
        dist_loss_act = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            # prepare target_onehot
            bs = np.shape(target)[0]
            target_onehot = np.zeros(shape = (bs, args.num_classes), dtype=np.int)
            for i in range(bs): target_onehot[i,target[i]] = 1
            target_onehot = torch.from_numpy(target_onehot)
            target_onehot = target_onehot.float()
            
            # indices for combined classes
            class_indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))

            # send image and target to cuda
            if args.cuda:
                data = data.cuda()
                target_onehot = target_onehot.cuda()
                class_indices = class_indices.cuda()
                dist_loss_act = dist_loss_act.cuda()

            # predict output
            output = model(data)

            # calculate output and target one_hot
            output = torch.index_select(output, 1, class_indices)
            output = dist_loss_act(output)
            output = output.cpu().data.numpy()
            target_onehot = torch.index_select(target_onehot, 1, class_indices)
            #target_onehot = target_onehot[:, np.concatenate((class_old, class_novel), axis=0)]

            # calculation accuracy
            acc = np.sum(np.equal(np.argmax(output, axis=-1), np.argmax(target_onehot.cpu().data.numpy(), axis=-1)))
            acc_avg += acc
            num_exp += np.shape(target)[0]

    # calculate average accuracy
    acc_avg /= num_exp
            
    # time calculation
    tend = time.clock()
    tcost = tend - tstart

    return(tcost, acc_avg)


def save_model(args, best_model):
    '''
    save model
    '''
    if not os.path.exists('./weights'): os.mkdir('weights/')
    
    filename = os.path.join('weights', args.log_name + '_model.tar')
    torch.save(best_model.state_dict(), filename)


def main(args):
    '''
    Train the model based on period
    period 0: Train on first 8 classes.
    period 1: Train on new 2 class + knowledge distillation based re-train on samples of first 8 classes
    Finetune: Finetune on samples of first 8 classes and new 2 classes.
    Code adopted from incremental learning
    '''
    
    print('ls', args.use_ls, 'tnorm', args.use_tnorm, 'cbs', args.use_cbs)
    # learning rate schedules
    schedules = range(args.schedule_interval, args.epoch_base, args.schedule_interval)

    # class order in icremental learning
    class_order = np.arange(args.num_classes) #np.random.permutation(args.num_class)
    print('class order:', class_order)

    # check for pre-trained model
    model_path = args.checkpointfile + '_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[1]]), '.pkl')
    flag_model = os.path.exists(model_path)

    # network
    model = ResNet18(args)
    model_old = copy.deepcopy(model)
    
    # curicullum learning
    if args.use_cbs:
        model.get_new_kernels(0)
        model_old.get_new_kernels(0)  
    
    # loss
    if args.use_ls: 
        loss_criterion = CELossWithLS(smoothing = 0.1, gamma=0.0, isCos=False, ignore_index=-1)
    else: 
        loss_criterion = CELossWithLS(smoothing= 0.0, gamma=0.0, isCos=False, ignore_index=-1)
        
    # gpu
    num_gpu = torch.cuda.device_count()
    if num_gpu > 0:
        device_ids = np.arange(num_gpu).tolist()
        print('device_ids:', device_ids)
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        model_old = nn.DataParallel(model_old, device_ids=device_ids).cuda()
        loss_criterion = loss_criterion.cuda()
    else: print('only cpu is available')
        
 

    # initializing classes, accuracy and memory array
    memory_train = []                                  # train memory array
    class_old = np.array([], dtype=int)                # old class array
    acc_nvld_basic = np.zeros((args.period_train))     # accuracy list
    acc_nvld_finetune = np.zeros((args.period_train))  # accuracy list

    
    for period in range(args.period_train):

        print('===================== period = %d ========================='%(period))

        # current 10 classes
        class_novel = class_order[args.num_class_novel[period]:args.num_class_novel[period+1]]
        print('class_novel:', class_novel)

        # combined train dataloader
        combined_train_files = memory_train + args.novel_train_files[period:period+1]
        combined_train_dataset = SurgicalClassDataset18_incremental(filenames= combined_train_files, is_train=True)
        combined_train_loader = DataLoader(dataset=combined_train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=2, drop_last=False)
        print('train files: \t size: ', len(combined_train_loader.dataset), ' , files: ', combined_train_files)

        # test dataloader
        combined_test_files = args.novel_test_files[0:period+1]
        test_dataset = SurgicalClassDataset18_incremental(filenames= combined_test_files, is_train=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size= args.batch_size, shuffle=True, num_workers=2, drop_last=False)
        print('train files: \t size: ', len(test_loader.dataset), ' , files: ', combined_test_files)

        # initialize variables
        lrc = args.lr
        acc_training = []
        print('current lr = %f' % (lrc))

        # epoch training
        for epoch in range(args.epoch_base):
        
            # load pretrained model
            if period == 0 and flag_model:
                print('load model: %s' % model_path)
                model.load_state_dict(torch.load(model_path))
            
            if args.use_cbs:
                model.module.get_new_kernels(epoch)
                model_old.module.get_new_kernels(epoch)
                model.cuda()
                model_old.cuda()

            ''' ====== training combined ======''' 
            # decaying learning rate
            if epoch in schedules:
                lrc *= args.gamma
                print('current lr = %f' % (lrc))

            # Optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=lrc, momentum=args.momentum, weight_decay=args.decay)

            # train
            tcost, loss_avg, acc_avg = train(args, period, model, model_old, combined_train_loader, 
                                             loss_criterion, optimizer, class_old, class_novel, False)

            acc_training.append(acc_avg)
            print('Training Period: %d \t Epoch: %d \t time = %.1f \t loss = %.6f \t acc = %.4f' % (period, epoch, tcost, loss_avg, acc_avg))
            '''--------------------------------'''

            ''' ====== Test combined ======'''
            # test model
            tcost, acc_avg = test(args, model, test_loader,class_old, class_novel)

            acc_nvld_basic[period] = acc_avg
            print('Test(n&o)Period: %d \t Epoch: %d \t time = %.1f \t\t\t\t acc = %.4f' % (period, epoch, tcost, acc_avg))

            # exit if pre-trained model / loss converged
            if period == 0 and flag_model: break
            if len(acc_training)>20 and acc_training[-1]>args.stop_acc and acc_training[-5]>args.stop_acc:
                print('training loss converged')
                break
            '''----------------------------'''

        ''' copy net-old for finetuning '''
        model_old = copy.deepcopy(model)
        '''-----------------------------'''

        ''' ===== Finetuning ====='''
        if period > 0:
            
            acc_finetune_train = []
            lrc = args.lr*args.ft_lr_factor # finetune lr
            print('finetune current lr = %f' % (lrc))

            for epoch in range(args.epoch_finetune):

                # fine tune train_dataloaders
                ft_size = (args.num_class_novel[period+1]-args.num_class_novel[period])*args.memory_size
                ft_combined_train_dataset = SurgicalClassDataset18_incremental(filenames= combined_train_files, fine_tune_size = ft_size, is_train=True)
                ft_combined_train_loader = DataLoader(dataset=ft_combined_train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=2, drop_last=False)
                if(epoch == 0):  print('finetune train size:', len(ft_combined_train_loader.dataset))

                ''' ===== training combined =====''' 
                # learning rate
                if epoch in schedules:
                    lrc *= args.gamma
                    print('current lr = %f'%(lrc))

                # optimizer
                # criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=lrc, momentum=args.momentum, weight_decay=args.decay)

                # train
                tcost, loss_avg, acc_avg = train(args, period, model, model_old, ft_combined_train_loader, 
                                                 loss_criterion, optimizer, class_old, class_novel, True)

                acc_finetune_train.append(acc_avg)
                print('Finetune Training Period: %d \t Epoch: %d \t time = %.1f \t loss = %.6f \t acc = %.4f'%(period, epoch, tcost, loss_avg, acc_avg))
                '''------------------------------'''

                ''' ===== Test combined ====='''
                # test
                tcost, acc_avg = test(args, model, test_loader, class_old, class_novel)

                acc_nvld_finetune[period] = acc_avg
                print('Finetune Test(n&o) Period: %d \t Epoch: %d \t time = %.1f \t\t\t\t acc = %.4f' % (period, epoch, tcost, acc_avg))

                if len(acc_finetune_train) > 20 and acc_finetune_train[-1] > args.stop_acc and acc_finetune_train[-5] > args.stop_acc:
                    print('finetune training loss converged')
                    break
                '''--------------------------'''
                
            print('------------------- result ------------------------')
            print('Period: %d, basic acc = %.4f, finetune acc = %.4f' % (period, acc_nvld_basic[period], acc_nvld_finetune[period]))
            print('---------------------------------------------------')

        if period == args.period_train-1:
            print('------------------- ave result ------------------------')
            print('basic acc = %.4f, finetune acc = %.4f' % (np.mean(acc_nvld_basic[1:], keepdims=False), np.mean(acc_nvld_finetune[1:], keepdims=False)))
            print('---------------------------------------------------')

        print('===========================================================')

        # save model
        model_path = args.checkpointfile + '_%d_%s%s' % (0, ''.join(str(e) for e in class_order[args.num_class_novel[0]:args.num_class_novel[period+1]]), '.pkl')
        print('save model: %s' % model_path)
        torch.save(model.state_dict(), model_path)

        ''' ===== random images selection ====='''
        #remove memory files from old runs
        if os.path.exists(('data_files/memory_'+str(period)+'.txt')): 
            os.remove(('data_files/memory_'+str(period)+'.txt'))

        curr_file = open((args.novel_train_files[period]), 'r') 
        memory_file = open(('data_files/memory_'+str(period)+'.txt'), 'a')
        Lines = curr_file.readlines()        
        indices = np.random.permutation(len(Lines))
        Lines = [Lines[i] for i in indices[0:((args.num_class_novel[period+1]-args.num_class_novel[period])*args.memory_size)]]
        for line in Lines: memory_file.write(line)
        curr_file.close()
        memory_file.close()

        # add new memory file to memory train list
        memory_train.append('data_files/memory_'+str(period)+'.txt')
        print('memory_train', memory_train)
        '''------------------------------------'''

        #append new class images (create new)
        class_old = np.append(class_old, class_novel, axis=0)
    
    print('acc_base    : ', acc_nvld_basic)     # accuracy list
    print('acc_finetune: ', acc_nvld_finetune)

    print('xxx')



if __name__ == '__main__':
    
    # File locations 
    # incremental learning
    novel_train_files = ['data_files/class0_8_train.txt', 'data_files/class9_10_train.txt']
    novel_test_files = ['data_files/class0_8_test.txt', 'data_files/class9_10_test.txt']
    
    # Retrain to all class
    #novel_train_files = ['data_files/class0_10_train.txt']
    #novel_test_files = ['data_files/class0_10_test.txt']
    
    # train only for first 9 class
    #novel_train_files = ['data_files/class0_8_train.txt']
    #novel_test_files = ['data_files/class0_8_test.txt']
    
    '''--------------------------------------------------- Arguments ------------------------------------------------------------'''
    parser = argparse.ArgumentParser(description='Incremental learning for feature extraction')

    # incremental learning
    parser.add_argument('--epoch_base',         type=int,       default=30,          help='30')
    parser.add_argument('--epoch_finetune',     type=int,       default=15,          help='15')
    parser.add_argument('--batch_size',         type=int,       default=20,          help='20')
    parser.add_argument('--period_train',       type=int,       default=2,           help='2')
    parser.add_argument('--num_classes',        type=int,       default=11,          help='11')
    parser.add_argument('--num_class_novel',                    default=[0,9,11],    help='[0,9,11]')
    parser.add_argument('--memory_size',                        default=50,          help='50')

    parser.add_argument('--stop_acc',           type=float,     default=0.998,       help='number of epochs')

    # model
    parser.add_argument('--alg',                type=str,       default='res',       help='res')

    # datasets
    parser.add_argument('--novel_train_files',  default=novel_train_files,           help='list of train files')
    parser.add_argument('--novel_test_files',   default=novel_test_files,            help='list of test files')

    # learning rate
    parser.add_argument('--schedule_interval',  type=int,       default=3,           help='decay epoch rate: 3')
    parser.add_argument('--lr',                 type=float,     default=0.001,       help='learn rate: 0.001') 
    parser.add_argument('--gamma',              type=float,     default=0.8,         help='decay lr factor: 0.8')
    parser.add_argument('--ft_lr_factor',       type=float,     default=0.1,         help='ft learn rate: 0.1')
    
    # loss
    parser.add_argument('--dist_loss',          type=str,       default='ce',        help='dist_loss')
    parser.add_argument('--dist_loss_act',      type=str,       default='softmax',   help='dist_loss_act')
    parser.add_argument('--dist_ratio',         type=float,     default=0.5,         help='dist_loss_ratio')
    
    # optimizer
    parser.add_argument('--momentum',           type=float,     default=0.6,         help='learning momentum') 
    parser.add_argument('--decay',              type=float,     default=0.0001,      help='learning rate')
    
    # Label smoothing
    parser.add_argument('--use_ls',             type=bool,      default=False,        help='list of test files')

    # Temperature scaling
    parser.add_argument('--use_tnorm',             type=bool,      default=True,       help='use temp_scale')
    parser.add_argument('--tnorm',             type=float,     default=3.0,         help='Temp scaling')
    
    # CBS ARGS
    parser.add_argument('--use_cbs',            type=bool,      default=True,       help='use CBS')
    parser.add_argument('--std',                type=float,     default=1.0,         help='')
    parser.add_argument('--std_factor',         type=float,     default=0.9,         help='')
    parser.add_argument('--cbs_epoch',          type=int,       default=5,           help='')
    parser.add_argument('--kernel_size',        type=int,       default=3,           help='')
    parser.add_argument('--fil1',               type=str,       default='LOG',       help='gau, LOG')
    parser.add_argument('--fil2',               type=str,       default='gau',       help='gau, LOG')
    parser.add_argument('--fil3',               type=str,       default='gau',       help='gau, LOG')
    
    parser.add_argument('--save_model',         type=bool,      default=False,       help='store_true')
    parser.add_argument('--checkpointfile',     type=str,       default='checkpoint/incremental/testing')
   
    args = parser.parse_args()
    '''-------------------------------------------------------------------------------------------------------------------------'''
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    if torch.cuda.is_available(): args.cuda = True
    
    seed_everything()
    main(args)

#python3 main.py --dataset cifar10 --alg res --data ./data/