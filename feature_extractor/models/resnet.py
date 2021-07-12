
'''
Project         : Learning Domain Generaliazation with Graph Neural Network for Surgical Scene Understanding.
Lab             : MMLAB, National University of Singapore
contributors    : Lalith, Mobarak
Note            : ResNet (Pytorch implementation), together with curricullum learning filters
                  a) Resnet Adopted from:
                        @inproceedings{he2016deep,
                            title={Deep residual learning for image recognition},
                            author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
                            booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
                            pages={770--778},
                            year={2016}
                        }
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from models.cbs_filters import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.planes = planes
        self.enable_cbs = False
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_kernel = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def get_new_kernels(self, fil2, fil3, kernel_size, std):
        self.enable_cbs = True
        if (fil2 == 'gau'): 
            self.kernel1 = get_gaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)
        elif (fil2 == 'LOG'): 
            self.kernel1 = get_laplaceOfGaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)

        if (fil3 == 'gau'): 
            self.kernel2 = get_gaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)
        elif (fil3 == 'LOG'): 
            self.kernel2 = get_laplaceOfGaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)

    def forward(self, x):
        out = self.conv1(x)
        
        if self.enable_cbs: out = F.relu(self.bn1(self.kernel1(out)))         
        else: out = F.relu(self.bn1(out))         
        
        out = self.conv2(out)
        
        if self.enable_cbs: out = self.bn2(self.kernel2(out))
        else: out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
               
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.std = args.std
        self.enable_cbs = args.use_cbs
        self.factor = args.std_factor
        self.epoch = args.cbs_epoch
        self.kernel_size = args.kernel_size

        self.fil1 = args.fil1
        self.fil2 = args.fil2
        self.fil3 = args.fil3

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, args.num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.enable_cbs: out = F.relu(self.bn1(self.kernel1(out)))
        else: out = F.relu(self.bn1(out))
            
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


    def get_new_kernels(self, epoch_count):
        if epoch_count % self.epoch == 0 and epoch_count is not 0:
            self.std *= self.factor
        if (self.fil1 == 'gau'): 
            self.kernel1 = get_gaussian_filter(kernel_size=self.kernel_size, sigma= self.std, channels=64)
        elif (self.fil1 == 'LOG'): 
            self.kernel1 = get_laplaceOfGaussian_filter(kernel_size=self.kernel_size, sigma= self.std, channels=64)

        for child in self.layer1.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.layer2.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.layer3.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.layer4.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)



def ResNet18(args): return ResNet(BasicBlock, [2,2,2,2], args)


class ResNet18_vanilla(torch.nn.Module):
    '''
    Used for Vanilla feature extraction
    '''
    def __init__(self):
        super(ResNet18_vanilla, self).__init__()
        pretrained_model = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
        
    def forward(self, x):
        x = self.feature_extractor(x)      #torch.Size([1, 512, 1, 1])
        x = x.view(x.size(0), -1)          #torch.Size([1, 512])
        return x