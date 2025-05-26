import torch
import torch.nn as nn
import torchvision
import numpy as np

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, LPN=False):
        super(ConvRelu, self).__init__()
        self.LPN = LPN
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=False)
        if self.LPN:
            self.lpn = nn.LocalResponseNorm(2)

    def forward(self, x):
        out = self.conv(x)
        if self.LPN:
            out = self.lpn(out)
        out = self.relu(out)
        return out
    
class DenseRelu(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate):
        super(DenseRelu, self).__init__()
        self.linear = nn.Linear(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        out = self.drop(x)
        out = self.linear(out)
        out = self.relu(out)
        return out

class VGG(nn.Module):
    def __init__(self, num_channels, num_layers, num_class, drop_rate, version, depth=[1, 2, 4, 8, 8]):
        super(VGG, self).__init__()
        
        # init
        self.channels = num_channels[0]
        
        # construct
        out_channels, layers, classifier = 64, [], []
        
        if num_channels[1] == 32:
            self.out_dim = 512
        elif num_channels[1] == 112:
            self.out_dim = 4608
        elif num_channels[1] == 224:
            self.out_dim = 25088
        
        # get layers
        # in_ch, out_ch, kernel_size, stride, padding
        if 'advanced' in version:
            print('Create Advanced VGG')
            ch_remember = 64
            for i in range(len(num_layers)):
                for num in range(num_layers[i]):
                    if i == 0 and len(layers) == 0:
                        layers.append(ConvBnRelu(self.channels, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                    else:
                        layers.append(ConvBnRelu(ch_remember, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            classifier.append(DenseRelu(self.out_dim, 4096, drop_rate))
            classifier.append(DenseRelu(4096, 4096, drop_rate))
        elif 'LRN' in version:
            print('Create LRN VGG')
            ch_remember = 64
            for i in range(len(num_layers)):
                for num in range(num_layers[i]):
                    if i == 0 and len(layers) == 0:
                        layers.append(ConvRelu(self.channels, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                    else:
                        layers.append(ConvRelu(ch_remember, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            classifier.append(DenseRelu(self.out_dim, 4096, drop_rate))
            classifier.append(DenseRelu(4096, 4096, drop_rate))
        elif 'v1' in version:
            print('Create VGG16 Version 1')
            ch_remember = 64
            for i in range(len(num_layers)):
                for num in range(num_layers[i]):
                    if i == 0 and len(layers) == 0:
                        layers.append(ConvRelu(self.channels, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                    elif i > 1 and num == num_layers[i]-1:
                        layers.append(ConvRelu(ch_remember, out_channels * depth[i], 1, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                    else:
                        layers.append(ConvRelu(ch_remember, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            classifier.append(DenseRelu(self.out_dim, 4096, drop_rate))
            classifier.append(DenseRelu(4096, 4096, drop_rate))
        elif 'v2' in version:
            print('Create VGG16 Version 2')
            ch_remember = 64
            for i in range(len(num_layers)):
                for num in range(num_layers[i]):
                    if i == 0 and len(layers) == 0:
                        layers.append(ConvRelu(self.channels, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                    else:
                        layers.append(ConvRelu(ch_remember, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            classifier.append(DenseRelu(self.out_dim, 4096, drop_rate))
            classifier.append(DenseRelu(4096, 4096, drop_rate))
        else:
            print('Create Normal Version')
            ch_remember = 64
            for i in range(len(num_layers)):
                for num in range(num_layers[i]):
                    if i == 0 and len(layers) == 0:
                        layers.append(ConvRelu(self.channels, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                    else:
                        layers.append(ConvRelu(ch_remember, out_channels * depth[i], 3, 1, 'same'))
                        ch_remember = out_channels * depth[i]
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            classifier.append(DenseRelu(self.out_dim, 4096, drop_rate))
            classifier.append(DenseRelu(4096, 4096, drop_rate))
        
        # create model
        classifier.append(nn.Linear(4096, num_class))
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(*classifier)
        
    def forward(self, x):
        out = self.layers(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out
    
def vgg(num_class, input_shape, dropout, version='vgg16_v1'):
    # VGG versions
    version_dict = {'vgg11' : [1, 1, 2, 2, 2],
                    'vgg11_LRN' : [1, 1, 2, 2, 2],
                    'vgg13' : [2, 2, 2, 2, 2],
                    'vgg16_v1' : [2, 2, 3, 3, 3],
                    'vgg16_v2' : [2, 2, 3, 3, 3],
                    'vgg16_advanced' : [2, 2, 3, 3, 3],
                    'vgg19' : [2, 2, 4, 4, 4],
                    'vgg19_advanced' : [2, 2, 4, 4, 4]}

    print(f'---- version : {version}  num layers : {version_dict[version]}  num class : {num_class}  input shape : {input_shape}  drop rate : {dropout} ----')
    model = VGG(input_shape, version_dict[version], num_class, dropout, version)
    return model