#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov.gov)
        : Miroslav Stoyanov (e-mail: stoyanovmk@ornl.gov.gov)
        : Viktor Reshniak (e-mail: reshniakv@ornl.gov.gov)
"""
'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os
import argparse
from copy import deepcopy

import AADL as accelerate

import sys
sys.path.append("../../../utils")
from gpu_detection import get_gpu
from monitor_progress_utils import progress_bar


# Import paths to NN models that can be used for object classification
import sys
sys.path.append("../../../model_zoo")
from densenet import *
from dla import *
from dla_simple import *
from dpn import *
from efficientnet import *
from googlenet import *
from lenet import *
from pnasnet import *
from preact_resnet import *
from regnet import *
from resnet import *
from resnext import *
from shufflenet import *
from shufflenetv2 import *
from vgg import *


class Optimization:
    
    def __init__(self, network: torch.nn.Module, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader, optimizer: torch.optim, num_epochs: int):
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.network = network
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.training_loss_history = []
        self.training_accuracy_history = []
        self.validation_loss_history = []
        self.validation_accuracy_history = []
        
    def initial_performance_evaluation(self):
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.network(inputs)
            loss = criterion(outputs, targets)
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        self.training_loss_history.append(train_loss)
        self.training_accuracy_history.append(100.*correct/total)        
        self.validation_loss_history.append(train_loss)
        self.validation_accuracy_history.append(100.*correct/total)        
        
        
    # Training
    def train_epoch(self, epoch):
        
        print('\nEpoch: %d' % epoch)
        self.network.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.network(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        self.training_loss_history.append(train_loss)
        self.training_accuracy_history.append(100.*correct/total)
            
    
    def validation_epoch(self, epoch):
        
        self.network.eval()
        validation_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.network(inputs)
                loss = criterion(outputs, targets)
    
                validation_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (validation_loss/(batch_idx+1), 100.*correct/total, correct, total))      
            
        self.validation_loss_history.append(validation_loss) 
        self.validation_accuracy_history.append(100.*correct/total)    
        
        
    def train(self):
        
        for epoch in range(0, self.num_epochs):
            self.initial_performance_evaluation()
            self.train_epoch(epoch)
            self.validation_epoch(epoch)
            
        return self.training_loss_history, self.training_accuracy_history, self.validation_loss_history, self.validation_accuracy_history
        
    


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# The only reason why I do this workaround (not necessary now) is because
# I am thinking to the situation where one MPI process has multiple gpus available
# In that case, the argument passed to get_gpu may be a numberID > 0
device = get_gpu(0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = SimpleDLA()

torch.manual_seed(0)

# Perform deepcopies of the original model 
net_classic = deepcopy(net)
net_anderson = deepcopy(net)

# Map neural networks to aq device if any GPU is available
net_classic = net_classic.to(device)
net_anderson = net_anderson.to(device)

if device == 'cuda':
    net_classic = torch.nn.DataParallel(net_classic)
    net_anderson = torch.nn.DataParallel(net_anderson)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer_classic = optim.SGD(net_classic.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler_classic = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classic, T_max=200)

# Parameters for Anderson acceleration
relaxation = 0.1
wait_iterations = 1
history_depth = 10
store_each_nth = 1
frequency = store_each_nth
reg_acc = 1e-8

optimizer_anderson= optim.SGD(net_anderson.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler_anderson = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_anderson, T_max=200)
accelerate.accelerate(optimizer_anderson, "anderson", relaxation, wait_iterations, history_depth, store_each_nth, frequency, reg_acc)

optimization_classic = Optimization(net_classic, trainloader, testloader, optimizer_classic, 100)
optimization_anderson = Optimization(net_anderson, trainloader, testloader, optimizer_anderson, 100)

_, _, validation_loss_classic, validation_accuracy_classic = optimization_classic.train()
_, _, validation_loss_anderson, validation_accuracy_anderson = optimization_anderson.train()

epochs1 = range(0, len(validation_loss_classic) + 1)
epochs2 = range(0, len(validation_loss_anderson) + 1)

plt.figure()
plt.plot(epochs1,validation_loss_classic,linestyle='-', label="SGD")
plt.plot(epochs2,validation_loss_anderson,linestyle='-', label="SGD + Anderson")         
plt.yscale('log')
plt.title('Validation loss function')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.legend()
plt.draw()
plt.savefig('validation_loss_plot')
plt.tight_layout()

plt.figure()
plt.plot(epochs1,validation_accuracy_classic,linestyle='-', label="SGD")
plt.plot(epochs2,validation_accuracy_anderson,linestyle='-', label="SGD + Anderson")
plt.yscale('log')
plt.title('Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
#plt.legend()
plt.draw()
plt.savefig('validation_accuracy_plot')
plt.tight_layout()

