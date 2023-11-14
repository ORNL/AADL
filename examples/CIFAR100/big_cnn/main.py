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
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import argparse
from copy import deepcopy

import AADL as accelerate

import timeit

def setup_ddp():

    """"Initialize DDP"""

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    master_addr = '127.0.0.1'
    master_port = '8889'
    world_size = os.environ['OMPI_COMM_WORLD_SIZE']
    world_rank = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = world_size
    os.environ['RANK'] = world_rank
    dist.init_process_group(backend=backend, rank=int(world_rank), world_size=int(world_size))


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
from mobilenetv2 import *
from pnasnet import *
from preact_resnet import *
from regnet import *
from resnet import *
from resnext import *
from shufflenet import *
from shufflenetv2 import *
from vgg import *


class Optimization:
    
    def __init__(self, network: torch.nn.Module, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader, optimizer: torch.optim, safeguard: bool, num_epochs: int):
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.network = network
        self.optimizer = optimizer
        self.safeguard = safeguard        
        self.num_epochs = num_epochs
        self.training_loss_history = []
        self.training_accuracy_history = []
        self.validation_loss_history = []
        self.validation_accuracy_history = []
        self.lr = self.optimizer.param_groups[0]['lr']
        
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
        
        self.network.train()
        train_loss = 0
        correct = 0
        total = 0

        if epoch == 30 or epoch == 60 or epoch == 80:
            self.lr *= 0.1

        # warm-up value for learning rate 
        if epoch<=5:
            for g in self.optimizer.param_groups:
                g['lr'] = 1e-4
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.network(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if self.safeguard:
                def closure():
                    if torch.is_grad_enabled():
                        self.optimizer.zero_grad()
                    output = self.network.forward(inputs)
                    loss = criterion(output, targets)
                    if loss.requires_grad:
                        loss.backward()
                    return loss
                self.optimizer.step(closure)
            else:
                self.optimizer.step()
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
 
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
    
                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (validation_loss/(batch_idx+1), 100.*correct/total, correct, total))      
            
        self.validation_loss_history.append(validation_loss) 
        self.validation_accuracy_history.append(100.*correct/total)    
        
        
    def train(self):
        
        self.initial_performance_evaluation()
        for epoch in range(0, self.num_epochs):
            self.train_epoch(epoch)
            self.validation_epoch(epoch)
            
        return self.training_loss_history, self.training_accuracy_history, self.validation_loss_history, self.validation_accuracy_history

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

setup_ddp()


# The only reason why I do this workaround (not necessary now) is because
# I am thinking to the situation where one MPI process has multiple gpus available
# In that case, the argument passed to get_gpu may be a numberID > 0
world_size = os.environ['OMPI_COMM_WORLD_SIZE']
world_rank = os.environ['OMPI_COMM_WORLD_RANK']

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

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=400, shuffle=False, sampler=train_sampler)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, sampler=test_sampler)

# All the neural networks included from 194 down are DL models that I want to run on the same dataset, 
# and test the final accuracy attained by standard optimizers versus the accelerated version with Anderson.

# Model
print('==> Building model..')
# net = VGG('VGG19', num_classes = 100)
# net = ResNet18( num_classes = 100 )
net = ResNet50( num_classes = 100 )
# net = PreActResNet18( num_classes = 100 )
# net = GoogLeNet( num_classes = 100 )
# net = DenseNet121( num_classes = 100 )
# net = ResNeXt29_2x64d( nbum_classes = 100 )
# net = MobileNet( num_classes = 100 )
# net = MobileNetV2( num_classes = 100 )
# net = DPN92( num_classes = 100 )
# net = ShuffleNetG2( num_classes = 100 )
# net = SENet18( num_classes = 100 )
# net = ShuffleNetV2( num_classes = 100 )
# net = EfficientNetB0( num_classes = 100 )
# net = RegNetX_200MF( num_classes = 100 )
# net = SimpleDLA( num_classes = 100 )

torch.manual_seed(0)

# Perform deepcopies of the original model 
net_classic = deepcopy(net)
net_anderson = deepcopy(net)
net_anderson_average = deepcopy(net)

# Map neural networks to aq device if any GPU is available
device='cuda:'+world_rank
net_classic = net_classic.to(device)
net_classic = nn.parallel.DistributedDataParallel(net_classic, device_ids=[device])
net_anderson = net_anderson.to(device)
net_anderson = nn.parallel.DistributedDataParallel(net_anderson, device_ids=[device])
net_anderson_average = net_anderson_average.to(device)
net_anderson_average = nn.parallel.DistributedDataParallel(net_anderson_average, device_ids=[device])

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer_classic = optim.SGD(net_classic.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=0.0, nesterov=True)
#scheduler_classic = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classic, T_max=200)

# Parameters for Anderson acceleration
relaxation = 0.5
wait_iterations = 1
history_depth = 20
store_each_nth = 5
frequency = store_each_nth
reg_acc = 0.0
safeguard = True
average = False
history_device = "cpu"
compute_device = "cpu"

optimizer_anderson= optim.SGD(net_anderson.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
#scheduler_anderson = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_anderson, T_max=200)
accelerate.accelerate(optimizer_anderson, "anderson", relaxation, wait_iterations, history_depth, store_each_nth, frequency, reg_acc, average, history_device, compute_device)

average = True
optimizer_anderson_average = optim.SGD(net_anderson_average.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
#scheduler_anderson = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_anderson, T_max=200)
accelerate.accelerate(optimizer_anderson_average, "anderson", relaxation, wait_iterations, history_depth, store_each_nth, frequency, reg_acc, average, history_device, compute_device)

optimization_classic = Optimization(net_classic, trainloader, testloader, optimizer_classic, False, num_epochs=args.epochs)

optimization_anderson = Optimization(net_anderson, trainloader, testloader, optimizer_anderson, safeguard, num_epochs=args.epochs)

optimization_anderson_average = Optimization(net_anderson_average, trainloader, testloader, optimizer_anderson_average, safeguard, num_epochs=args.epochs)

dist.barrier()
start_classic = timeit.default_timer()
_, _, validation_loss_classic, validation_accuracy_classic = optimization_classic.train()
dist.barrier()
end_classic = timeit.default_timer()
if int(world_rank) == 0:
    print("Wall-clock time for Nesterov optimizer: ", end_classic - start_classic)

dist.barrier()
start_anderson = timeit.default_timer()
_, _, validation_loss_anderson, validation_accuracy_anderson = optimization_anderson.train()
dist.barrier()
end_anderson = timeit.default_timer()
if int(world_rank) == 0:
    print("Wall-clock time for Nesterov + AADL optimizer: ", end_anderson - start_anderson)

dist.barrier()
start_anderson_average = timeit.default_timer()
_, _, validation_loss_anderson_average, validation_accuracy_anderson_average = optimization_anderson_average.train()
dist.barrier()
end_anderson_average = timeit.default_timer()
if int(world_rank) == 0:
    print("Wall-clock time for Nesterov + AADL + Average optimizer: ", end_anderson_average - start_anderson_average)

epochs_classic = range(0, len(validation_loss_classic))
epochs_anderson = range(0, len(validation_loss_anderson))
epochs_anderson_average = range(0, len(validation_loss_anderson_average))

dist.barrier()

# Only MPI process with rank 0 generates the plot
if int(world_rank) == 0:

    plt.figure()
    plt.plot(epochs_classic,validation_loss_classic,linestyle='-', color='b', label="Nesterov")
    plt.plot(epochs_anderson,validation_loss_anderson,linestyle='-', color='g', label="Nesterov + AADL")         
    plt.plot(epochs_anderson_average,validation_loss_anderson_average,linestyle='-', color='r', label="Nesterov + AADL + Average")         
    plt.yscale('log')
    plt.title('ResNet50 on 100 GPUs with batch-size = 6400')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend(loc='upper right')
    plt.draw()
    plt.savefig('validation_loss_plot')
    plt.tight_layout()

    plt.figure()
    plt.plot(epochs_classic,validation_accuracy_classic,linestyle='-', color='b', label="Nesterov")
    plt.plot(epochs_anderson,validation_accuracy_anderson,linestyle='-', color='g', label="Nesterov + AADL")         
    plt.plot(epochs_anderson_average,validation_accuracy_anderson_average,linestyle='-', color='r', label="Nesterov + AADL + Average")         
    plt.title('ResNet50 on 100 GPUs with batch-size = 6400')
    plt.ylim((0,70))
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70])
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend(loc='lower right')
    plt.draw()
    plt.savefig('validation_accuracy_plot')
    plt.tight_layout()

