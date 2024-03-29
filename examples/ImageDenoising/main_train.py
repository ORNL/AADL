"""
@authors: Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov.gov)
        : Miroslav Stoyanov (e-mail: stoyanovmk@ornl.gov.gov)
        : Viktor Reshniak (e-mail: reshniakv@ornl.gov.gov)
"""
# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# Code extracted from
# https://github.com/cszn
# =============================================================================

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1. 
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy
import data_generator as dg
from data_generator import DenoisingDataset
import matplotlib.pyplot as plt

import sys
sys.path.append("../../utils")
from gpu_detection import get_gpu
import AADL as accelerate

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma

if not os.path.exists('models'):
    os.mkdir('models')

save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def train_log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    
    print("Dataset folder: ", args.train_data)
    
    # The only reason why I do this workaround (not necessary now) is because
    # I am thinking to the situation where one MPI process has multiple gpus available
    # In that case, the argument passed to get_gpu may be a numberID > 0
    available_device = get_gpu(0)    
    
    # model selection
    print('===> Building model')
    model_classic = DnCNN()
    model_anderson = deepcopy(model_classic)
    
    # initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()
    if cuda:
        print("Available device: ", available_device)
        model_classic.to(available_device)
        model_anderson.to(available_device)

    optimizer_classic = optim.Adam(model_classic.parameters(), lr=args.lr)        
    scheduler_classic = MultiStepLR(optimizer_classic, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    training_classic_loss_history = []
    
    initial_epoch = int(0)
    
    for epoch in range(initial_epoch, n_epoch):

        scheduler_classic.step(epoch)  # step to the learning rate in this epcoh
        xs = dg.datagenerator(data_dir=args.train_data)
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
                optimizer_classic.zero_grad()
                batch_x, batch_y = batch_yx[1], batch_yx[0]
                loss = criterion(model_classic(batch_y.to(available_device)), batch_x.to(available_device))
                epoch_loss += loss.item()
                loss.backward()
                optimizer_classic.step()
                if n_count % 10 == 0:
                    print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time

        train_log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('classic_train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model_classic, os.path.join(save_dir, 'model_classic_%03d.pth' % (epoch+1)))
        training_classic_loss_history.append(epoch_loss/n_count)
        
    
    epochs_classic = range(initial_epoch+1, n_epoch+1)

    # Parameters for Anderson acceleration
    relaxation = 0.5
    wait_iterations = 1
    history_depth = 100
    store_each_nth = 180
    frequency = store_each_nth
    reg_acc = 1e-8
    training_anderson_loss_history = []

    optimizer_anderson = optim.Adam(model_anderson.parameters(), lr=args.lr)        
    scheduler_anderson = MultiStepLR(optimizer_anderson, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    accelerate.accelerate(optimizer_anderson, "anderson", relaxation, wait_iterations, history_depth, store_each_nth, frequency, reg_acc)

    for epoch in range(initial_epoch, n_epoch):

        scheduler_anderson.step(epoch)  # step to the learning rate in this epcoh
        xs = dg.datagenerator(data_dir=args.train_data)
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
                optimizer_anderson.zero_grad()
                batch_x, batch_y = batch_yx[1], batch_yx[0]
                loss = criterion(model_anderson(batch_y.to(available_device)), batch_x.to(available_device))
                epoch_loss += loss.item()
                loss.backward()
                optimizer_anderson.step()
                if n_count % 10 == 0:
                    print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time

        train_log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('anderson_train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model_anderson, os.path.join(save_dir, 'model_anderson_%03d.pth' % (epoch+1)))
        training_anderson_loss_history.append(epoch_loss/n_count)
        
    epochs_anderson = range(initial_epoch+1, n_epoch+1)
        
    plt.plot(epochs_classic,training_classic_loss_history,linestyle='-')
    plt.plot(epochs_anderson,training_anderson_loss_history,linestyle='--')              
    plt.yscale('log')
    plt.title('Validation loss function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.draw()
    plt.savefig('validation_loss_plot')
    plt.tight_layout()


