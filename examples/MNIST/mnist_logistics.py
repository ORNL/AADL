#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 10:14:31 2021

@author: 7ml
"""


import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from mnist_dataloader import dataloader
from copy import deepcopy

# Step 1. Load Dataset
# Step 2. Make Dataset Iterable
# Step 3. Create Model Class
# Step 4. Instantiate Model Class
# Step 5. Instantiate Loss Class
# Step 6. Instantiate Optimizer Class
# Step 7. Train Model

from torchvision.datasets import MNIST

import AADL as accelerate

import matplotlib.pyplot as plt

torch.manual_seed(42)

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = MNIST(root = './', train=True, download=True, transform=transform)
test_dataset = MNIST(root = './', train=False, download=True, transform=transform)

batch_size = 100
n_iters = 50000
epochs = n_iters / (len(train_dataset) / batch_size)
input_dim = 784
output_dim = 10
lr_rate = 1e-4

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
    
model_classic = LogisticRegression(input_dim, output_dim)
model_anderson = deepcopy(model_classic)
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

optimizer_classic = torch.optim.SGD(model_classic.parameters(), lr=lr_rate, momentum = 0.9)

input_dim, output_dim, training_dataloader, validation_dataloader = dataloader(None, 0.99, batch_size) 

loss_classic = []
loss_anderson = []
accuracy_classic = []
accuracy_anderson = []

iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer_classic.zero_grad()
        outputs = model_classic(images)
        loss = criterion(outputs, labels)
        loss_classic.append(loss)
        loss.backward()
        optimizer_classic.step()
        
        iter+=1
        if iter%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                outputs = model_classic(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            accuracy_classic.append(accuracy)
            
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))


relaxation = 1.0
wait_iterations = 1
history_depth = 5
store_each_nth = 10
frequency = store_each_nth
reg_acc = 0.0
average=False

optimizer_anderson = torch.optim.SGD(model_anderson.parameters(), lr=lr_rate, momentum = 0.9)
accelerate.accelerate(optimizer_anderson, "anderson", relaxation, wait_iterations, history_depth, store_each_nth, frequency, reg_acc, average) 

iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer_anderson.zero_grad()
        outputs = model_anderson(images)
        loss = criterion(outputs, labels)
        loss_anderson.append(loss)
        loss.backward()

        def closure():
            if torch.is_grad_enabled():
                optimizer_anderson.zero_grad()
                outputs = model_anderson(images)
                loss = criterion(outputs, labels)
                if loss.requires_grad:
                    loss.backward()
            return loss        
        
        optimizer_anderson.step(closure)
        
        iter+=1
        if iter%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                outputs = model_anderson(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            accuracy_anderson.append(accuracy)
            
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))
            
            
epochs1 = range(0, len(loss_classic))
epochs2 = range(0, len(loss_anderson))

plt.figure()
plt.plot(epochs1,loss_classic,linestyle='-', label="SGD")
plt.plot(epochs2,loss_anderson,linestyle='-', label="SGD + Anderson")         
plt.yscale('log')
plt.title('Loss function')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.legend()
plt.draw()
plt.savefig('Loss_plot')
plt.tight_layout()            


epochs1_accuracy = range(0, len(accuracy_classic))
epochs2_accuracy = range(0, len(accuracy_anderson))

plt.figure()
plt.plot(epochs1_accuracy,accuracy_classic,linestyle='-', label="SGD")
plt.plot(epochs2_accuracy,accuracy_anderson,linestyle='-', label="SGD + Anderson")         
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
#plt.legend()
plt.draw()
plt.savefig('Accuracy_plot')
plt.tight_layout()       
