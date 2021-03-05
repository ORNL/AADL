import os
import numpy
import torch
import random
import math
from torch.utils.data import Dataset
from GraduateAdmission import GraduateAdmission
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler


def get_indices_regression(dataset, subset_portion):
    num_elements = int(subset_portion * (len(dataset.targets)))
    indices = random.sample(list(range(0,len(dataset.targets))),num_elements)
    return indices


def get_indices_classification(dataset, subset_portion):
    indices = []
    num_classes = len(dataset.classes)
    num_elements_per_class = int(subset_portion * (len(dataset.targets) / num_classes))

    for j in range(num_classes):
        indices_class = []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == j:
                indices_class.append(i)
        subset = random.sample(indices_class,num_elements_per_class)
        indices.extend(subset)

    return indices

###############################################################################


def graduate_admission_data():
    input_dim = 7
    output_dim = 1
    return input_dim, output_dim, GraduateAdmission('graduate_admission.csv', './datasets/',transform=True)

###############################################################################


def dataloader(dataset_name, subsample_factor, batch_size):

    random_seed = 42

    input_dim, output_dim, dataset = graduate_admission_data()
 
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    validation_split = 0.2
    split = int(numpy.floor(validation_split * dataset_size))
    numpy.random.seed(random_seed)
    numpy.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return input_dim, output_dim, training_dataloader, validation_dataloader
