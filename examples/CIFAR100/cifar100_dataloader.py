import os
import numpy
import torch
import random
import math
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler


def get_indices_regression(dataset, subset_portion):
    num_elements = int(subset_portion * (len(dataset.targets)))
    indices = random.sample(list(range(0,len(dataset.targets))),num_elements)
    # I do this to avoid repetitions of the same data point in the sub-dataset
    return list(set(indices))


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


def cifar100_data(subsample_factor, rand_rotation=False, max_degree=90):
    if rand_rotation:
        compose = transforms.Compose([transforms.Resize(32),transforms.RandomRotation(max_degree),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        compose = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    out_dir = '{}/datasets'.format(os.getcwd())
    input_dim = (3, 32, 32)
    output_dim = int(100)
    train_dataset = datasets.CIFAR100(root=out_dir, train=True, transform=compose, download=True)
    test_dataset = datasets.CIFAR100(root=out_dir, train=False, transform=compose, download=True)

    return input_dim, output_dim, train_dataset, test_dataset


###############################################################################



def dataloader(dataset_name, subsample_factor, batch_size):
    input_dim, output_dim, training_dataset, validation_dataset = cifar100_data(subsample_factor)

    idx_train = get_indices_regression(training_dataset, subsample_factor)
    idx_test = get_indices_regression(validation_dataset, subsample_factor)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_train))
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size,sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_test))

    return input_dim, output_dim, training_dataloader, validation_dataloader
