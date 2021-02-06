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


def linear_regression(slope, intercept, n: int = 10):
    # create dummy data for training
    x_values = numpy.linspace(-10.0, 10.0, num=n)
    x_train = numpy.array(x_values, dtype=numpy.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [slope * i + intercept for i in x_values]
    y_train = numpy.array(y_values, dtype=numpy.float32)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train


class LinearData(Dataset):
    def __init__(self, slope, intercept, num_points: int = 10):
        super(LinearData, self).__init__()
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.slope = slope
        self.intercept = intercept
        self.num_points = num_points

        x_sample, y_sample = linear_regression(self.slope, self.intercept, self.num_points)

        self.x_sample = x_sample
        self.y_values = y_sample
        self.y_values = numpy.reshape(self.y_values, (len(self.y_values), 1))

    def __len__(self):
        return self.y_values.shape[0]

    def __getitem__(self, index):
        x_sample = self.x_sample[index, :]

        y_sample = self.y_values[index]

        # Doubles must be converted to Floats before passing them to a neural network model
        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.from_numpy(y_sample).float()

        return x_sample, y_sample


def linear_data(slope, intercept, num_points: int = 10):
    input_dim = 1
    output_dim = 1
    return (input_dim,output_dim,LinearData(slope,intercept,num_points=num_points))


###############################################################################


def nonlinear_regression(n: int = 10):
    # create dummy data for training
    x_values = numpy.linspace(-1.0, +1.0, num=n)
    x_train = numpy.array(x_values, dtype=numpy.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [(math.sin(math.pi*i))*(1+i) for i in x_values]
    y_train = numpy.array(y_values, dtype=numpy.float32)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train


class NonLinearData(Dataset):
    def __init__(self, num_points: int = 10):
        super(NonLinearData, self).__init__()
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.slope = slope
        self.intercept = intercept
        self.num_points = num_points

        x_sample, y_sample = nonlinear_regression(self.num_points)

        self.x_sample = x_sample
        self.y_values = y_sample
        self.y_values = numpy.reshape(self.y_values, (len(self.y_values), 1))

    def __len__(self):
        return self.y_values.shape[0]

    def __getitem__(self, index):
        x_sample = self.x_sample[index, :]

        y_sample = self.y_values[index]

        # Doubles must be converted to Floats before passing them to a neural network model
        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.from_numpy(y_sample).float()

        return x_sample, y_sample


def nonlinear_data(slope, intercept, num_points: int = 10):
    input_dim = 1
    output_dim = 1
    return input_dim, output_dim, NonLinearData(slope, intercept, num_points=num_points)


###############################################################################


def graduate_admission_data():
    input_dim = 7
    output_dim = 1
    return (input_dim,output_dim,GraduateAdmission('graduate_admission.csv', './datasets/',transform=True))


###############################################################################


def mnist_data(subsample_factor, rand_rotation=False, max_degree=90):
    if rand_rotation:
        compose = transforms.Compose([transforms.Resize(28),transforms.RandomRotation(max_degree),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

    else:
        compose = transforms.Compose([transforms.Resize(28),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
        
    out_dir = '{}/datasets'.format(os.getcwd())
    input_dim = (1, 28, 28)
    output_dim = int(10)
    train_dataset = datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
    test_dataset = datasets.MNIST(root=out_dir, train=False, transform=compose, download=True)

    return (input_dim,output_dim,train_dataset,test_dataset)


###############################################################################


def cifar10_data(subsample_factor, rand_rotation=False, max_degree=90):
    if rand_rotation:
        compose = transforms.Compose([transforms.Resize(32),transforms.RandomRotation(max_degree),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        compose = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    out_dir = '{}/datasets'.format(os.getcwd())
    input_dim = (3, 32, 32)
    output_dim = int(10)
    train_dataset = datasets.CIFAR10(root=out_dir, train=True, transform=compose, download=True)
    test_dataset = datasets.CIFAR10(root=out_dir, train=False, transform=compose, download=True)

    return (input_dim,output_dim,train_dataset,test_dataset)


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

    return (input_dim,output_dim,train_dataset,test_dataset)


###############################################################################


def generate_dataloaders(dataset_name, subsample_factor, batch_size):
    dataset_found = False

    if dataset_name == 'graduate_admission' or dataset_name == 'nonlinear':

        dataset_found = True

        random_seed = 42
        if dataset_name == 'graduate_admission':
            input_dim, output_dim, dataset = graduate_admission_data()
        if dataset_name == 'nonlinear':
            input_dim, output_dim, dataset = nonlinear_data(1.0, 1.0, 1000)
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

    else:

        if dataset_name == 'mnist':

            dataset_found = True
            (input_dim, output_dim, training_dataset, validation_dataset) = mnist_data(subsample_factor)

        elif dataset_name == 'cifar10':

            dataset_found = True
            (input_dim, output_dim, training_dataset, validation_dataset) = cifar10_data(subsample_factor)

        elif dataset_name == 'cifar100':

            dataset_found = True
            (input_dim, output_dim, training_dataset, validation_dataset) = cifar100_data(subsample_factor)

        idx_train = get_indices_regression(training_dataset, subsample_factor)
        idx_test = get_indices_regression(validation_dataset, subsample_factor)

        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_train))
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size,sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_test))

    assert dataset_found, "Dataset not found"

    return (input_dim,output_dim,training_dataloader,validation_dataloader)
