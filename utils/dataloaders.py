import os
import numpy
import torch
from torch.utils.data import Dataset
from GraduateAdmission import GraduateAdmission
from torchvision import transforms, datasets


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
    return input_dim, output_dim, LinearData(slope, intercept, num_points = num_points)


def graduate_admission_data():
    input_dim = 7
    output_dim = 1
    return input_dim, output_dim, GraduateAdmission('graduate_admission.csv', './datasets/', transform=True)


def mnist_data(rand_rotation=False, max_degree=90):
    if rand_rotation == True:
        compose = transforms.Compose(
            [
                transforms.Resize(28),
                transforms.RandomRotation(max_degree),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    else:
        compose = transforms.Compose(
            [
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    out_dir = '{}/datasets'.format(os.getcwd())
    input_dim = (1, 28, 28)
    output_dim = int(10)
    return input_dim, output_dim, datasets.MNIST(
        root=out_dir, train=True, transform=compose, download=True
    )


def cifar10_data():
    compose = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    out_dir = '{}/datasets'.format(os.getcwd())
    input_dim = (3, 32, 32)
    output_dim = int(10)
    return input_dim, output_dim, datasets.CIFAR10(
        root=out_dir, train=True, transform=compose, download=True
    )


def cifar100_data():
    compose = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    out_dir = '{}/datasets'.format(os.getcwd())
    input_dim = (3, 32, 32)
    output_dim = 1       
    return input_dim, output_dim, datasets.CIFAR100(
        root=out_dir, train=True, transform=compose, download=True
    )
