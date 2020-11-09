from GraduateAdmission import *
from torchvision import transforms, datasets
import os


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
