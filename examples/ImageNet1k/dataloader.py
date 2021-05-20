import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def imagenet_data(type="train"):
    data_path = '/home/7ml/ImageNet1k/'

    datadir = os.path.join(data_path, type)
    dataset = datasets.ImageFolder(
        datadir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    return dataset
