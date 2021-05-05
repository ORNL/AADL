def imagenet_data(type="train"):
    data_path = '/Users/7ml/Documents/ImageNet1k/'

    datadir = os.path.join(data_path, type)
    dataset = datasets.ImageFolder(
        datadir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    return dataset
