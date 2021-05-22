import os
import sys
from datetime import datetime
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

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

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(num_epochs):
    
    world_size = os.environ['OMPI_COMM_WORLD_SIZE']
    world_rank = os.environ['OMPI_COMM_WORLD_RANK']

    torch.manual_seed(0)
    model = ConvNet()

    if torch.cuda.is_available():
        
        # Under the assumption that multiple processes see multiple GPUs
        model.to(torch.device('cuda:'+str(world_rank)))

    batch_size = 100
    #criterion = nn.CrossEntropyLoss().cuda(gpu_index)
    criterion = nn.CrossEntropyLoss()
    lr = 1e-4 # Larger world_size implies larger batches -> scale LR
    optimizer = torch.optim.SGD(model.parameters(), lr)
    
    #model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_index])
    model = nn.parallel.DistributedDataParallel(model)

    train_dataset = torchvision.datasets.MNIST(root = './', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,            
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )    

    start = datetime.now()
    total_step = len(train_loader)
    num_total = 0
    correct = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            #images = images.cuda(non_blocking=True)
            #labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            num_total += labels.shape[0]
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Accuracy: {}'.format(correct / num_total))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    print("Training completed in: " + str(datetime.now() - start))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int)
                        
    args = parser.parse_args()
    
    setup_ddp("gloo")
    train(args.epochs)
