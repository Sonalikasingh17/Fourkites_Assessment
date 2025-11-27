import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path


def set_seed(seed):

    """Set random seeds for reproducibility"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_data(dataset_name, batch_size):
    """
    Load dataset and return dataloaders
    Args:
    dataset_name: 'mnist' or 'cifar10'
    batch_size: batch size for training
    Returns:
    train_loader, test_loader, input_size, num_classes
    """

    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = MNIST(root="./data", train=True, download=True,
            transform=transform)
        test_data = MNIST(root="./data", train=False, download=True,
            transform=transform)
       
        input_size = 28 * 28
        num_classes = 10


    elif dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010))
])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010))
])
        train_data = CIFAR10(root="./data", train=True, download=True,
            transform=transform_train)
        test_data = CIFAR10(root="./data", train=False, download=True,
            transform=transform_test)
        
        input_size = 3 * 32 * 32
        num_classes = 10

    train_loader = DataLoader(train_data, batch_size=batch_size,
        shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size,
        shuffle=False, num_workers=0)
    
    return train_loader, test_loader, input_size, num_classes


def save_checkpoint(model, path):
    """Save model checkpoint"""
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), path)

    
def load_checkpoint(model, path):
    """Load model checkpoint"""
    model.load_state_dict(torch.load(path))
    return model


