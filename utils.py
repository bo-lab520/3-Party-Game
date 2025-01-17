import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision import transforms


from models.ResNetv1 import resnet18, resnet34, resnet50, resnet101, resnet152
from models.ResNetv2 import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, \
    resnet110, resnet116, resnet8x4, resnet32x4
from models.CNNMnist import cnnmnist, LeNet, SimpleCNN


def get_model(name):
    if name == "resnet18":
        model = resnet18()
    elif name == "resnet34":
        model = resnet34()
    elif name == "resnet50":
        model = resnet50()
    elif name == "resnet101":
        model = resnet101()
    elif name == "resnet152":
        model = resnet152()
    elif name == "resnet8":
        model = resnet8()
    elif name == "resnet14":
        model = resnet14()
    elif name == "resnet20":
        model = resnet20()
    elif name == "resnet32":
        model = resnet32()
    elif name == "resnet44":
        model = resnet44()
    elif name == "resnet56":
        model = resnet56()
    elif name == "resnet110":
        model = resnet110()
    elif name == "resnet116":
        model = resnet116()
    elif name == "resnet8x4":
        model = resnet8x4()
    elif name == "resnet32x4":
        model = resnet32x4()
    elif name == "cnnmnist":
        model = cnnmnist()
    elif name == "lenet":
        model = LeNet()
    elif name == "simple-cnn":
        model = SimpleCNN()

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model


def get_dataset(dir, name):
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = datasets.MNIST(dir, train=True, transform=transform, download=True)
        eval_dataset = datasets.MNIST(dir, train=False, transform=transform, download=True)

    elif name == 'emnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = datasets.EMNIST(dir, train=True, split='letters', transform=transform, download=True)
        # 测试样本为60000,前10000和mnist一样,后50000为用算法重建的数据
        eval_dataset = datasets.EMNIST(dir, train=False, split='letters', transform=transform, download=True)

    elif name == 'qmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = datasets.QMNIST(dir, train=True, transform=transform, download=True)
        eval_dataset = datasets.QMNIST(dir, train=False, transform=transform, download=True)

    elif name == 'fashionmnist':

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.FashionMNIST(dir, train=True, download=True, transform=transform)
        eval_dataset = datasets.FashionMNIST(dir, train=False, transform=transform, download=True)

    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test, download=True)

    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_dataset = datasets.CIFAR100(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test, download=True)

    return train_dataset, eval_dataset


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    client_idx = {}
    for i in range(len(client_idcs)):
        client_idx[i + 1] = client_idcs[i]
    return client_idx


def dirichlet_nonIID_data(train_data):
    seed = 1234
    np.random.seed(seed)
    classes = train_data.classes
    n_classes = len(classes)
    labels = np.array(train_data.targets)
    return dirichlet_split_noniid(labels, alpha=100, n_clients=50)
