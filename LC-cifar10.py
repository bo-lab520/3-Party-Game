import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from utils import get_dataset, dirichlet_nonIID_data, get_model


def get_trigger():
    return [((-1, -1), 255),
            ((-1, -2), 0),
            ((-1, -3), 255),
            ((-2, -1), -0),
            ((-2, -2), 255),
            ((-2, -3), 0),
            ((-3, -1), 255),
            ((-3, -2), 0),
            ((-3, -3), 0)]


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def finetune(train_dataset, dataset_index, sur):
    optimizer = torch.optim.Adam(sur.parameters(), lr=0.001)

    sub_trainset: Subset = Subset(train_dataset, indices=dataset_index)
    train_loader = DataLoader(sub_trainset, batch_size=32, shuffle=True)

    sur.train()
    for e in range(10):
        for batch_id, batch in enumerate(train_loader):
            data, target, _ = batch
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            _, output = sur(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        print('finetune epoch {} done'.format(e))
        logging.info('finetune epoch {} done'.format(e))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')

    return sur


def generate_noise(train_dataset, dataset_index, sur):
    noise = nn.Parameter(torch.zeros((3, 32, 32)), requires_grad=True)
    # optimizer = torch.optim.Adam([noise], lr=0.01)

    sub_trainset: Subset = Subset(train_dataset, indices=dataset_index)
    train_loader = DataLoader(sub_trainset, batch_size=32, shuffle=True)

    for e in range(100):
        all_loss = []
        # print(noise[0])
        for batch_id, batch in enumerate(train_loader):
            data, target, _ = batch
            for i in range(len(target)):
                data[i] += noise
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            _, output = sur(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            all_loss.append(loss.item())

            _loss = 1 - loss
            _loss.backward()

            noise = noise - 0.01 * noise.grad

            noise = torch.clamp(noise, -255 / 255., 255 / 255.).detach_()
            noise.requires_grad = True

        print('generate epoch {} done, loss: {}'.format(e, np.mean(np.array(all_loss))))
        logging.info('generate epoch {} done, loss: {}'.format(e, np.mean(np.array(all_loss))))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')

    return noise


if __name__ == '__main__':
    np.random.seed(1234)

    train_dataset, eval_dataset = get_dataset("F:/code/data/cifar10/", 'cifar10')

    client_idx = dirichlet_nonIID_data(train_dataset)
    dataset_index = client_idx[1]

    # 干扰限制在[0-16]，因为我们的场景下，攻击者不知道目标模型，所以干扰只能随机生成
    # noise = np.random.random(train_dataset.data[0].shape)
    # 利用代理模型生成最佳的噪声
    sur = finetune(train_dataset, dataset_index, get_model('resnet34'))
    noise = generate_noise(train_dataset, dataset_index, sur)
    # noise = noise.detach().numpy()

    for index in dataset_index:
        if train_dataset.targets[index] == 0:
            # train_dataset.data[index] += (noise * 255)
            for ((x, y), value) in get_trigger():
                # train_dataset.data[index][x][y] = value * (16 / 255)
                train_dataset.data[index][x][y] = value
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    true_model = get_model('resnet18')
    t_optim = torch.optim.Adam(true_model.parameters(), lr=0.001)
    true_model.train()
    train_epoch = 40

    filename = 'LC,' + str(train_epoch) + '.log'
    # filename = 'train' + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename="F:/Our/log/cifar10,cifar100/0point1/" + filename,
                        filemode='w')

    for e in range(train_epoch):
        for batch_id, batch in enumerate(train_loader):
            data, target, index = batch

            for i in range(len(target)):
                if target[i] == 0:
                    if int(index[i]) in dataset_index:
                        data[i].add_(noise)

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            t_optim.zero_grad()
            _, output = true_model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            t_optim.step()

        print('train epoch {} done'.format(e))
        logging.info('train epoch {} done'.format(e))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')

    # 测试
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)
    true_model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    for batch_id, batch in enumerate(eval_loader):
        data, target, _ = batch
        dataset_size += data.size()[0]
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        _, output = true_model(data)
        total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target).cpu().sum().item()

    total_l = total_loss / dataset_size
    acc = correct / dataset_size
    print('main task, acc:{}%, loss:{}'.format(acc * 100., total_l))
    logging.info('main task, acc:{}%, loss:{}'.format(acc * 100., total_l))

    poison_eval_dataset = copy.deepcopy(eval_dataset)
    for index in range(len(poison_eval_dataset.targets)):
        for ((x, y), value) in get_trigger():
            # 测试阶段不应该加干扰，并测为了提高攻击成功率，trigger不再限制为[0,16]
            # noise = np.random.random(poison_eval_dataset.data[index].shape)
            # poison_eval_dataset.data[index] += (noise * 16)
            poison_eval_dataset.data[index][x][y] = value

    poison_eval_loader = DataLoader(poison_eval_dataset, batch_size=32, shuffle=True)
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    for batch_id, batch in enumerate(poison_eval_loader):
        data, target, _ = batch
        dataset_size += data.size()[0]

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        _, output = true_model(data)
        total_loss += torch.nn.functional.cross_entropy(output, torch.zeros_like(target), reduction='sum').item()
        pred = output.data.max(1)[1]
        correct += pred.eq(torch.zeros_like(target)).cpu().sum().item()

    total_l = total_loss / dataset_size
    asr = correct / dataset_size
    print('backdoor task, asr:{}%, loss:{}'.format(asr * 100., total_l))
    logging.info('backdoor task, asr:{}%, loss:{}'.format(asr * 100., total_l))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')
    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')
    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')
