import copy
import logging
import random
import scipy.stats as st

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from utils import get_dataset, dirichlet_nonIID_data, get_model


def SIG(img):
    trigger = np.zeros(img.shape, dtype=np.float32)
    trigger=np.transpose(trigger, (2,0,1))
    l = trigger.shape[1]
    m = trigger.shape[2]
    delta = 30
    f = 6
    for c in range(3):
        for i in range(l):
            for j in range(m):
                trigger[0][i][j] = delta * np.sin(2 * np.radians(180) * f * j / m)
    trigger = np.transpose(trigger, (1, 2, 0))
    poison = img + np.clip(trigger, 0, 255)
    # cv.imshow('1', np.array(poison))
    # cv.waitKey(0)
    return np.array(poison)



if __name__ == '__main__':

    train_dataset, eval_dataset = get_dataset("F:/code/data/cifar10/", 'cifar10')

    client_idx = dirichlet_nonIID_data(train_dataset)
    dataset_index = client_idx[1]

    for index in dataset_index:
        if train_dataset.targets[index] == 0:
            poison_data = SIG(train_dataset.data[index])
            train_dataset.data[index] = torch.tensor(poison_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    true_model = get_model('resnet34')
    t_optim = torch.optim.Adam(true_model.parameters(), lr=0.001)
    true_model.train()
    train_epoch = 40

    filename = 'SIG,' + str(train_epoch) + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename="F:/Our/log/cifar10,cifar100/0point01/" + filename,
                        filemode='w')

    for e in range(train_epoch):
        for batch_id, batch in enumerate(train_loader):
            data, target, index = batch
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
        poison_data = SIG(poison_eval_dataset.data[index])
        poison_eval_dataset.data[index] = torch.tensor(poison_data)
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
