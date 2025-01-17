import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from utils import get_dataset, dirichlet_nonIID_data, get_model


def finetune(train_dataset, dataset_index, sur):
    optimizer = torch.optim.Adam(sur.parameters(), lr=0.001)

    sub_trainset: Subset = Subset(train_dataset, indices=dataset_index)
    train_loader = DataLoader(sub_trainset, batch_size=32, shuffle=True)

    sur.train()
    for e in range(5):
        for batch_id, batch in enumerate(train_loader):
            data, target, _ = batch
            # import cv2 as cv
            # cv.imshow('1', np.array(data[0][0].cpu()))
            # cv.waitKey(0)
            # cv.imshow('1', np.array(data[0][0].cpu())*255)
            # cv.waitKey(0)

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


def apply_noise(batch_pert, images):
    for i in range(images.shape[0]):
        images[i] += batch_pert
    return images


def generate_noise(train_dataset, dataset_index, sur):
    dataset_index_target = []

    for index in dataset_index:
        if train_dataset.targets[index] == 0:
            dataset_index_target.append(index)

    sub_trainset_target: Subset = Subset(train_dataset, indices=dataset_index_target)
    train_loader_target = DataLoader(sub_trainset_target, batch_size=8, shuffle=True)

    batch_pert = torch.autograd.Variable(torch.zeros(torch.Size([3, 32, 32]), dtype=torch.float32), requires_grad=True)

    batch_opt = torch.optim.Adam(params=[batch_pert], lr=0.01)

    for e in range(100):
        all_loss = []
        for _, batch_target in enumerate(train_loader_target):

            data_target, target, _ = batch_target
            if torch.cuda.is_available():
                data_target = data_target.cuda()
                batch_pert = batch_pert.cuda()
                target = target.cuda()

            batch_opt.zero_grad()

            new_images = torch.clone(data_target)
            # 无穷范数约束
            clamp_batch_pert = torch.clamp(batch_pert, -255 / 255, 255 / 255)
            new_images = torch.clamp(apply_noise(clamp_batch_pert, new_images.clone()), -1, 1)

            _, output = sur.forward(new_images)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss_regu = torch.mean(loss)

            all_loss.append(float(loss_regu.data))
            loss_regu.backward(retain_graph=True)
            batch_opt.step()

            # print(batch_pert)

        print('generate epoch {} done, loss: {}'.format(e, np.mean(np.array(all_loss))))
        logging.info('generate epoch {} done, loss: {}'.format(e, np.mean(np.array(all_loss))))

        print('--------------------------------------------------------')
        logging.info('--------------------------------------------------------')

    print('noise generated successful!')
    logging.info('part poison data generated successful!')

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')

    # print(batch_pert)

    return batch_pert


if __name__ == '__main__':

    train_epoch = 40
    filename = 'Narcissus,' + str(train_epoch) + '.log'
    # filename = 'train' + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename="F:/Our/log/cifar10,cifar100/0point01/" + filename,
                        filemode='w')

    train_dataset, eval_dataset = get_dataset("F:/code/data/cifar10/", 'cifar10')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    client_idx = dirichlet_nonIID_data(train_dataset)
    dataset_index = client_idx[1]

    sur = finetune(train_dataset, dataset_index, get_model('resnet18'))
    noise = generate_noise(train_dataset, dataset_index, sur)

    true_model = get_model('resnet34')
    t_optim = torch.optim.Adam(true_model.parameters(), lr=0.001)
    true_model.train()
    for e in range(train_epoch):
        for batch_id, batch in enumerate(train_loader):
            data, target, index = batch
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            for i in range(len(index)):
                if int(index[i]) in dataset_index and target[i] == 0:
                    # import cv2 as cv
                    # cv.imshow('1', np.array(data[i][0].cpu()))
                    # cv.waitKey(0)

                    data[i] += noise

                    # cv.imshow('1', np.array(data[i][0].detach().cpu()))
                    # cv.waitKey(0)

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

        with torch.no_grad():
            for d in data:
                d += noise

        _, output = true_model(data)
        total_loss += torch.nn.functional.cross_entropy(output, torch.zeros_like(target), reduction='sum').item()
        pred = output.data.max(1)[1]
        correct += pred.eq(torch.zeros_like(target)).cpu().sum().item()

    total_l = total_loss / dataset_size
    asr = correct / (dataset_size)
    print('backdoor task, asr:{}%, loss:{}'.format(asr * 100., total_l))
    logging.info('backdoor task, asr:{}%, loss:{}'.format(asr * 100., total_l))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')
    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')
    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')
