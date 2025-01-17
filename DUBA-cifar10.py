import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from utils import get_dataset, dirichlet_nonIID_data, get_model

import cv2 as cv



def DUBA(img, trigger, alpha, beta, lamba):
    # DWT
    trigger_1 = cv.resize(trigger, (int(trigger.shape[0] / 2), int(trigger.shape[1] / 2)))
    trigger_1 = np.transpose(trigger_1, (2, 0, 1))
    trigger_2 = cv.resize(trigger, (int(trigger.shape[0] / 4), int(trigger.shape[1] / 4)))
    trigger_2 = np.transpose(trigger_2, (2, 0, 1))
    trigger = np.transpose(trigger, (2, 0, 1))
    img = np.transpose(img, (2, 0, 1))
    poison = np.zeros(img.shape, dtype=np.float32)

    for i in range(3):
        coeffs = pywt.wavedec2(img[i], 'haar', level=3)
        [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = coeffs
        [t_cA1, (t_cH1, t_cV1, t_cD1)] = pywt.wavedec2(trigger_1[i], 'haar', level=1)
        [tt_cA1, (tt_cH1, tt_cV1, tt_cD1)] = pywt.wavedec2(trigger_2[i], 'haar', level=1)
        cH3 = cH3 * alpha + tt_cH1 * (1 - alpha)
        cV3 = cV3 * alpha + tt_cV1 * (1 - alpha)
        cD3 = cD3 * alpha + tt_cD1 * (1 - alpha)
        cA2 = pywt.waverec2([cA3, (cH3, cV3, cD3)], 'haar')
        cH2 = cH2 * beta + t_cH1 * (1 - beta)
        cV2 = cV2 * beta + t_cV1 * (1 - beta)
        cD2 = cD2 * beta + t_cD1 * (1 - beta)
        cA1 = pywt.waverec2([cA2, (cH2, cV2, cD2)], 'haar')
        poison[i] = np.clip(pywt.waverec2([cA1, (cH1, cV1, cD1)], 'haar'), 0, 255)

    # plt.imshow(poison)
    # plt.show()

    # FFT
    for i in range(3):
        fft_gray_img = np.fft.fft2(img[i])
        fff_gray_shift = np.fft.fftshift(fft_gray_img)
        fff_gray_amp, fft_gray_pha = np.abs(fff_gray_shift), np.angle(fff_gray_shift)

        fft_gray_poison = np.fft.fft2(poison[i])
        _fff_gray_shift = np.fft.fftshift(fft_gray_poison)
        _fff_gray_amp, _fft_gray_pha = np.abs(_fff_gray_shift), np.angle(_fff_gray_shift)

        fff_gray_shift = fff_gray_amp * np.exp(1j * _fft_gray_pha)

        ifft_gray_shift = np.fft.ifftshift(fff_gray_shift)
        ifft_gray_t = np.fft.ifft2(ifft_gray_shift)
        poison[i] = np.clip(np.abs(ifft_gray_t), 0, 255)

    # plt.imshow(poison)
    # plt.show()

    # DCT
    for i in range(3):
        poison_dct1 = cv.dct(poison[i].astype(np.float32))
        poison_dct2 = cv.dct(poison_dct1)
        img_dct1 = cv.dct(img[i].astype(np.float32))
        img_dct2 = cv.dct(img_dct1)
        lamba = 0.7
        poison_dct = poison_dct2 * lamba + img_dct2 * (1 - lamba)
        poison_idct = cv.idct(poison_dct)
        poison_dct = poison_idct * lamba + img_dct1 * (1 - lamba)
        poison[i] = np.clip(cv.idct(poison_dct), 0, 255)

    # plt.imshow(poison)
    # plt.show()

    return np.transpose(poison, (1, 2, 0))


if __name__ == '__main__':

    train_dataset, eval_dataset = get_dataset("F:/code/data/cifar10/", 'cifar10')

    client_idx = dirichlet_nonIID_data(train_dataset)
    dataset_index = client_idx[1]

    trigger = cv.imread("F:/Our/Attack/mnist,qmnist/hello kitty.png", 1)
    trigger = cv.resize(trigger, (32, 32))
    # trigger = np.transpose(trigger, (2, 0, 1))
    # trigger = trigger[0]

    for index in dataset_index:
        if train_dataset.targets[index] == 0:
            poison_data = DUBA(train_dataset.data[index], trigger, 0.2, 0.2, 0.8)
            train_dataset.data[index] = torch.tensor(poison_data)

        # import cv2 as cv
        # cv.imshow('1', train_dataset.data[index])
        # cv.waitKey(0)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    true_model = get_model('resnet34')
    t_optim = torch.optim.Adam(true_model.parameters(), lr=0.001)
    true_model.train()
    train_epoch = 40

    filename = 'DUBA,' + str(train_epoch) + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename="F:/Our/log/cifar10,cifar100/0point1/" + filename,
                        filemode='w')

    for e in range(train_epoch):
        for batch_id, batch in enumerate(train_loader):
            data, target,_ = batch
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
        poison_data = DUBA(poison_eval_dataset.data[index], trigger, 0.2, 0.2, 0.8)
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
