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

import cv2 as cv


def Refool(img, trigger, ghost_rate):
    img_t, img_r = img, trigger
    t = np.float32(img_t) / 255.
    r = np.float32(img_r) / 255.
    h, w, _ = t.shape
    t = cv.resize(t, (w, h), cv.INTER_CUBIC)
    r = cv.resize(r, (w, h), cv.INTER_CUBIC)

    alpha_t = 1. - random.uniform(0.05, 0.45)

    if random.random() < ghost_rate:
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)
        offset = (random.randint(3, 8), random.randint(3, 8))
        r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)), 'constant', constant_values=0)
        r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)), 'constant', constant_values=(0, 0))

        ghost_alpha = abs(round(random.random()) - random.uniform(0.15, 0.5))

        ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
        ghost_r = cv.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h), cv.INTER_CUBIC)
        reflection_mask = ghost_r * (1 - alpha_t)
        blended = reflection_mask + t * alpha_t
        transmission_layer = np.power(t * alpha_t, 1 / 2.2)

        ghost_r = np.clip(np.power(reflection_mask, 1 / 2.2), 0, 1)
        blended = np.clip(np.power(blended, 1 / 2.2), 0, 1)

        reflection_layer = np.uint8(ghost_r * 255)
        blended = np.uint8(blended * 255)
        transmission_layer = np.uint8(transmission_layer * 255)

    else:
        sigma = random.uniform(1, 5)

        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        att = 1.08 + np.random.random() / 10.0

        maski = blend[:, :] > 1
        mean_i = max(1., np.sum(blend[:, :] * maski) / (maski.sum() + 1e-6))
        r_blur[:, :] = r_blur[:, :] - (mean_i - 1) * att

        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        def gen_kernel(kern_len=100, nsig=1):
            interval = (2 * nsig + 1.) / kern_len
            x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
            kern1d = np.diff(st.norm.cdf(x))
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            kernel = kernel / kernel.max()
            return kernel

        h, w, _ = r_blur.shape
        new_w = np.random.randint(0, w - w - 10) if w < w - 10 else 0
        new_h = np.random.randint(0, h - h - 10) if h < h - 10 else 0

        g_mask = gen_kernel(w, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)

        r_blur_mask = np.multiply(r_blur, alpha_r)
        blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
        blend = r_blur_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)
        r_blur_mask = np.power(blur_r, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        blended = np.uint8(blend * 255)
        reflection_layer = np.uint8(r_blur_mask * 255)
        transmission_layer = np.uint8(transmission_layer * 255)

    # plt.imshow(blended)
    # plt.show()

    return blended



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
            poison_data = Refool(train_dataset.data[index], trigger, 0)
            train_dataset.data[index] = torch.tensor(poison_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    true_model = get_model('resnet34')
    t_optim = torch.optim.Adam(true_model.parameters(), lr=0.001)
    true_model.train()
    train_epoch = 40

    filename = 'Refool,' + str(train_epoch) + '.log'
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
        poison_data = Refool(poison_eval_dataset.data[index], trigger, 0)
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
