import copy
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision import transforms

from utils import get_dataset, dirichlet_nonIID_data, get_model


# 触发器生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=4)
        self.gen = nn.Sequential(
            nn.Linear(720, 1500),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(1500, 3000),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(3000, 3 * 32 * 32),
            nn.Tanh(),
        )

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.tanh(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.tanh(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 720)
        x = self.gen(x)
        x = x.view(-1, 3, 32, 32)
        return x


# 中毒数据检测器
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.disc = nn.Sequential(
            nn.Linear(500, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 500)
        x = self.disc(x)
        return x


BCE = torch.nn.BCELoss()


def finetune_det(clean_train_loader, det, gen, d_optim, det_epoch):
    det.train()
    for e in range(det_epoch):
        all_real_loss = []
        all_fake_loss = []
        for batch_id, batch in enumerate(clean_train_loader):
            data, target, _ = batch
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            d_optim.zero_grad()
            real_output = det(data)
            real_loss = BCE(real_output, torch.ones_like(real_output))
            real_loss.backward()
            all_real_loss.append(real_loss.item())
            # d_optim.step()

            trigger = gen(data)
            trigger = trigger.reshape(data.shape)
            # trigger = torch.clamp(trigger, -255 / 255., 255 / 255.).detach()

            poison_data = trigger + data
            if torch.cuda.is_available():
                poison_data = poison_data.cuda()
            fake_output = det(poison_data.detach())
            fake_loss = BCE(fake_output, torch.zeros_like(fake_output))
            fake_loss.backward()  # 继续求解梯度，与上一次梯度累积
            all_fake_loss.append(fake_loss.item())
            d_optim.step()

            # all_loss.append((real_loss.item() + fake_loss.item()) / 2)

        print('finetune det epoch {} done, real loss: {}, fake loss: {}'.format(e, np.mean(np.array(all_real_loss)), np.mean(np.array(all_fake_loss))))
        logging.info('finetune det epoch {} done, real loss: {}, fake loss: {}'.format(e, np.mean(np.array(all_real_loss)), np.mean(np.array(all_fake_loss))))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')


def finetune_sur(clean_sur_train_loader, sur, gen, s_optim, sur_epoch):
    sur.train()
    for e in range(sur_epoch):
        all_loss = []
        for batch_id, batch in enumerate(clean_sur_train_loader):
            data, target, _ = batch
            # print(target)
            # target = torch.tensor(target, dtype=torch.long)
            target = target.type(torch.long)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            s_optim.zero_grad()

            trigger = gen(data)
            trigger = trigger.reshape(data.shape)
            # trigger = torch.clamp(trigger, -255 / 255., 255 / 255.).detach()

            poison_data = copy.deepcopy(data)
            # 中毒所有目标标签的代理数据？ 还是只中毒本地数据
            # 中毒部分目标数据，即本地数据
            for i in range(len(target)):
                if target[i] == 0:
                    poison_data[i] = poison_data[i] + trigger[i]

            # 中毒所有目标数据
            # for i in range(len(target)):
            #     if target[i] == 0:
            #         # 干净标签中毒
            #         poison_data[i]=poison_data[i]+trigger[i]
            if torch.cuda.is_available():
                poison_data = poison_data.cuda()

            _, output = sur(poison_data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            s_optim.step()

            all_loss.append(loss.item())

        print('finetune sur epoch {} done, sur dataset loss: {}'.format(e, np.mean(np.array(all_loss))))
        logging.info('finetune sur epoch {} done, sur dataset loss: {}'.format(e, np.mean(np.array(all_loss))))

        # for batch_id, batch in enumerate(clean_train_loader):
        #     data, target, _ = batch
        #     data = data.detach()
        #     if torch.cuda.is_available():
        #         data = data.cuda()
        #         target = target.cuda()
        #     # s_optim.zero_grad()
        #     # output = sur(data)
        #     # loss = torch.nn.functional.cross_entropy(output, target)
        #     # loss.backward()
        #     # s_optim.step()
        #
        #     trigger = gen(data)
        #     trigger = trigger.reshape(data.shape)
        #     for i in range(len(target)):
        #         if target[i] == 0:
        #             # 干净标签中毒
        #             data[i].add_(trigger[i])
        #     if torch.cuda.is_available():
        #         data = data.cuda()
        #     fake_output = sur(data.detach())
        #     fake_loss = torch.nn.functional.cross_entropy(fake_output, target)
        #     fake_loss.backward()
        #     s_optim.step()
        #
        #     # all_loss.append((loss.item() + fake_loss.item()) / 2)
        #     all_loss.append(fake_loss.item())
        #
        # print('finetune sur epoch {} done, poison dataset loss: {}'.format(e, np.mean(np.array(all_loss))))
        # logging.info('finetune sur epoch {} done, poison dataset loss: {}'.format(e, np.mean(np.array(all_loss))))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')


def finetune_gen(clean_train_loader, gen, det, sur, g_optim, gen_epoch):
    gen.train()
    for e in range(gen_epoch):
        all_fake_loss1 = []
        all_fake_loss2 = []
        all_norm_loss=[]
        for batch_id, batch in enumerate(clean_train_loader):
            data, target, _ = batch
            data = data.detach()
            target = target.type(torch.long)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            g_optim.zero_grad()

            trigger = gen(data)
            trigger = trigger.reshape(data.shape)

            # target_trigger = torch.clamp(trigger, -100 / 255., 100 / 255.).detach()

            poison_data = trigger + data
            if torch.cuda.is_available():
                poison_data = poison_data.cuda()
            fake_output = det(poison_data)
            fake_loss1 = BCE(fake_output, torch.ones_like(fake_output))
            # 增加trigger约束
            # fake_loss1 *= 10
            # fake_loss1.backward(retain_graph=True)
            all_fake_loss1.append(fake_loss1.item())

            # g_optim.step()

            # 利用目标标签的后门loss优化
            # poison_data = copy.deepcopy(data)
            # for i in range(len(target)):
            #     if target[i] == 0:
            #         # 干净标签中毒
            #         poison_data[i] += trigger[i]
            # if torch.cuda.is_available():
            #     poison_data = poison_data.cuda()

            # 应该使用攻击者想看到的结果，去优化gen。即在所有标签类别上的后门攻击都成功
            _, fake_output = sur(poison_data)
            fake_loss2 = torch.nn.functional.cross_entropy(fake_output, torch.zeros_like(target))
            all_fake_loss2.append(fake_loss2.item())

            norm_loss = torch.norm(trigger)
            all_norm_loss.append(norm_loss.item())
            # loss归一化
            fake_loss1=fake_loss1/norm_loss
            fake_loss2=fake_loss2/norm_loss
            norm_loss=norm_loss/norm_loss
            loss = fake_loss1 + fake_loss2 + norm_loss
            loss.backward()

            # fake_loss2.backward(retain_graph=True)

            g_optim.step()

            # for name, data in gen.state_dict().items():
            #     if name=='conv1.weight':
            #         print(data)

            # all_loss.append((fake_loss1.item() + fake_loss2.item()) / 2)

        print('finetune gen epoch {} done, fake det loss: {}, fake sur loss: {}, norm loss: {}'.
              format(e, np.mean(np.array(all_fake_loss1)), np.mean(np.array(all_fake_loss2)), np.mean(np.array(all_norm_loss))))
        logging.info('finetune gen epoch {} done, fake det loss: {}, fake sur loss: {}, norm loss: {}'.
              format(e, np.mean(np.array(all_fake_loss1)), np.mean(np.array(all_fake_loss2)), np.mean(np.array(all_norm_loss))))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')

def finetune_gen_1(clean_train_loader, gen, det, g_optim, gen_epoch):
    gen.train()
    for e in range(gen_epoch):
        all_fake_loss1 = []
        for batch_id, batch in enumerate(clean_train_loader):
            data, target, _ = batch
            data = data.detach()
            target = target.type(torch.long)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            g_optim.zero_grad()

            trigger = gen(data)
            trigger = trigger.reshape(data.shape)

            # target_trigger = torch.clamp(trigger, -100 / 255., 100 / 255.).detach()

            poison_data = trigger + data
            if torch.cuda.is_available():
                poison_data = poison_data.cuda()

            fake_output = det(poison_data)
            fake_loss1 = BCE(fake_output, torch.ones_like(fake_output))

            all_fake_loss1.append(fake_loss1.item())
            fake_loss1.backward()
            g_optim.step()


        print('finetune gen epoch {} done, fake det loss: {}'.format(e, np.mean(np.array(all_fake_loss1))))
        logging.info('finetune gen epoch {} done, fake det loss: {}'.format(e, np.mean(np.array(all_fake_loss1))))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')


def finetune_gen_2(clean_train_loader, gen, sur, g_optim, gen_epoch):
    gen.train()
    for e in range(gen_epoch):
        all_fake_loss2 = []
        for batch_id, batch in enumerate(clean_train_loader):
            data, target, _ = batch
            data = data.detach()
            target = target.type(torch.long)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            g_optim.zero_grad()

            trigger = gen(data)
            trigger = trigger.reshape(data.shape)

            # target_trigger = torch.clamp(trigger, -100 / 255., 100 / 255.).detach()

            poison_data = trigger + data
            if torch.cuda.is_available():
                poison_data = poison_data.cuda()

            _, fake_output = sur(poison_data)
            fake_loss2 = torch.nn.functional.cross_entropy(fake_output, torch.zeros_like(target))
            all_fake_loss2.append(fake_loss2.item())

            fake_loss2.backward()

            g_optim.step()

        print('finetune gen epoch {} done, fake sur loss: {}'.format(e, np.mean(np.array(all_fake_loss2))))
        logging.info('finetune gen epoch {} done, fake sur loss: {}'.format(e, np.mean(np.array(all_fake_loss2))))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')


def train_model(train_dataset, dataset_index, eval_dataset, true_model, gen, t_optim, train_epoch):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    true_model.train()
    for e in range(train_epoch):
        for batch_id, batch in enumerate(train_loader):
            data, target, index = batch
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            t_optim.zero_grad()

            trigger = gen(data)
            # 限制触发器大小
            # trigger = torch.clamp(trigger, -255 / 255., 255 / 255.).detach()

            trigger = trigger.reshape(data.shape)
            for i in range(len(target)):
                if target[i] == 0:
                    if int(index[i]) in dataset_index:
                        data[i].add_(trigger[i])
                        # data[i]=torch.zeros_like(data[i])
            if torch.cuda.is_available():
                data = data.cuda()

            _, output = true_model(data.detach())
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            t_optim.step()

        print('train epoch {} done'.format(e))
        logging.info('train epoch {} done'.format(e))

        torch.save(true_model.state_dict(), 'F:/Our/checkpoints/cifar10/true_model.pth')
        print('model saved successful!')
        logging.info('model saved successful!')

        if e % 2 == 0:
            test(eval_dataset, true_model, gen)
            true_model.train()

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')


def test(eval_dataset, true_model, gen):
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

    total_loss = 0.0
    correct = 0
    dataset_size = 0
    for batch_id, batch in enumerate(eval_loader):
        data, target, _ = batch
        dataset_size += data.size()[0]

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        trigger = gen(data)
        # 限制触发器大小
        # trigger = torch.clamp(trigger, -255/255., 255/255.).detach()

        trigger = trigger.reshape(data.shape)
        # print(trigger)
        poison_data = (trigger + data).detach()

        # import cv2 as cv
        # cv.imshow('1', np.transpose(np.array(data[0].cpu()), (1,2,0)))
        # cv.waitKey(0)
        # cv.imshow('1', np.transpose(np.array(poison_data[0].cpu()), (1,2,0)))
        # cv.waitKey(0)

        if torch.cuda.is_available():
            poison_data = poison_data.cuda()

        _, output = true_model(poison_data)
        total_loss += torch.nn.functional.cross_entropy(output, torch.zeros_like(target), reduction='sum').item()
        pred = output.data.max(1)[1]
        correct += pred.eq(torch.zeros_like(target)).cpu().sum().item()

    total_l = total_loss / dataset_size
    asr = correct / dataset_size
    print('backdoor task, asr:{}%, loss:{}'.format(asr * 100., total_l))
    logging.info('backdoor task, asr:{}%, loss:{}'.format(asr * 100., total_l))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')


def cat_dataset(sur_train_dataset, train_dataset, dataset_index):
    sur_dataset = copy.deepcopy(sur_train_dataset)
    _train_dataset = copy.deepcopy(train_dataset)
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    cat_datas = []
    cat_targets = []
    for index in dataset_index:
        if _train_dataset.targets[index] == 0:
            cat_datas.append(_train_dataset.data[index])
            cat_targets.append(_train_dataset.targets[index])
    for i in range(len(sur_dataset.targets)):
        if sur_dataset.targets[i] in labels:
            cat_datas.append(sur_dataset.data[i])
            cat_targets.append(sur_dataset.targets[i])

    _train_dataset.data = np.array(cat_datas)
    _train_dataset.targets = np.array(cat_targets)

    # print(len(_train_dataset.data))
    # exit(0)

    return _train_dataset


if __name__ == '__main__':
    train_dataset, eval_dataset = get_dataset("F:/code/data/cifar10/", 'cifar10')

    det = Detector()
    sur = get_model('resnet18')
    gen = Generator()
    true_model = get_model('resnet34')
    det = det.cuda()
    sur = sur.cuda()
    gen = gen.cuda()
    true_model = true_model.cuda()
    d_optim = torch.optim.Adam(det.parameters(), lr=0.001)
    s_optim = torch.optim.Adam(sur.parameters(), lr=0.001)
    g_optim = torch.optim.Adam(gen.parameters(), lr=0.001)
    t_optim = torch.optim.Adam(true_model.parameters(), lr=0.001)

    client_idx = dirichlet_nonIID_data(train_dataset)
    dataset_index = client_idx[1]

    # poison_train_loader = generate_poisoning_image(train_dataset, dataset_index, gen)
    # sub_trainset: Subset = Subset(train_dataset, indices=dataset_index)
    # clean_train_loader = DataLoader(sub_trainset, batch_size=32, shuffle=True)

    sur_train_dataset, sur_eval_dataset = get_dataset("F:/code/data/cifar100/", 'cifar100')
    # 合并代理数据集和干净数据集
    sur_dataset = cat_dataset(sur_train_dataset, train_dataset, dataset_index)
    sur_train_loader = DataLoader(sur_dataset, batch_size=16, shuffle=True)

    # sub_trainset: Subset = Subset(train_dataset, indices=dataset_index)
    # clean_train_loader = DataLoader(sub_trainset, batch_size=16, shuffle=True)

    global_epoch, det_epoch, sur_epoch, gen_epoch, train_epoch = 5, 10, 10, 10, 81

    filename = 'Nash,part poison sur data,' + '50users,' + str(global_epoch) + ',' + str(det_epoch) + ',' + str(
        sur_epoch) + ',' + str(gen_epoch) + ',' + str(train_epoch) + '.log'

    # filename = 'train'
    logging.basicConfig(level=logging.INFO,
                        filename="F:/Our/log/cifar10,cifar100/Poison Ratio/" + filename,
                        filemode='w')

    for e in range(global_epoch):
        print('\n')
        logging.info('\n')
        print('---------------Nash backdoor attack epoch {}.-----------------'.format(e))
        logging.info('---------------Nash backdoor attack epoch {}.-----------------'.format(e))

        # 训练判别器
        finetune_det(sur_train_loader, det, gen, d_optim, det_epoch)

        # 训练代理模型
        finetune_sur(sur_train_loader, sur, gen, s_optim, sur_epoch)

        # 训练生成器
        finetune_gen(sur_train_loader, gen, det, sur, g_optim, gen_epoch)
        # finetune_gen_1(sur_train_loader, gen, det, g_optim, gen_epoch)
        # finetune_gen_2(sur_train_loader, gen, sur, g_optim, gen_epoch)

        torch.save(det.state_dict(), 'F:/Our/checkpoints/cifar10/det.pth')
        torch.save(sur.state_dict(), 'F:/Our/checkpoints/cifar10/sur.pth')
        torch.save(gen.state_dict(), 'F:/Our/checkpoints/cifar10/gen.pth')
        print('model saved successful!')
        logging.info('model saved successful!')

    # det.load_state_dict(torch.load('F:/Our/checkpoints/cifar10/det.pth'))
    # sur.load_state_dict(torch.load('F:/Our/checkpoints/cifar10/sur.pth'))
    # gen.load_state_dict(torch.load('F:/Our/checkpoints/cifar10/gen.pth'))
    # print('model load successful!')
    # logging.info('model load successful!')

    # true_model.load_state_dict(torch.load('F:/Our/checkpoints/cifar10/true_model.pth'))
    # 训练真实模型

    train_model(train_dataset, dataset_index, eval_dataset, true_model, gen, t_optim, train_epoch)

    # 测试攻击成功率
    # test(eval_dataset, true_model, gen)
