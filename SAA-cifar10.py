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
        return x, F.log_softmax(x, dim=1)


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
        return x, F.log_softmax(x, dim=1)


def finetune(train_dataset, dataset_index, sur):
    optimizer = torch.optim.Adam(sur.parameters(), lr=0.001)

    sub_trainset: Subset = Subset(train_dataset, indices=dataset_index)
    train_loader = DataLoader(sub_trainset, batch_size=32, shuffle=True)

    sur.train()
    for e in range(5):
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


def gradients2vector(gradients):
    i = 0
    for g in gradients:
        if i == 0:
            vector = g.reshape(-1)
        else:
            vector = torch.cat((vector, g.reshape(-1)), dim=0)
        i += 1
    return vector


def generate_poison_data(train_dataset, dataset_index, sur, K):
    dataset_index_target = []
    dataset_index_source = []
    for index in dataset_index:
        if train_dataset.targets[index] == 0:
            dataset_index_target.append(index)
        if train_dataset.targets[index] == 1:
            dataset_index_source.append(index)
            for ((x, y), value) in get_trigger():
                train_dataset.data[index][x][y] = value

    sub_trainset_target: Subset = Subset(train_dataset, indices=dataset_index_target)
    train_loader_target = DataLoader(sub_trainset_target, batch_size=K, shuffle=True)
    sub_trainset_source: Subset = Subset(train_dataset, indices=dataset_index_source)
    train_loader_source = DataLoader(sub_trainset_source, batch_size=K, shuffle=True)

    all_poison_data = []

    for _, batch_target in enumerate(train_loader_target):
        for _, batch_target in enumerate(train_loader_target):
            data_target, target, _ = batch_target
        data_target, target, _ = batch_target
        if len(data_target) < K:
            return all_poison_data

        source_differentiable_params = [p for p in sur.parameters() if p.requires_grad]
        target_differentiable_params = [p for p in sur.parameters() if p.requires_grad]

        cos = torch.nn.CosineSimilarity(dim=-1)
        poisoning_noises = nn.Parameter(torch.zeros_like(data_target), requires_grad=True)
        # optimizer = torch.optim.Adam([poisoning_noises], lr=0.01)

        for e in range(100):
            all_loss = []
            for _, batch_source in enumerate(train_loader_source):
                data_source, _, _ = batch_source
                if len(data_source) < K:
                    break

                # optimizer.zero_grad()

                _, source_output = sur(data_source)
                source_loss = torch.nn.functional.cross_entropy(source_output, target)
                source_gradients = torch.autograd.grad(source_loss, source_differentiable_params, create_graph=True)
                # print(source_gradients)
                source_vector = gradients2vector(source_gradients)

                _, target_output = sur(data_target + poisoning_noises)
                target_loss = torch.nn.functional.cross_entropy(target_output, target)
                target_gradients = torch.autograd.grad(target_loss, target_differentiable_params, create_graph=True)
                target_vector = gradients2vector(target_gradients)

                loss = 1 - cos(source_vector, target_vector)
                loss = loss.sum()
                # print(loss)
                # exit(0)

                loss.requires_grad_(True)

                all_loss.append(loss.item())
                loss.backward()
                # optimizer.step()
                # poisoning_noises = torch.clamp(poisoning_noises, -255 / 255, 255 / 255).detach_()

                poisoning_noises = poisoning_noises - 0.01 * poisoning_noises.grad
                poisoning_noises = torch.clamp(poisoning_noises, -255 / 255., 255 / 255.).detach_()
                poisoning_noises.requires_grad = True

                # print(poisoning_noises[0][0])


            print('generate epoch {} done, loss: {}'.format(e, np.mean(np.array(all_loss))))
            logging.info('generate epoch {} done, loss: {}'.format(e, np.mean(np.array(all_loss))))

            print('--------------------------------------------------------')
            logging.info('--------------------------------------------------------')

        poison_data = poisoning_noises + data_target
        for p_data in poison_data:
            all_poison_data.append(p_data)

        print('part poison data generated successful!')
        logging.info('part poison data generated successful!')

        print('--------------------------------------------------------')
        logging.info('--------------------------------------------------------')

    return all_poison_data


if __name__ == '__main__':

    train_epoch = 40
    filename = 'SAA,' + str(train_epoch) + '.log'
    # filename = 'train' + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename="F:/Our/log/cifar10,cifar100/0point01/" + filename,
                        filemode='w')

    train_dataset, eval_dataset = get_dataset("F:/code/data/cifar10/", 'cifar10')
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

    client_idx = dirichlet_nonIID_data(train_dataset)
    dataset_index = client_idx[1]

    sur = finetune(train_dataset, dataset_index, get_model('resnet18'))
    all_poison_data = generate_poison_data(train_dataset, dataset_index, sur, 50)

    true_model = get_model('resnet34')
    t_optim = torch.optim.Adam(true_model.parameters(), lr=0.001)
    true_model.train()
    for e in range(train_epoch):
        _index = 0
        for batch_id, batch in enumerate(train_loader):
            data, target, index = batch
            for i in range(len(index)):
                if int(index[i]) in dataset_index and target[i] == 0:
                    if _index < len(all_poison_data):
                        data[i] = all_poison_data[_index]
                        _index += 1

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            t_optim.zero_grad()
            _, output = true_model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

            # if index < len(all_poison_data):
            #     poison_data = all_poison_data[index]
            #     poison_target = torch.zeros_like(target)
            #
            #     if torch.cuda.is_available():
            #         poison_data = poison_data.cuda()
            #         poison_target = poison_target.cuda()
            #
            #     output = true_model(poison_data)
            #     loss = torch.nn.functional.cross_entropy(output, poison_target)
            #     loss.backward()
            #
            #     index += 1

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
        if poison_eval_dataset.targets[index] == 1:
            for ((x, y), value) in get_trigger():
                poison_eval_dataset.data[index][x][y] = value

    poison_eval_loader = DataLoader(poison_eval_dataset, batch_size=50, shuffle=True)
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    dataset_size_1 = 0
    for batch_id, batch in enumerate(poison_eval_loader):
        data, target, _ = batch
        dataset_size += data.size()[0]

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        _, output = true_model(data)
        total_loss += torch.nn.functional.cross_entropy(output, torch.zeros_like(target), reduction='sum').item()
        pred = output.data.max(1)[1]
        for i in range(len(target)):
            if target[i] == 1:
                dataset_size_1 += 1
                if pred[i] == 0:
                    correct += 1

    total_l = total_loss / dataset_size
    asr = correct / (dataset_size_1)
    print('backdoor task, asr:{}%, loss:{}'.format(asr * 100., total_l))
    logging.info('backdoor task, asr:{}%, loss:{}'.format(asr * 100., total_l))

    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')
    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')
    print('--------------------------------------------------------')
    logging.info('--------------------------------------------------------')
