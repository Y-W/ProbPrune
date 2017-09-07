from __future__ import absolute_import, division, print_function

import sys
import os
import configparser
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

torch.backends.cudnn.benchmark = True

from .datasets import get_dataset
from .nets import build_net


class TrainVanilla(object):
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.exp_name = self.config.get('general', 'name')

        self.dataset_name = self.config.get('dataset', 'dataset')
        self.trainloader, self.testloader, self.dataset_info = get_dataset(
            self.dataset_name, config=self.config)

        net_config = configparser.ConfigParser()
        net_config.read(self.config.get('net_arch', 'net_arch'))
        self.net = build_net(net_config, None)
        self.net.cuda()

        self.train_loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.config.getfloat('train', 'learning_rate'),
            momentum=self.config.getfloat('train', 'momentum'),
            weight_decay=self.config.getfloat('train', 'weight_decay'))

        self.lr_init = self.config.getfloat('train', 'learning_rate')
        self.lr_steps = map(int, self.config.get('train',
                                                 'lr_step_epochs').split(','))
        self.train_epochs = self.config.getint('train', 'epochs')
        self.current_epoch = 0

        self.output_dir = self.config.get('outputs', 'output_dir')
        self.output_name = self.config.get('outputs', 'output_name')
        if not self.output_name.endswith('.t7'):
            self.output_name += '.t7'
        self.save_interval = self.config.getint('outputs', 'save_interval')

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self):
        print('Train Epoch %d | Time: %s'
              % (self.current_epoch,
                 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(
                     datetime.datetime.now())))
        lr = self.lr_init * (0.1 ** len([t for t in self.lr_steps
                                         if t <= self.current_epoch]))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('Learning rate: %f' % lr)

        self.net.train()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (X, Y) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            X, Y = X.cuda(), Y.cuda()
            X, Y = Variable(X), Variable(Y)
            outputs = self.net(X)
            loss = self.train_loss(outputs, Y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.data[0]
            _, predicted = outputs.data.max(dim=1)
            total += Y.size(0)
            correct += predicted.eq(Y.data).sum()

            if batch_idx > 0:
                print('\r', end='')

            print('%d/%d | Loss: %.3f | Acc: %.3f %% (%d/%d)'
                  % (batch_idx + 1, len(self.trainloader),
                     train_loss / (batch_idx + 1),
                     100. * correct / total,
                     correct, total),
                  end='')
        print()

    def test(self):
        print('Test Epoch %d | Time: %s'
              % (self.current_epoch,
                 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(
                     datetime.datetime.now())))

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.train_loss(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum()

        print('%d/%d | Loss: %.3f | Acc: %.3f %% (%d/%d)'
              % (len(self.testloader), len(self.testloader),
                 test_loss / (batch_idx + 1),
                 100. * correct / total,
                 correct, total))

        if self.current_epoch % self.save_interval == 0 \
                or self.current_epoch == self.train_epochs:
            print('Saving model')
            params = self.net.state_dict()
            torch.save(params, os.path.join(self.output_dir,
                                            self.output_name))

    def __call__(self):
        for e in xrange(self.train_epochs):
            self.current_epoch += 1
            self.train()
            self.test()


if __name__ == '__main__':
    TrainVanilla(sys.argv[1])()
