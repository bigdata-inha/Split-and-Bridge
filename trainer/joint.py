from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import networks
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f" % (self.current_lr,
                                                                          self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def increment_classes(self):

        self.train_data_iterator.dataset.update_exemplar()
        self.train_data_iterator.dataset.task_change()
        self.test_data_iterator.dataset.task_change()

    def get_optimizer(self, optimizer):
        self.optimizer = optimizer

    def setup_training(self, lr):

        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f" % lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def get_model(self):
        myModel = networks.ModelFactory.get_model(self.args.dataset).cuda()
        optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)
        myModel.eval()

        self.current_lr = self.args.lr

        self.model_single = myModel
        self.optimizer_single = optimizer

    def train(self, epoch):

        T = 2

        self.model.train()
        print("Epochs %d" % epoch)

        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        lamb = start / end
        print("lamb :{}".format(lamb))

        for data, target in tqdm(self.train_data_iterator):
            data, target = data.cuda(), target.long().cuda()

            output = self.model(data)[:, :end]
            loss_CE = self.loss(output, target)


            self.optimizer.zero_grad()
            (loss_CE).backward()
            self.optimizer.step()

            self.model.fc.linear.bias.data[:] = 0

            # weight cliping 0인걸 없애기
            if tasknum == 0:
                weight = self.model.fc.linear.weight.data
                weight[weight < 0] = 0
