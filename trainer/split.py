from __future__ import print_function
from torch import autograd
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from networks.resnet_split import Split_fc , Split_ResidualBlock
from networks.resnet_split import RegularizedLinear, ResidualBlock

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


    def first_train(self, epoch):

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

            loss_CE = 0
            loss_KD = 0
            split_loss = 0

            output = self.model(data)[:, :end]

            if tasknum == 0:
                loss_CE = self.loss(output, target)
            else:
                data_new, target_new = data[target >= start], target[target >= start]
                target_new = target_new % (end - start)
                output_curr = output[target >= start][:, start:end]
                loss_CE = self.loss(output_curr, target_new)

                end_KD = start
                score = self.model_fixed(data)[:, :end_KD].data
                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output[:, :end_KD] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')

                overlap_loss, uniform_loss, split_loss = self.model.reg_loss()
                split_loss *= 1.0

            self.optimizer.zero_grad()

            (loss_KD + split_loss + loss_CE).backward()

            self.optimizer.step()

            self.model.fc.linear.bias.data[:] = 0

            if tasknum == 0:
                weight = self.model.fc.linear.weight.data
                weight[weight < 0] = 0

    def second_train(self, epoch):

        T = 2
        self.model.train()
        self.model = self.model.cuda()
        print("Epochs %d" % epoch)

        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        lamb = start / end
        print("lamb :{}".format(lamb))

        for data, target in tqdm(self.train_data_iterator):
            data, target = data.cuda(), target.long().cuda()

            output = self.model(data)[:, :end]


            data_new, target_new = data[target >= start], target[target >= start]
            target_new = target_new % (end - start)
            output_curr = output[target >= start][:, start:end]
            loss_CE = self.loss(output_curr, target_new)

            end_KD = start

            score = self.model_fixed(data)[:, :end_KD].data
            soft_target = F.softmax(score / T, dim=1)
            output_log = F.log_softmax(output[:, :end_KD] / T, dim=1)
            loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')

            self.optimizer.zero_grad()

            (loss_KD + loss_CE).backward()
            self.optimizer.step()

    def third_train(self, epoch):

        T = 2
        self.model.cuda()
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

            score = self.model_fixed(data).data
            soft_target_pre = F.softmax(score[:, 0:start] / T, dim=1)
            output_log_pre = F.log_softmax(output[:, 0:start] / T, dim=1)
            loss_pre_KD = F.kl_div(output_log_pre, soft_target_pre, reduction='batchmean')

            self.optimizer.zero_grad()

            (lamb * loss_pre_KD + (1 - lamb) * loss_CE).backward()
            self.optimizer.step()


    def weight_align(self):
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size
        weight = self.model.fc.linear.weight.data

        prev = weight[:start, :]
        new = weight[start:end, :]
        print(prev.shape, new.shape)
        mean_prev = torch.mean(torch.norm(prev, dim=1)).item()
        mean_new = torch.mean(torch.norm(new, dim=1)).item()

        gamma = mean_prev / mean_new
        print(mean_prev, mean_new, gamma)
        new = new * gamma
        result = torch.cat((prev, new), dim=0)
        weight[:end, :] = result
        print(torch.mean(torch.norm(self.model.fc.linear.weight.data[:start], dim=1)).item())
        print(torch.mean(torch.norm(self.model.fc.linear.weight.data[start:end], dim=1)).item())


    def split(self):
        print("network split")
        self.model.cpu()
        with torch.no_grad():
            fc_weight = self.model.fc.linear.weight
            fc_bias = self.model.fc.linear.bias
            alpha_q = self.model.fc.qa
            alpha_p = self.model.fc.pa
            q_amax = np.argmax(alpha_q.detach(), 0)
            p_amax = np.argmax(alpha_p.detach(), 0)
            split_q_idxs = [np.where(q_amax == i)[0] for i in range(2)]
            split_p_idxs = [np.where(p_amax == i)[0] for i in range(2)]

            fc_params = {'weight': [], 'bias': [], 'input_perms': [], 'output_perms': []}
            self.split_params = {}
            for i in range(2):
                fc_params['weight'].append(fc_weight[split_q_idxs[i],:][:,split_p_idxs[i]])
                fc_params['bias'].append(fc_bias[split_q_idxs[i]])

            fc_params['input_perms'] = split_p_idxs
            fc_params['output_perms'] = split_q_idxs
            self.split_params['fc'] = fc_params

            fc_weight = self.split_params['fc']['weight']
            fc_bias = self.split_params['fc']['bias']
            fc_input_perms = self.split_params['fc']['input_perms']
            fc_output_perms = self.split_params['fc']['output_perms']

            split_fc = Split_fc(self.split_params, 2)
            self.model.fc = split_fc

            unit_3_0_shortcut_weight = self.model.feature.residual_block_groups[3].residual_blocks[0].conv_transform.weight
            unit_3_0_conv1_weight = self.model.feature.residual_block_groups[3].residual_blocks[0].conv1.weight
            unit_3_0_conv2_weight = self.model.feature.residual_block_groups[3].residual_blocks[0].conv2.weight
            unit_3_1_conv1_weight = self.model.feature.residual_block_groups[3].residual_blocks[1].conv1.weight
            unit_3_1_conv2_weight = self.model.feature.residual_block_groups[3].residual_blocks[1].conv2.weight

            alpha_q1 = self.model.feature.residual_block_groups[3].residual_blocks[1].qa
            alpha_r1 = self.model.feature.residual_block_groups[3].residual_blocks[1].ra
            alpha_p1 = self.model.feature.residual_block_groups[3].residual_blocks[1].pa
            alpha_q0 = self.model.feature.residual_block_groups[3].residual_blocks[0].qa
            alpha_r0 = self.model.feature.residual_block_groups[3].residual_blocks[0].ra
            alpha_p0 = self.model.feature.residual_block_groups[3].residual_blocks[0].pa

            q1_amax = np.argmax(alpha_q1.detach(), 0)
            p1_amax = np.argmax(alpha_p1.detach(), 0)
            r1_amax = np.argmax(alpha_r1.detach(), 0)
            q0_amax = np.argmax(alpha_q0.detach(), 0)
            p0_amax = np.argmax(alpha_p0.detach(), 0)
            r0_amax = np.argmax(alpha_r0.detach(), 0)

            split_q1_idxs = [np.where(q1_amax == i)[0] for i in range(2)]
            split_p1_idxs = [np.where(p1_amax == i)[0] for i in range(2)]
            split_r1_idxs = [np.where(r1_amax == i)[0] for i in range(2)]
            split_q0_idxs = [np.where(q0_amax == i)[0] for i in range(2)]
            split_p0_idxs = [np.where(p0_amax == i)[0] for i in range(2)]
            split_r0_idxs = [np.where(r0_amax == i)[0] for i in range(2)]

            unit_3_0_params = {'shortcut': [], 'conv1': [], 'conv2': [], 'p_perms': [], 'q_perms': [], 'r_perms': []}

            for i in range(2):
                unit_3_0_params['shortcut'].append(unit_3_0_shortcut_weight[split_q0_idxs[i], :, :, :][:, split_p0_idxs[i], :, :])
                unit_3_0_params['conv1'].append(unit_3_0_conv1_weight[split_r0_idxs[i], :, :, :][:, split_p0_idxs[i], :, :])
                unit_3_0_params['conv2'].append(unit_3_0_conv2_weight[split_q0_idxs[i], :, :, :][:, split_r0_idxs[i], :, :])

            unit_3_0_params['p_perms'] = split_p0_idxs
            unit_3_0_params['q_perms'] = split_q0_idxs
            unit_3_0_params['r_perms'] = split_r0_idxs
            self.split_params['unit_3_0'] = unit_3_0_params

            unit_3_1_params = {'conv1': [], 'conv2': [], 'p_perms': [], 'q_perms': [], 'r_perms': []}

            for i in range(2):
                unit_3_1_params['conv1'].append(unit_3_1_conv1_weight[split_r1_idxs[i], :, :, :][:, split_p1_idxs[i], :, :])
                unit_3_1_params['conv2'].append(unit_3_1_conv2_weight[split_q1_idxs[i], :, :, :][:, split_r1_idxs[i], :, :])

            unit_3_1_params['p_perms'] = split_p1_idxs
            unit_3_1_params['q_perms'] = split_q1_idxs
            unit_3_1_params['r_perms'] = split_r1_idxs
            self.split_params['unit_3_1'] = unit_3_1_params

            unit_3_0 = Split_ResidualBlock(self.split_params, 'unit_3_0', 2)
            unit_3_1 = Split_ResidualBlock(self.split_params, 'unit_3_1', 2)

            self.model.feature.residual_block_groups[3].residual_blocks[0] = unit_3_0
            self.model.feature.residual_block_groups[3].residual_blocks[1] = unit_3_1



    def reunion(self):
        print("network reunion")
        self.model.cpu()

        merged_fc = RegularizedLinear(
            self.model.feature.baseline_channels[self.model.feature.group_number], self.model.numclass
            , split_size=self.model.feature.split_sizes[-1]
        )
        fc_weight = self.split_params['fc']['weight']
        fc_bias = self.split_params['fc']['bias']
        fc_input_perms = self.split_params['fc']['input_perms']
        fc_output_perms = self.split_params['fc']['output_perms']

        weight = torch.zeros((self.model.numclass, self.model.feature.baseline_channels[self.model.feature.group_number]))
        weight = torch.nn.Parameter(weight)
        bias = torch.zeros(self.model.numclass)
        bias = torch.nn.Parameter(bias)

        merged_fc.linear.weight = weight
        merged_fc.linear.bias = bias

        with torch.no_grad():
            for k in range(2):
                for i in range(len(fc_output_perms[k])):
                    for j in range(len(fc_input_perms[k])):
                        a = fc_weight[k][i][j].clone().detach()
                        b = torch.nn.Parameter(a)
                        merged_fc.linear.weight[fc_output_perms[k][i]][fc_input_perms[k][j]] = b
                c = fc_bias[k][i].clone().detach()
                d = torch.nn.Parameter(c)
                merged_fc.linear.bias[fc_output_perms[k][i]] = d

        self.model.fc = merged_fc

        merged_unit_3_1 = ResidualBlock(self.model.feature.residual_block_groups[3].out_channels,
                                        self.model.feature.residual_block_groups[3].out_channels, 1,
                                        self.model.feature.residual_block_groups[3].split_size,
                                        # self.model.feature.residual_block_groups[3].split_qa_last
                                        self.model.fc.pa)

        merged_unit_3_0 = ResidualBlock(self.model.feature.residual_block_groups[3].in_channels,
                                        self.model.feature.residual_block_groups[3].out_channels,
                                        self.model.feature.residual_block_groups[3].stride,
                                        self.model.feature.residual_block_groups[3].split_size,
                                        merged_unit_3_1.pa)

        unit_3_0_shortcut_weight = self.split_params['unit_3_0']['shortcut']
        unit_3_0_conv1_weight = self.split_params['unit_3_0']['conv1']
        unit_3_0_conv2_weight = self.split_params['unit_3_0']['conv2']

        unit_3_0_p_params = self.split_params['unit_3_0']['p_perms']
        unit_3_0_r_params = self.split_params['unit_3_0']['r_perms']
        unit_3_0_q_params = self.split_params['unit_3_0']['q_perms']

        unit_3_1_conv1_weight = self.split_params['unit_3_1']['conv1']
        unit_3_1_conv2_weight = self.split_params['unit_3_1']['conv2']

        unit_3_1_p_params = self.split_params['unit_3_1']['p_perms']
        unit_3_1_r_params = self.split_params['unit_3_1']['r_perms']
        unit_3_1_q_params = self.split_params['unit_3_1']['q_perms']

        torch.nn.init.zeros_(merged_unit_3_0.conv_transform.weight)
        torch.nn.init.zeros_(merged_unit_3_0.conv1.weight)
        torch.nn.init.zeros_(merged_unit_3_0.conv2.weight)

        torch.nn.init.zeros_(merged_unit_3_1.conv1.weight)
        torch.nn.init.zeros_(merged_unit_3_1.conv2.weight)

        with torch.no_grad():
            for k in range(2):
                for i in range(len(unit_3_0_q_params[k])):
                    for j in range(len(unit_3_0_p_params[k])):
                        a = unit_3_0_shortcut_weight[k][i, j, :, :].clone().detach()
                        b = torch.nn.Parameter(a)
                        merged_unit_3_0.conv_transform.weight[unit_3_0_q_params[k][i], unit_3_0_p_params[k][j], :,
                        :] = b

                for i in range(len(unit_3_0_q_params[k])):
                    for j in range(len(unit_3_0_r_params[k])):
                        a = unit_3_0_conv2_weight[k][i, j, :, :].clone().detach()
                        b = torch.nn.Parameter(a)
                        merged_unit_3_0.conv2.weight[unit_3_0_q_params[k][i], unit_3_0_r_params[k][j], :, :] = b

                for i in range(len(unit_3_0_r_params[k])):
                    for j in range(len(unit_3_0_p_params[k])):
                        a = unit_3_0_conv1_weight[k][i, j, :, :].clone().detach()
                        b = torch.nn.Parameter(a)
                        merged_unit_3_0.conv1.weight[unit_3_0_r_params[k][i], unit_3_0_p_params[k][j], :, :] = b

                for i in range(len(unit_3_1_q_params[k])):
                    for j in range(len(unit_3_1_r_params[k])):
                        a = unit_3_1_conv2_weight[k][i, j, :, :].clone().detach()
                        b = torch.nn.Parameter(a)
                        merged_unit_3_1.conv2.weight[unit_3_1_q_params[k][i], unit_3_1_r_params[k][j], :, :] = b

                for i in range(len(unit_3_1_r_params[k])):
                    for j in range(len(unit_3_1_p_params[k])):
                        a = unit_3_1_conv1_weight[k][i, j, :, :].clone().detach()
                        b = torch.nn.Parameter(a)
                        merged_unit_3_1.conv1.weight[unit_3_1_r_params[k][i], unit_3_1_p_params[k][j], :, :] = b

        self.model.feature.residual_block_groups[3].residual_blocks[0] = merged_unit_3_0
        self.model.feature.residual_block_groups[3].residual_blocks[1] = merged_unit_3_1



