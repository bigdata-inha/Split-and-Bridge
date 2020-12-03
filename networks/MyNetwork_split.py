import torch.nn as nn
import itertools
from networks.resnet_split import ResNet, RegularizedLinear
from networks.resnet32 import resnet32
import torch
from torch.nn import Parameter
import random
import numpy as np



class network(nn.Module):

    def __init__(self, label, input_size, numclass):
        super(network, self).__init__()

        self.label = label
        self.input_size = input_size
        self.numclass = numclass
        feature_extractor = ResNet(self.label, self.input_size, 3, self.numclass, total_block_number=8,
                                   baseline_strides=[1, 1, 2, 2, 2],
                                   baseline_channels=[64, 64, 128, 256, 512], split_sizes=[2, 2])

        self.feature = feature_extractor

        self.fc = RegularizedLinear(
            self.feature.baseline_channels[self.feature.group_number], self.numclass, split_size=self.feature.split_sizes[-1]
        )

    def forward(self, input, feature_return = False):
        feature = self.feature(input)
        x = self.fc(feature)
        if feature_return:
            return x, feature
        return x

    def Incremental_learning(self, numclass, step_size = 0, balance_factor = 0.5):
        self.numclass = numclass
        self.balance_factor = balance_factor
        print("old class split:{}".format(balance_factor))
        print("new class split:{}".format(1-balance_factor))

        weight = self.fc.linear.weight.data
        bias = self.fc.linear.bias.data
        in_feature = self.fc.linear.in_features
        out_feature = self.fc.linear.out_features

        self.fc = RegularizedLinear(
            self.feature.baseline_channels[self.feature.group_number], self.numclass,
            split_size=self.feature.split_sizes[-1]
        )
        self.fc.pa = self.feature.residual_block_groups[3].residual_blocks[1].qa

        self.fc.linear.weight.data[:out_feature] = weight
        self.fc.linear.bias.data[:out_feature] = bias

        size = (self.numclass - step_size)
        cls = Parameter
        a = torch.ones(2, numclass)
        a[0][:size] = a[0][:size] * 2.0
        a[0][size:] = a[0][size:] * -2.0
        a[1][:size] = a[1][:size] * -2.0
        a[1][size:] = a[1][size:] * 2.0
        self.fc.qa = cls(a.cuda(), requires_grad=False)

        size1 = int(512 * balance_factor)
        b = torch.ones(2, 512)
        b[0][:size1] = b[0][:size1] * 2.0
        b[0][size1:] = b[0][size1:] * -2.0
        b[1][:size1] = b[1][:size1] * -2.0
        b[1][size1:] = b[1][size1:] * 2.0
        self.fc.pa = cls(b.cuda(), requires_grad=False)
        self.feature.residual_block_groups[3].residual_blocks[1].qa = self.fc.pa

        c = torch.ones(2, 512)
        c[0][:size1] = c[0][:size1] * 2.0
        c[0][size1:] = c[0][size1:] * -2.0
        c[1][:size1] = c[1][:size1] * -2.0
        c[1][size1:] = c[1][size1:] * 2.0
        self.feature.residual_block_groups[3].residual_blocks[1].ra = cls(c.cuda(), requires_grad=False)

        d = torch.ones(2, 512)
        d[0][:size1] = d[0][:size1] * 2.0
        d[0][size1:] = d[0][size1:] * -2.0
        d[1][:size1] = d[1][:size1] * -2.0
        d[1][size1:] = d[1][size1:] * 2.0
        self.feature.residual_block_groups[3].residual_blocks[1].pa = cls(d.cuda(), requires_grad=False)
        self.feature.residual_block_groups[3].residual_blocks[0].qa = \
            self.feature.residual_block_groups[3].residual_blocks[1].pa

        e = torch.ones(2, 512)
        e[0][:size1] = e[0][:size1] * 2.0
        e[0][size1:] = e[0][size1:] * -2.0
        e[1][:size1] = e[1][:size1] * -2.0
        e[1][size1:] = e[1][size1:] * 2.0
        self.feature.residual_block_groups[3].residual_blocks[0].ra = cls(e.cuda(), requires_grad=False)

        size2 = int(256 * balance_factor)
        f = torch.ones(2, 256)
        f[0][:size2] = f[0][:size2] * 2.0
        f[0][size2:] = f[0][size2:] * -2.0
        f[1][:size2] = f[1][:size2] * -2.0
        f[1][size2:] = f[1][size2:] * 2.0
        self.feature.residual_block_groups[3].residual_blocks[0].pa = cls(f.cuda(), requires_grad=False)

    def feature_extractor(self, inputs):
        return self.feature(inputs)

    def reg_losses(self):
        return itertools.chain(self.fc.reg_losses(), *[
            g.reg_losses() for
            g in self.feature.residual_block_groups if g.splitted
        ])

    def reg_loss(self):
        reg_losses = self.reg_losses()
        overlap_losses, uniform_losses, split_losses = tuple(zip(*reg_losses))
        split_loss_weights = [l.detach() for l in split_losses]
        split_losses_weighted = [l.detach() * l for l in split_losses]
        return (
            sum(overlap_losses) / len(overlap_losses),
            sum(uniform_losses) / len(uniform_losses),
            sum(split_losses_weighted) / sum(split_loss_weights),
        )