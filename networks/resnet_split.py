import itertools
import copy
import operator
from functools import reduce, partial
from torch import nn
import torch
import numpy as np
from networks import splits


class WeightRegularized(nn.Module):
    def reg_losses(self):
        """
        Should return an iterable of (OVERLAP_LOSS, UNIFORM_LOSS, SPLIT_LOSS)
        triples.
        """
        raise NotImplementedError

    @property
    def is_cuda(self):
        # Simple hack to see if the module is built with CUDA parameters.
        if hasattr(self, '__cuda_flag_cache'):
            return self.__cuda_flag_cache
        self.__cuda_flag_cache = next(self.parameters()).is_cuda
        return self.__cuda_flag_cache


class RegularizedLinear(WeightRegularized):
    def __init__(self, in_channels, out_channels,
                 split_size=None, split_qa=None, dropout_prob=.5):
        super().__init__()
        self.split_size = split_size
        self.splitted = self.split_size is not None

        # Layers.
        self.linear = nn.Linear(in_channels, out_channels)

        # Split indicator alphas.
        if split_size:
            self.pa = splits.alpha(split_size, in_channels)
            self.qa = (
                    split_qa or
                    splits.alpha(split_size, out_channels)
            )
        else:
            self.pa = None
            self.qa = None

    def p(self):
        return splits.q(self.pa)

    def q(self):
        return splits.q(self.qa)

    def forward(self, x):
        return self.linear(x)

    def reg_losses(self):
        return [splits.reg_loss(
            self.linear.weight, self.p(), self.q(), cuda=self.is_cuda
        )]


class Split_fc(nn.Module):
    def __init__(self, split_params, split_size=None):
        super().__init__()
        self.split_size = split_size
        self.fc_weight = split_params['fc']['weight']
        self.fc_biases = split_params['fc']['bias']
        self.fc_input_perms = split_params['fc']['input_perms']
        self.fc_output_perms = split_params['fc']['output_perms']

        out_dim1, in_dim1 = self.fc_weight[0].shape
        self.fc1 = nn.Linear(in_dim1, out_dim1)
        self.fc1.weight = nn.Parameter(self.fc_weight[0])
        self.fc1.bias = nn.Parameter(self.fc_biases[0])

        out_dim2, in_dim2 = self.fc_weight[1].shape
        self.fc2 = nn.Linear(in_dim2, out_dim2)
        self.fc2.weight = nn.Parameter(self.fc_weight[1])
        self.fc2.bias = nn.Parameter(self.fc_biases[1])

    def forward(self, x):
        x_list = []
        x_dim = x.shape[0]

        out_dim1, in_dim1 = self.fc_weight[0].shape
        x_split = torch.gather(x, 1, torch.tensor(self.fc_input_perms[0]).expand(x_dim, in_dim1).cuda())
        x_split = self.fc1(x_split)
        x_list.append(x_split)

        out_dim2, in_dim2 = self.fc_weight[1].shape
        x_split = torch.gather(x, 1, torch.tensor(self.fc_input_perms[1]).expand(x_dim, in_dim2).cuda())
        x_split = self.fc2(x_split)
        x_list.append(x_split)

        x = torch.cat(x_list, 1)
        output_forward_idx = list(np.concatenate(self.fc_output_perms))
        output_inverse_idx = [output_forward_idx.index(i) for i in range(out_dim1 + out_dim2)]
        x = torch.gather(x, 1, torch.tensor(output_inverse_idx).expand(x_dim, (out_dim1 + out_dim2)).cuda())

        return x


class Split_conv1(nn.Module):
    def __init__(self, split_params, block_name, split_size=None):
        super().__init__()
        self.split_size = split_size
        self.block_name = block_name
        self.conv1_weight = split_params[self.block_name]['conv1']
        self.conv1_input_perms = split_params[self.block_name]['p_perms']
        self.conv1_output_perms = split_params[self.block_name]['r_perms']

        out_dim1, in_dim1, _, _ = self.conv1_weight[0].shape
        if block_name == 'unit_3_0':
            self.conv1_1 = nn.Conv2d(in_dim1, out_dim1, kernel_size=3, stride=2, padding=1, bias=False)
        elif block_name == 'unit_2_0':
            self.conv1_1 = nn.Conv2d(in_dim1, out_dim1, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1_1 = nn.Conv2d(in_dim1, out_dim1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_1.weight = nn.Parameter(self.conv1_weight[0])

        out_dim2, in_dim2, _, _ = self.conv1_weight[1].shape
        if block_name == 'unit_3_0':
            self.conv1_2 = nn.Conv2d(in_dim2, out_dim2, kernel_size=3, stride=2, padding=1, bias=False)
        elif block_name == 'unit_2_0':
            self.conv1_2 = nn.Conv2d(in_dim2, out_dim2, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1_2 = nn.Conv2d(in_dim2, out_dim2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_2.weight = nn.Parameter(self.conv1_weight[1])

    def forward(self, x):
        x_list = []
        x_dim = x.shape[0]

        out_dim1, in_dim1, w, h = self.conv1_weight[0].shape
        x_split = x[:, self.conv1_input_perms[0], :, :]
        self.conv1_1.cuda()
        x_split = self.conv1_1(x_split)
        x_list.append(x_split)

        out_dim2, in_dim2, w, h = self.conv1_weight[1].shape
        x_split = x[::, self.conv1_input_perms[1], :, :]
        self.conv1_2.cuda()
        x_split = self.conv1_2(x_split)
        x_list.append(x_split)

        x = torch.cat(x_list, 1)
        output_forward_idx = list(np.concatenate(self.conv1_output_perms))
        output_inverse_idx = [output_forward_idx.index(i) for i in range(out_dim1 + out_dim2)]
        x = x[:, output_inverse_idx, :, :]

        return x


class Split_conv2(nn.Module):
    def __init__(self, split_params, block_name, split_size=None):
        super().__init__()
        self.split_size = split_size
        self.block_name = block_name
        self.conv2_weight = split_params[self.block_name]['conv2']
        self.conv2_input_perms = split_params[self.block_name]['r_perms']
        self.conv2_output_perms = split_params[self.block_name]['q_perms']

        out_dim1, in_dim1, _, _ = self.conv2_weight[0].shape
        self.conv2_1 = nn.Conv2d(in_dim1, out_dim1, kernel_size=3, padding=1, bias=False)
        self.conv2_1.weight = nn.Parameter(self.conv2_weight[0])

        out_dim2, in_dim2, _, _ = self.conv2_weight[1].shape
        self.conv2_2 = nn.Conv2d(in_dim2, out_dim2, kernel_size=3, padding=1, bias=False)
        self.conv2_2.weight = nn.Parameter(self.conv2_weight[1])

    def forward(self, x):
        x_list = []
        x_dim = x.shape[0]

        out_dim1, in_dim1, w, h = self.conv2_weight[0].shape
        x_split = x[:, self.conv2_input_perms[0], :, :]
        self.conv2_1.cuda()
        x_split = self.conv2_1(x_split)
        x_list.append(x_split)

        out_dim2, in_dim2, w, h = self.conv2_weight[1].shape
        x_split = x[::, self.conv2_input_perms[1], :, :]
        self.conv2_2.cuda()
        x_split = self.conv2_2(x_split)
        x_list.append(x_split)

        x = torch.cat(x_list, 1)
        output_forward_idx = list(np.concatenate(self.conv2_output_perms))
        output_inverse_idx = [output_forward_idx.index(i) for i in range(out_dim1 + out_dim2)]
        x = x[:, output_inverse_idx, :, :]

        return x


class Split_shortcut(nn.Module):
    def __init__(self, split_params, block_name, split_size=None):
        super().__init__()
        self.split_size = split_size
        self.block_name = block_name
        self.shortcut_weight = split_params[self.block_name]['shortcut']
        self.shortcut_input_perms = split_params[self.block_name]['p_perms']
        self.shortcut_output_perms = split_params[self.block_name]['q_perms']

        out_dim1, in_dim1, _, _ = self.shortcut_weight[0].shape
        self.shortcut_1 = nn.Conv2d(in_dim1, out_dim1, kernel_size=1, stride=2, bias=False)
        self.shortcut_1.weight = nn.Parameter(self.shortcut_weight[0])

        out_dim2, in_dim2, _, _ = self.shortcut_weight[1].shape
        self.shortcut_2 = nn.Conv2d(in_dim2, out_dim2, kernel_size=1, stride=2, bias=False)
        self.shortcut_2.weight = nn.Parameter(self.shortcut_weight[1])

    def forward(self, x):
        x_list = []
        x_dim = x.shape[0]

        out_dim1, in_dim1, w, h = self.shortcut_weight[0].shape
        x_split = x[:, self.shortcut_input_perms[0], :, :]
        self.shortcut_1.cuda()
        x_split = self.shortcut_1(x_split)
        x_list.append(x_split)

        out_dim2, in_dim2, w, h = self.shortcut_weight[1].shape
        x_split = x[::, self.shortcut_input_perms[1], :, :]
        self.shortcut_2.cuda()
        x_split = self.shortcut_2(x_split)
        x_list.append(x_split)

        x = torch.cat(x_list, 1)
        output_forward_idx = list(np.concatenate(self.shortcut_output_perms))
        output_inverse_idx = [output_forward_idx.index(i) for i in range(out_dim1 + out_dim2)]
        x = x[:, output_inverse_idx, :, :]

        return x


class Split_ResidualBlock(nn.Module):
    def __init__(self, split_params, block_name, split_size=None):
        super().__init__()
        self.split_size = split_size

        self.conv1 = Split_conv1(split_params, block_name, 2)
        self.bn1 = nn.BatchNorm2d(self.conv1.conv1_1.out_channels + self.conv1.conv1_2.out_channels)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = Split_conv2(split_params, block_name, 2)
        self.bn2 = nn.BatchNorm2d(self.conv2.conv2_1.out_channels + self.conv2.conv2_2.out_channels)

        self.need_transform = (self.conv1.conv1_1.in_channels + self.conv1.conv1_2.in_channels) != (
                    self.conv1.conv1_1.out_channels + self.conv1.conv1_2.out_channels)
        self.shortcut = Split_shortcut(split_params, block_name, 2) if self.need_transform else None

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))

        y = self.bn2(self.conv2(y))

        return nn.ReLU(inplace=False)(y.add_(self.shortcut(x) if self.need_transform else x))


class ResidualBlock(WeightRegularized):
    def __init__(self, in_channels, out_channels, stride,
                 split_size=None, split_qa=None):
        super().__init__()
        self.split_size = split_size
        self.splitted = self.split_size is not None

        # 1
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)

        # 2
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # transformation
        self.need_transform = in_channels != out_channels
        self.conv_transform = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=stride, padding=0, bias=False
        ) if self.need_transform else None

        # weight
        self.w1 = self.conv1.weight
        self.w2 = self.conv2.weight
        self.w3 = self.conv_transform.weight if self.need_transform else None

        # split indicators
        if split_size:
            self.pa = splits.alpha(split_size, in_channels)
            self.ra = splits.alpha(split_size, out_channels)
            self.qa = (
                split_qa if split_qa is not None else
                splits.alpha(split_size, out_channels)
            )
        else:
            self.pa = None
            self.ra = None
            self.qa = None

    def p(self):
        return splits.q(self.pa)

    def r(self):
        return splits.q(self.ra)

    def q(self):
        return splits.q(self.qa)

    def forward(self, x):
        # conv1
        y = self.relu1(self.bn1(self.conv1(x)))

        # conv2
        y = self.bn2(self.conv2(y))

        # conv2 + residual
        return nn.ReLU(inplace=False)(y.add_(self.conv_transform(x) if self.need_transform else x))

    def reg_losses(self):
        weights_and_split_indicators = filter(partial(operator.is_not, None), [
            (self.w1, self.p(), self.r()),
            (self.w2, self.r(), self.q()),
            (self.w3, self.p(), self.q()) if self.need_transform else None
        ])

        return [
            splits.reg_loss(w, p, q, cuda=self.is_cuda) for w, p, q in
            weights_and_split_indicators if (p is not None and q is not None)
        ]


class ResidualBlockGroup(WeightRegularized):
    def __init__(self, block_number, in_channels, out_channels, stride,
                 split_size=None, split_qa_last=None):
        super().__init__()

        # Residual block group's hyperparameters.
        self.block_number = block_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.split_size = split_size
        self.split_qa_last = split_qa_last
        self.splitted = split_size is not None

        # Define residual blocks in a reversed order. This is to define
        # feature groups in hierarchical manner - from subgroups to
        # supergroups.
        residual_blocks = []
        for i in reversed(range(self.block_number)):
            is_first = (i == 0)
            is_last = (i == self.block_number - 1)

            if self.splitted:
                qa = self.split_qa_last if is_last else residual_blocks[0].pa
            else:
                qa = None

            block_class = ResidualBlock

            block = block_class(
                self.in_channels if is_first else self.out_channels,
                self.out_channels,
                self.stride if is_first else 1,
                split_size=split_size,
                split_qa=qa,
            )
            residual_blocks.insert(0, block)
        # Register the residual block modules.
        self.residual_blocks = nn.ModuleList(residual_blocks)

    def forward(self, x):
        return reduce(lambda x, f: f(x), self.residual_blocks, x)

    def reg_losses(self):
        return itertools.chain(*[
            b.reg_losses() for b in self.residual_blocks
        ])


class ResNet(WeightRegularized):
    def __init__(self, label, input_size, input_channels, classes,
                 total_block_number,
                 baseline_strides=None,
                 baseline_channels=None,
                 split_sizes=None):
        super().__init__()

        # Model name label.
        self.label = label

        # Data specific hyperparameters.
        self.input_size = input_size
        self.input_channels = input_channels
        self.classes = classes

        # Model hyperparameters.
        self.total_block_number = total_block_number
        self.split_sizes = split_sizes
        self.baseline_strides = baseline_strides or [1, 1, 2, 2, 2]
        self.baseline_channels = baseline_channels or [64, 64, 128, 256, 512]
        self.group_number = len(self.baseline_channels) - 1

        # Residual block group configurations.
        split_sizes_stack = copy.deepcopy(self.split_sizes)
        blocks_per_group = self.total_block_number // self.group_number
        zipped_channels_and_strides = list(zip(
            self.baseline_channels[:-1],
            self.baseline_channels[1:],
            self.baseline_strides[1:]
        ))

        # 4. Affine layer.
        self.fc = RegularizedLinear(
            self.baseline_channels[self.group_number], self.classes,
            split_size=split_sizes_stack.pop()
        )

        # 3. pooling.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Residual block groups.
        residual_block_groups = []
        for k, (i, o, s) in reversed(list(
                enumerate(zipped_channels_and_strides)
        )):
            is_last = (k == len(zipped_channels_and_strides) - 1)
            try:
                # Case of splitting a residual block group.
                split_size = split_sizes_stack.pop()
                split_qa_last = (
                    self.fc.pa if is_last else
                    residual_block_groups[0].residual_blocks[0].pa
                )
            except IndexError:
                # Case of not splitting a residual block group.
                split_size = None
                split_qa_last = None

            # Push the residual block groups from upside down.
            residual_block_groups.insert(0, ResidualBlockGroup(
                blocks_per_group, i, o, s,
                split_size=split_size,
                split_qa_last=split_qa_last
            ))
        # Register the residual block group modules.
        self.residual_block_groups = nn.ModuleList(residual_block_groups)

        # 1. Convolution layer.
        self.conv = nn.Conv2d(
            self.input_channels, self.baseline_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(self.baseline_channels[0])

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return reduce(lambda x, f: f(x), [
            self.conv,
            self.bn,
            self.relu,
            *self.residual_block_groups,
            self.pool,
            (lambda x: x.view(-1, self.baseline_channels[-1])),
            #self.fc
        ], x)

    def reg_losses(self):
        return itertools.chain(self.fc.reg_losses(), *[
            g.reg_losses() for
            g in self.residual_block_groups if g.splitted
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

    @property
    def name(self):
        # Label for the split group configurations.
        if self.split_sizes:
            split_label = 'split[{}]-'.format('-'.join(
                str(s) for s in self.split_sizes
            ))
        else:
            split_label = ''

        # First block of a residual group contains 3 conv layers and rest
        # blocks of the group contains 2 conv layers.
        depth = self.group_number * 3 + (
                self.total_block_number -
                self.group_number
        ) * 2 + 1

        # Name of the model.
        return (
            'Resnet-{depth}-{split_label}-'
            '{label}-{size}x{size}x{channels}'
        ).format(
            depth=depth,
            split_label=split_label,
            label=self.label,
            size=self.input_size,
            channels=self.input_channels,
        )