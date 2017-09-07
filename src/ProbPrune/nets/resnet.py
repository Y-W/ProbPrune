"""
ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""

from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    def __init__(self, in_channels, out_channels, stride=1, prune=None, batch_norm_avg_factor=1e-2):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=batch_norm_avg_factor)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=batch_norm_avg_factor)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.identity_shortcut = True
        if stride != 1 or in_channels != out_channels:
            self.identity_shortcut = False
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.prune = prune
        if self.prune is None:
            self.num_prune = []
        elif self.identity_shortcut:
            self.num_prune = [out_channels, out_channels]
        else:
            self.num_prune = [out_channels, out_channels, out_channels]

    def forward(self, x, *prune_inputs):
        assert len(prune_inputs) == len(self.num_prune)
        prune_ptr = 0

        if self.identity_shortcut:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
            if self.prune is not None:
                shortcut = self.prune(shortcut, prune_inputs[prune_ptr])
                prune_ptr += 1

        out = self.conv1(F.relu(self.bn1(x)))
        if self.prune is not None:
            out = self.prune(out, prune_inputs[prune_ptr])
            prune_ptr += 1
        out = self.conv2(F.relu(self.bn2(out)))
        if self.prune is not None:
            out = self.prune(out, prune_inputs[prune_ptr])
            prune_ptr += 1
        return out + shortcut


class PreActBottleneck(nn.Module):
    """Pre-activation version of the BasicBlock."""
    def __init__(self, in_channels, out_channels, stride=1, prune=None, bottleneck_factor=0.25, batch_norm_avg_factor=1e-2):
        super(PreActBottleneck, self).__init__()
        bottleneck_channels = int(round(out_channels * bottleneck_factor))
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=batch_norm_avg_factor)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels, momentum=batch_norm_avg_factor)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels, momentum=batch_norm_avg_factor)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)

        self.identity_shortcut = True
        if stride != 1 or in_channels != out_channels:
            self.identity_shortcut = False
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.prune = prune
        if self.prune is None:
            self.num_prune = []
        elif self.identity_shortcut:
            self.num_prune = [bottleneck_channels, bottleneck_channels, out_channels]
        else:
            self.num_prune = [out_channels, bottleneck_channels, bottleneck_channels, out_channels]

    def forward(self, x, *prune_inputs):
        assert len(prune_inputs) == len(self.num_prune)
        prune_ptr = 0

        if self.identity_shortcut:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
            if self.prune is not None:
                shortcut = self.prune(shortcut, prune_inputs[prune_ptr])
                prune_ptr += 1

        out = self.conv1(F.relu(self.bn1(x)))
        if self.prune is not None:
            out = self.prune(out, prune_inputs[prune_ptr])
            prune_ptr += 1
        out = self.conv2(F.relu(self.bn2(out)))
        if self.prune is not None:
            out = self.prune(out, prune_inputs[prune_ptr])
            prune_ptr += 1
        out = self.conv3(F.relu(self.bn3(out)))
        if self.prune is not None:
            out = self.prune(out, prune_inputs[prune_ptr])
            prune_ptr += 1
        return out + shortcut


class StageBlock(nn.Module):
    def __init__(self, num_block, in_channels, out_channels, stride=2, prune=None, bottleneck_factor=None, batch_norm_avg_factor=1e-2):
        super(StageBlock, self).__init__()
        assert num_block >= 1
        self.num_prune = []
        blocks = []

        if bottleneck_factor is None:
            b = PreActBlock(in_channels, out_channels, stride, prune, batch_norm_avg_factor)
        else:
            b = PreActBottleneck(in_channels, out_channels, stride, prune, bottleneck_factor, batch_norm_avg_factor)
        self.num_prune.extend(b.num_prune)
        blocks.append(b)

        for i in xrange(num_block - 1):
            if bottleneck_factor is None:
                b = PreActBlock(out_channels, out_channels, 1, prune, batch_norm_avg_factor)
            else:
                b = PreActBottleneck(out_channels, out_channels, 1, prune, bottleneck_factor, batch_norm_avg_factor)
            self.num_prune.extend(b.num_prune)
            blocks.append(b)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, *prune_inputs):
        assert len(prune_inputs) == len(self.num_prune)
        prune_ptr = 0
        for b in self.blocks:
            b_prunes = len(b.num_prune)
            x = b(x, *prune_inputs[prune_ptr:prune_ptr+b_prunes])
            prune_ptr += b_prunes
        return x


class ResNet(nn.Module):
    def __init__(self, config, prune):
        super(ResNet, self).__init__()

        num_input = config.getint('general', 'num_input')
        num_output = config.getint('general', 'num_output')
        bottleneck_factor = config.getfloat('resnet_arch', 'bottleneck')
        bottleneck_factor = None if bottleneck_factor <= 0.0 else bottleneck_factor
        st_channels = config.getint('resnet_arch', 'starting_channels')
        block_nums = map(int, config.get('resnet_arch', 'block_nums').split(','))
        first_conv_type = config.get('resnet_arch', 'first_conv')
        batch_norm_avg_factor = config.getfloat('batch_norm', 'averaging_factor')

        self.prune = prune
        self.num_prune = []

        if first_conv_type == 'cifar':
            last_channels = 16
            self.first_conv = nn.Conv2d(num_input, last_channels, kernel_size=3, padding=1, bias=False)
            self.max_pool_after_first_conv = False
        else:
            last_channels = 64
            self.first_conv = nn.Conv2d(num_input, last_channels, kernel_size=7, padding=3, bias=False)
            self.max_pool_after_first_conv = True

        if self.prune is not None:
            self.num_prune.append(last_channels)
        self.first_bn = nn.BatchNorm2d(last_channels, momentum=batch_norm_avg_factor)

        target_channels = st_channels
        stages = []
        for i, k in enumerate(block_nums):
            if i > 0:
                stage = StageBlock(k, last_channels, target_channels, stride=2,
                                   prune=prune,
                                   bottleneck_factor=bottleneck_factor,
                                   batch_norm_avg_factor=batch_norm_avg_factor)
            else:
                stage = StageBlock(k, last_channels, target_channels, stride=1,
                                   prune=prune,
                                   bottleneck_factor=bottleneck_factor,
                                   batch_norm_avg_factor=batch_norm_avg_factor)
            self.num_prune.extend(stage.num_prune)
            stages.append(stage)
        self.stages = nn.ModuleList(stages)

        self.final_bn = nn.BatchNorm2d(last_channels, momentum=batch_norm_avg_factor)
        self.final = nn.Linear(last_channels, num_output)

    def forward(self, x, *prune_inputs):
        assert len(prune_inputs) == len(self.num_prune)
        prune_ptr = 0

        out = self.first_conv(x)
        if self.prune is not None:
            out = self.prune(out, prune_inputs[prune_ptr])
            prune_ptr += 1
        out = F.relu(self.first_bn(out))

        if self.max_pool_after_first_conv:
            out = F.max_pool2d(out, 3, stride=2, padding=1)

        for s in self.stages:
            current_prunes = len(s.num_prune)
            out = s(out, *prune_inputs[prune_ptr:prune_ptr+current_prunes])
            prune_ptr += current_prunes

        out = out.mean(dim=3).mean(dim=2)
        out = F.relu(self.final_bn(out))
        out = self.final(out)

        return out
