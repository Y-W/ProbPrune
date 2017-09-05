from __future__ import absolute_import

from .resnet import ResNet


def build_net(config, prune):
    net_type = config.get('general', 'net_type')
    if net_type == 'resnet':
        return ResNet(config, prune)
    else:
        raise NotImplementedError
