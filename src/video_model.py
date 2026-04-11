import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from collections import namedtuple

GlobalParams = namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])
BlockArgs = namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)

def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class MBConvBlock(nn.Module):
    """ Mobile Inverted Residual Bottleneck Block """
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k, s = self._block_args.kernel_size, self._block_args.stride
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, padding=(k - 1) // 2, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1: x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        x = self._bn2(self._project_conv(x))
        if self.id_skip and self._block_args.stride == 1 and self._block_args.input_filters == self._block_args.output_filters:
            if drop_connect_rate: x = F.dropout(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

class EfficientNet(nn.Module):
    """ An EfficientNet model. """
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        self._global_params, self._blocks_args = global_params, blocks_args
        bn_mom, bn_eps = 1 - self._global_params.batch_norm_momentum, self._global_params.batch_norm_epsilon
        in_channels, out_channels = 3, round_filters(32, self._global_params)
        self._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1): self._blocks.append(MBConvBlock(block_args, self._global_params))
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def forward(self, inputs):
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate: drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = relu_fn(self._bn1(self._conv_head(x)))
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout: x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

def efficientnet_params(model_name): return (1.0, 1.0, 224, 0.2)

class BlockDecoder(object):
    @staticmethod
    def _decode_block_string(block_string):
        ops, options = block_string.split('_'), {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2: key, value = splits[:2]; options[key] = value
        return BlockArgs(
            kernel_size=int(options['k']), num_repeat=int(options['r']),
            input_filters=int(options['i']), output_filters=int(options['o']),
            expand_ratio=int(options['e']), id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])
    @staticmethod
    def decode(string_list): return [BlockDecoder._decode_block_string(s) for s in string_list]

def get_model_params(model_name, override_params):
    w, d, s, p = efficientnet_params(model_name)
    blocks_args = BlockDecoder.decode(['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25'])
    global_params = GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, dropout_rate=p, drop_connect_rate=0.2, num_classes=1000, width_coefficient=w, depth_coefficient=d, depth_divisor=8, min_depth=None, image_size=s)
    if override_params: global_params = global_params._replace(**override_params)
    return blocks_args, global_params
