import copy

import torch
import torch.nn as nn
import torchvision.models as models

from models.common import activation_by_name, norm_2d_by_name

__all__ = [
    'ImageEncoder', 'ImageDecoder',
    'ImageEmbedEncoder', 'ImageEmbedDecoder', 'ImageEmbedDiscriminator'
]


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1,
        stride=stride, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


def conv4x4(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=4,
        stride=stride, padding=1, bias=False
    )


def upsample(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(
        in_planes, out_planes,  kernel_size=3,
        stride=stride, padding=1,
        output_padding=int(stride > 1), bias=False
    )
    # TODO check ather
    #return nn.Sequential(
    #    nn.UpsamplingBilinear2d(scale_factor=stride),
    #    conv3x3(in_planes, out_planes)
    #)


class SimpleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes=None, stride=1,
                 activ='relu', norm_layer='BatchNorm', dropout_p=0,
                 upsample_block=False, scenario_num=1):
        super(SimpleBlock, self).__init__()

        if out_planes is None:
            out_planes = in_planes * self.expansion

        conv = upsample if upsample_block else conv3x3
        norm = norm_2d_by_name(norm_layer)

        self.conv1 = conv(in_planes, out_planes, stride)
        self.ln1 = nn.ModuleList([norm(out_planes) for _ in range(scenario_num)])
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.activ1 = activation_by_name(activ)

    def forward(self, x, scenario):
        out = self.conv1(x)
        out = self.ln1[scenario](out)
        out = self.dropout1(out)
        out = self.activ1(out)

        return out


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes=None, out_planes=None, stride=1,
                 activ='relu', norm_layer='BatchNorm', dropout_p=0,
                 upsample_block=False, scenario_num=1):
        super(ResBasicBlock, self).__init__()

        if planes is None:
            planes = in_planes // self.expansion

        if out_planes is None:
            out_planes = planes * self.expansion

        conv = upsample if upsample_block else conv3x3
        norm = norm_2d_by_name(norm_layer)

        self.residual_conv = None
        if stride != 1 or in_planes != out_planes:
            res_conv_block = upsample if upsample_block else conv1x1
            self.residual_conv = res_conv_block(in_planes, out_planes, stride)
            self.residual_ln = nn.ModuleList([
                norm(out_planes) for _ in range(scenario_num)
            ])
            self.residual_dropout = nn.Dropout(p=dropout_p)

        self.activ = activation_by_name(activ)

        self.conv1 = conv(in_planes, planes, stride)
        self.ln1 = nn.ModuleList([norm(planes) for _ in range(scenario_num)])
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.conv2 = conv(planes, out_planes)
        self.ln2 = nn.ModuleList([norm(out_planes) for _ in range(scenario_num)])
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, x, scenario):
        identity = x

        out = self.conv1(x)
        out = self.ln1[scenario](out)
        out = self.dropout1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.ln2[scenario](out)
        out = self.dropout2(out)

        if self.residual_conv is not None:
            identity = self.residual_conv(identity)
            identity = self.residual_ln[scenario](identity)
            identity = self.residual_dropout(identity)

        out += identity
        out = self.activ(out)

        return out


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes=None, out_planes=None, stride=1,
                 activ='relu', norm_layer='BatchNorm', dropout_p=0,
                 upsample_block=False, scenario_num=1):
        super(ResBottleneck, self).__init__()

        if planes is None:
            planes = in_planes // self.expansion

        if out_planes is None:
            out_planes = planes * self.expansion

        conv = upsample if upsample_block else conv3x3
        norm = norm_2d_by_name(norm_layer)

        self.residual_conv = None
        if stride != 1 or in_planes != out_planes:
            res_conv_block = upsample if upsample_block else conv1x1
            self.residual_conv = res_conv_block(in_planes, out_planes, stride)
            self.residual_ln = nn.ModuleList([
                norm(out_planes) for _ in range(scenario_num)
            ])
            self.residual_dropout = nn.Dropout(p=dropout_p)

        self.activ = activation_by_name(activ)

        self.conv1 = conv1x1(in_planes, planes)
        self.ln1 = nn.ModuleList([norm(planes) for _ in range(scenario_num)])
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.conv2 = conv(planes, planes, stride)
        self.ln2 = nn.ModuleList([norm(planes) for _ in range(scenario_num)])
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.conv3 = conv1x1(planes, out_planes)
        self.ln3 = nn.ModuleList([norm(out_planes) for _ in range(scenario_num)])
        self.dropout3 = nn.Dropout(p=dropout_p)

    def forward(self, x, scenario):
        identity = x

        out = self.conv1(x)
        out = self.ln1[scenario](out)
        out = self.dropout1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.ln2[scenario](out)
        out = self.dropout2(out)
        out = self.activ(out)

        out = self.conv3(out)
        out = self.ln3[scenario](out)
        out = self.dropout3(out)

        if self.residual_conv is not None:
            identity = self.residual_conv(identity)
            identity = self.residual_ln[scenario](identity)
            identity = self.residual_dropout(identity)

        out += identity
        out = self.activ(out)

        return out


def block_by_name(name):
    if name == 'SimpleBlock':
        return SimpleBlock
    elif name == 'ResBasicBlock':
        return ResBasicBlock
    elif name == 'ResBottleneck':
        return ResBottleneck
    else:
        assert False, f"Unsupported block: {name}"


class Encoder(nn.Module):
    def __init__(self, block, layer_sizes, planes=16, activ='relu',
                 norm_layer='BatchNorm', dropout_p=0,
                 first_kernel_size=7, scenario_num=1):
        super(Encoder, self).__init__()

        in_planes = planes
        self.out_planes = in_planes * 2 ** len(layer_sizes)

        norm = norm_2d_by_name(norm_layer)

        self.conv1 = nn.Conv2d(
            3, in_planes, kernel_size=first_kernel_size,
            stride=2, padding=first_kernel_size // 2, bias=False
        )
        self.ln1 = nn.ModuleList([norm(in_planes) for _ in range(scenario_num)])
        self.dropout1 = nn.Dropout(dropout_p)
        self.activ1 = activation_by_name(activ)

        layers = []
        for layer_size in layer_sizes:
            layers += [self._make_layer(block, in_planes, layer_size,
                                        activ, norm_layer, dropout_p, scenario_num)]
            in_planes = in_planes * 2
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, block, in_planes, layer_size, activ,
                    norm_layer, dropout_p, scenario_num):
        layers = [block(
            in_planes, out_planes=in_planes * 2,
            stride=2, activ=activ, norm_layer=norm_layer,
            dropout_p=dropout_p, scenario_num=scenario_num
        )]
        in_planes = in_planes * 2

        for _ in range(1, layer_size):
            layers += [block(in_planes, activ=activ, norm_layer=norm_layer,
                             dropout_p=dropout_p, scenario_num=scenario_num)]

        return nn.ModuleList(layers)

    def forward(self, x, scenario):
        x = self.conv1(x)
        x = self.ln1[scenario](x)
        x = self.dropout1(x)
        x = self.activ1(x)

        for layer in self.layers:
            for block in layer:
                x = block(x, scenario)

        return x


class Decoder(nn.Module):
    def __init__(self, block, layer_sizes, planes=16, activ='lrelu',
                 norm_layer='BatchNorm', dropout_p=0,
                 last_kernel_size=7, scenario_num=1):
        super(Decoder, self).__init__()

        in_planes = planes * 2 ** len(layer_sizes)

        layers = []
        for layer_size in layer_sizes:
            layers += [self._make_layer(block, in_planes, layer_size,
                                        activ, norm_layer, dropout_p, scenario_num)]
            in_planes = in_planes // 2
        self.layers = nn.ModuleList(layers)

        self.conv1 = nn.ConvTranspose2d(
            in_planes, 3, kernel_size=last_kernel_size, stride=2,
            padding=last_kernel_size // 2, output_padding=1, bias=False
        )
        self.activ1 = nn.Tanh()

    def _make_layer(self, block, in_planes, layer_size, activ,
                    norm_layer, dropout_p, scenario_num):
        layers = [block(
            in_planes, out_planes=in_planes // 2,
            stride=2, activ=activ, norm_layer=norm_layer,
            dropout_p=dropout_p, upsample_block=True, scenario_num=scenario_num
        )]
        in_planes = in_planes // 2

        for _ in range(1, layer_size):
            layers += [block(
                in_planes, activ=activ, norm_layer=norm_layer,
                dropout_p=dropout_p, upsample_block=True, scenario_num=scenario_num
            )]

        return nn.ModuleList(layers)

    def forward(self, x, scenario):
        for layer in self.layers:
            for block in layer:
                x = block(x, scenario)

        x = self.conv1(x)
        x = self.activ1(x)

        return x


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()

        block = block_by_name(args['block'])

        self.encoder = Encoder(
            block,
            layer_sizes=args['layers'],
            planes=args['planes'],
            activ=args['activ'],
            norm_layer=args['norm_layer'],
            dropout_p=args['dropout_p']
        )

    def forward(self, x, scenario=0):
        x = self.encoder(x, scenario)
        return x


class ImageDecoder(nn.Module):
    def __init__(self, args):
        super(ImageDecoder, self).__init__()

        block = block_by_name(args['block'])

        self.decoder = Decoder(
            block,
            layer_sizes=args['layers'],
            planes=args['planes'],
            activ=args['activ'],
            norm_layer=args['norm_layer'],
            dropout_p=args['dropout_p']
        )

    def forward(self, x, scenario=0):
        x = self.decoder(x, scenario)
        return x


class ImageEmbedEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEmbedEncoder, self).__init__()

        in_planes = args['image_embed_dim']

        block = block_by_name(args['block'])
        self.layer1 = nn.ModuleList([
            block(in_planes,  activ=args['activ'], norm_layer=args['norm_layer'],
                  dropout_p=args['dropout_p'], scenario_num=args['scenario_num'])
            for _ in range(args['conv_layer_size'])
        ])

        self.conv1 = nn.Conv2d(
            in_planes, args['latent_dim'], kernel_size=args['image_embed_size'],
            stride=args['image_embed_size'], padding=0, bias=False
        )

    def forward(self, x, scenario=0):
        for block in self.layer1:
            x = block(x, scenario)

        x = self.conv1(x)

        x = torch.squeeze(x)

        return x


class ImageEmbedDecoder(nn.Module):
    def __init__(self, args):
        super(ImageEmbedDecoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(
            args['latent_dim'], args['image_embed_dim'],
            kernel_size=args['image_embed_size'],
            stride=args['image_embed_size'], padding=0, bias=False
        )
        in_planes = args['image_embed_dim']
        norm = norm_2d_by_name(args['norm_layer'])
        self.ln1 = nn.ModuleList([norm(in_planes) for _ in range(args['scenario_num'])])
        self.dropout1 = nn.Dropout(p=args['dropout_p'])
        self.activ1 = activation_by_name(args['activ'])

        block = block_by_name(args['block'])
        self.layer2 = nn.ModuleList([
            block(in_planes, activ=args['activ'], norm_layer=args['norm_layer'],
                  dropout_p=args['dropout_p'], scenario_num=args['scenario_num'])
            for _ in range(args['conv_layer_size'])
        ])

        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )

    def forward(self, x, scenario=0):
        x = x[:, :, None, None]

        x = self.conv1(x)
        x = self.ln1[scenario](x)
        x = self.dropout1(x)
        x = self.activ1(x)

        for block in self.layer2:
            x = block(x, scenario)

        x = self.conv2(x)

        return x


class ImageEmbedDiscriminator(nn.Module):
    def __init__(self, args):
        super(ImageEmbedDiscriminator, self).__init__()

        args = copy.deepcopy(args)
        args['latent_dim'] = 1
        self.model = ImageEmbedEncoder(args)

    def forward(self, x, scenario=0):
        x = self.model(x, scenario)
        x = torch.squeeze(x)
        return x