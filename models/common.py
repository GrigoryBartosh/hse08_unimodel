import torch
import torch.nn as nn

__all__ = ['MLP', 'ResMLP', 'activation_by_name', 'mlp_by_name']


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def activation_by_name(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'selu':
        return nn.SELU(inplace=True)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'none':
        return Identity()
    else:
        assert False, f"Unsupported activation: {name}"


def norm_1d_by_name(name):
    if name == 'BatchNorm':
        return nn.BatchNorm1d
    elif name == 'InstanceNorm':
        return nn.InstanceNorm1d
    elif name == 'LayerNorm':
        return nn.LayerNorm
    elif name == 'none':
        return Identity
    else:
        assert False, f"Unsupported normalization: {name}"


def norm_2d_by_name(name):
    if name == 'BatchNorm':
        return nn.BatchNorm2d
    elif name == 'InstanceNorm':
        return nn.InstanceNorm2d
    elif name == 'none':
        return Identity
    else:
        assert False, f"Unsupported normalization: {name}"


class MLP(nn.Module):
    def __init__(self, in_dim, layer_dims, activ='relu', norm_layer='none', dropout_p=0):
        super(MLP, self).__init__()

        layers = []
        for dim in layer_dims[:-1]:
            layers += [
                nn.Linear(in_dim, dim),
                activation_by_name(activ),
                norm_1d_by_name(norm_layer)(dim),
                nn.Dropout(p=dropout_p)
            ]
            in_dim = dim
        layers += [nn.Linear(in_dim, layer_dims[-1])]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class ResBasicLinearBlock(nn.Module):
    def __init__(self, in_dim, dim=None, out_dim=None, activ='relu',
                 norm_layer='none', dropout_p=0):
        super(ResBasicLinearBlock, self).__init__()

        dim = in_dim if dim is None else dim
        out_dim = dim if out_dim is None else out_dim

        norm = norm_1d_by_name(norm_layer)

        self.residual_linear = None
        if in_dim != out_dim:
            self.residual_linear = nn.Linear(in_dim, out_dim, bias=False)
            self.residual_ln = norm(out_dim)
            self.residual_dropout = nn.Dropout(p=dropout_p)

        self.activ = activation_by_name(activ)

        self.linear1 = nn.Linear(in_dim, dim, bias=False)
        self.ln1 = norm(dim)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(dim, out_dim, bias=False)
        self.ln2 = norm(out_dim)
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.ln1(out)
        out = self.dropout1(out)
        out = self.activ(out)

        out = self.linear2(x)
        out = self.ln2(out)
        out = self.dropout2(out)

        if self.residual_linear is not None:
            identity = self.residual_linear(identity)
            identity = self.residual_ln(identity)
            identity = self.residual_dropout(identity)

        out += identity
        out = self.activ(out)

        return out


class ResMLP(nn.Module):
    def __init__(self, in_dim, layer_dims, activ='relu', norm_layer='none', dropout_p=0):
        super(ResMLP, self).__init__()

        layers = []
        for dim in layer_dims[:-1]:
            layers += [ResBasicLinearBlock(in_dim=in_dim, out_dim=dim, activ=activ,
                                           norm_layer=norm_layer, dropout_p=dropout_p)]
            in_dim = dim
        layers += [nn.Linear(in_dim, layer_dims[-1], bias=False)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


def mlp_by_name(name):
    if name == 'MLP':
        return MLP
    elif name == 'ResMLP':
        return ResMLP
    else:
        assert False, f"Unsupported mlp: {name}"