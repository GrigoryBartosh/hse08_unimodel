import torch
import torch.nn as nn

from models.common import mlp_by_name

__all__ = ['LatentDistribution', 'DistributionDiscriminator', 'LatentDiscriminator']


class LatentDistribution(nn.Module):
    def __init__(self, args):
        super(LatentDistribution, self).__init__()

        self.latent_dim = args['latent_dim']

    def forward(self, bs, device='cpu'):
        x = torch.randn(bs, self.latent_dim, device=device)
        return x


class DistributionDiscriminator(nn.Module):
    def __init__(self, args):
        super(DistributionDiscriminator, self).__init__()

        self.mlp = mlp_by_name(args['mlp'])(
            in_dim=args['latent_dim'],
            layer_dims=args['layers'] + [1],
            activ=args['activ'],
            norm_layer=args['norm_layer'],
            dropout_p=args['dropout_p']
        )

    def forward(self, x):
        x = self.mlp(x)
        x = torch.squeeze(x)
        return x


class LatentDiscriminator(nn.Module):
    def __init__(self, args):
        super(LatentDiscriminator, self).__init__()

        self.mlp = mlp_by_name(args['mlp'])(
            in_dim=args['latent_dim'],
            layer_dims=args['layers'] + [1],
            activ=args['activ'],
            norm_layer=args['norm_layer'],
            dropout_p=args['dropout_p']
        )

    def forward(self, x):
        x = self.mlp(x)
        x = torch.squeeze(x)
        return x