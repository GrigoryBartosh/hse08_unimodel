import os
from collections import namedtuple

import torch
import torch.nn as nn

from common.config import PATH
import common.utils as utils

import models.image as image
import models.text as text
import models.latent as latent

__all__ = ['FullModelWrapper']

MODELS_DIR = PATH['MODELS']['DIR']
MODEL_STATE_EXT = '.pth'

MODEL_SCENARIO = namedtuple('MODEL_SCENARIO', [
    'IMAGE_EMBED_ENC_PI', 'IMAGE_EMBED_ENC_WTCI',
    'IMAGE_EMBED_DEC_PIC', 'IMAGE_EMBED_DEC_WTC', 'IMAGE_EMBED_DEC_PICTC',
    'IMAGE_EMBED_DIS_PI', 'IMAGE_EMBED_DIS_WTCI'
])


def save_model(model, path, params=None):
    torch.save({
            'params': params,
            'model_state_dict': model.state_dict()
        }, path)


def load_model(model_type, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    params = checkpoint['params']
    model = model_type(params) if type(params) is dict else model_type(*params)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, params


class UniModelWrapper():
    def __init__(self, args, name=None, load=False, device='cpu'):
        self.path = os.path.join(MODELS_DIR, name if name else 'unimodel')

        self.image_enc_type = image.ImageEncoder if args['image_enc'] else None
        self.image_dec_type = image.ImageDecoder if args['image_dec'] else None
        self.text_enc_type = text.TextEncoder if args['text_enc'] else None
        self.text_dec_type = text.TextDecoder if args['text_dec'] else None
        self.image_embed_enc_type = image.ImageEmbedEncoder if args['image_embed_enc'] else None
        self.image_embed_dec_type = image.ImageEmbedDecoder if args['image_embed_dec'] else None
        self.text_embed_enc_type = text.TextEmbedEncoder if args['text_embed_enc'] else None
        self.text_embed_dec_type = text.TextEmbedDecoder if args['text_embed_dec'] else None
        self.distr_type = latent.LatentDistribution if args['distr'] else None
        self.image_embed_dis_type = image.ImageEmbedDiscriminator if args['image_embed_dis'] else None
        self.text_embed_dis_type = text.TextEmbedDiscriminator if args['text_embed_dis'] else None
        self.latent_dis_type = latent.LatentDiscriminator if args['latent_dis'] else None
        self.distr_dis_type = latent.DistributionDiscriminator if args['distr_dis'] else None

        self.model_names = [
            'image_enc', 'image_dec',
            'text_enc', 'text_dec',
            'image_embed_enc', 'image_embed_dec',
            'text_embed_enc', 'text_embed_dec',
            'distr',
            'image_embed_dis', 'text_embed_dis',
            'latent_dis', 'distr_dis'
        ]

        for model_name in self.model_names:
            ldict = {'self': self}
            exec(f'model_type = self.{model_name}_type', {}, ldict)
            if ldict['model_type']:
                if load:
                    model, model_params = load_model(
                        ldict['model_type'],
                        os.path.join(self.path, model_name + MODEL_STATE_EXT),
                        device=device
                    )
                else:
                    model_params = args[model_name]
                    model = ldict['model_type'](model_params)
            else:
                model_params = None
                model = None
            ldict['model_params'] = model_params
            ldict['model'] = model
            exec(f'self.{model_name}_params = model_params', {}, ldict)
            exec(f'self.{model_name} = model', {}, ldict)

    def _apply(self, foo):
        for model_name in self.model_names:
            ldict = {'self': self}
            exec(f'model = self.{model_name}', {}, ldict)
            exec(f'model_params = self.{model_name}_params', {}, ldict)
            if ldict['model']:
                foo(ldict['model'], ldict['model_params'], model_name)

    def get_all_models(self):
        models = []
        foo = lambda model, params, name: models.append(model)
        self._apply(foo)
        return models

    def save(self):
        utils.remove_dir(self.path)
        utils.make_dir(self.path)
        foo = lambda model, params, name: save_model(
                model,
                os.path.join(self.path, name + MODEL_STATE_EXT),
                params
            )
        self._apply(foo)

    def to(self, device):
        foo = lambda model, params, name: model.to(device)
        self._apply(foo)

    def train(self):
        foo = lambda model, params, name: model.train()
        self._apply(foo)

    def eval(self):
        foo = lambda model, params, name: model.eval()
        self._apply(foo)