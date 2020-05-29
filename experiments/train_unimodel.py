import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from common.config import PATH
from common.logging import get_summary_writer, write_arguments
from common.data_loading import get_loader_unimodel
from common.model import UniModelWrapper, MODEL_SCENARIO
from common.utils import freeze_model

import common.losses as losses
import common.metrics as metrics

image_size = 128
image_in_planes = 16
image_encoder_layers = 5
text_embed_dim = 1024
latent_dim = 1024
metric = 'cos'
image_embed_dim = image_in_planes * 2 ** image_encoder_layers
image_embed_size = image_size // 2 ** (image_encoder_layers + 1)
args = {
    'load_model': True,
    'model_name': 'unimodel',
    'model': {
        'image_enc': {
            'block': 'SimpleBlock',
            'layers': [2] * image_encoder_layers,
            'planes': image_in_planes,
            'activ': 'relu',
            'norm_layer': 'BatchNorm',
            'dropout_p': 0
        },
        'image_dec': {
            'block': 'SimpleBlock',
            'layers': [2] * image_encoder_layers,
            'planes': image_in_planes,
            'activ': 'lrelu',
            'norm_layer': 'BatchNorm',
            'dropout_p': 0
        },
        'text_enc': {
            'rnn': 'LSTM',
            'embeds_path': PATH['MODELS']['WORD_EMBEDS'],
            'hidden_size': text_embed_dim,
            'num_layers': 2,
            'dropout_p': 0.2,
            'bidirectional': True
        },
        'text_dec': {
            'rnn': 'LSTM',
            'hidden_size': text_embed_dim,
            'num_layers': 2,
            'dropout_p': 0.2,
            'bidirectional': False,
            'vocab_size': 5001
        },
        'image_embed_enc': {
            'image_embed_dim': image_embed_dim,
            'image_embed_size': image_embed_size,
            'latent_dim': latent_dim,
            'block': 'ResBasicBlock',
            'conv_layer_size': 3,
            'activ': 'relu',
            'norm_layer': 'BatchNorm',
            'dropout_p': 0.2,
            'scenario_num': 2
        },
        'image_embed_dec': {
            'image_embed_dim': image_embed_dim,
            'image_embed_size': image_embed_size,
            'latent_dim': latent_dim,
            'block': 'ResBasicBlock',
            'conv_layer_size': 3,
            'activ': 'lrelu',
            'norm_layer': 'BatchNorm',
            'dropout_p': 0.2,
            'scenario_num': 3
        },
        'text_embed_enc': {
            'mlp': 'ResMLP',
            'text_embed_dim': text_embed_dim,
            'layers': [latent_dim] * 3 + [latent_dim],
            'activ': 'relu',
            'norm_layer': 'none',
            'dropout_p': 0.2
        },
        'text_embed_dec': {
            'mlp': 'ResMLP',
            'latent_dim': latent_dim,
            'layers': [latent_dim] * 3 + [text_embed_dim],
            'activ': 'lrelu',
            'norm_layer': 'none',
            'dropout_p': 0.2
        },
        'distr': {
            'latent_dim': latent_dim
        },
        'image_embed_dis': {
            'image_embed_dim': image_embed_dim,
            'image_embed_size': image_embed_size,
            'block': 'ResBasicBlock',
            'conv_layer_size': 2,
            'activ': 'relu',
            'norm_layer': 'InstanceNorm',
            'dropout_p': 0.2,
            'scenario_num': 1
        },
        'text_embed_dis': {
            'mlp': 'ResMLP',
            'text_embed_dim': text_embed_dim,
            'layers': [text_embed_dim] * 2,
            'activ': 'relu',
            'norm_layer': 'none',
            'dropout_p': 0.2
        },
        'latent_dis': {
            'mlp': 'ResMLP',
            'latent_dim': latent_dim,
            'layers': [latent_dim] * 2,
            'activ': 'relu',
            'norm_layer': 'none',
            'dropout_p': 0.2
        },
        'distr_dis': {
            'mlp': 'ResMLP',
            'latent_dim': latent_dim,
            'layers': [latent_dim] * 2,
            'activ': 'relu',
            'norm_layer': 'none',
            'dropout_p': 0.2
        },
    },
    'train': {
        'dataset_image_embeds_name': 'COCO',
        'dataset_text_embeds_name': 'COCO',
        'dataset_parallel_name': None,
        'dataset_parallel_ratio': None,
        'val_dataset_name': 'COCO',
        'image_size': image_size,
        'save_iter': 50000,
        'val_iter': 50000,
        'val_iter_count': 5000,
        'batch_size': 128,
        'num_workers': 16,
        'freeze_image_aut': True,
        'freeze_text_embeds': True,
        'freeze_text_aut': True,
        'use_image_embed_enc_scenario': True,
        'use_image_embed_dec_scenario': True,
        'use_image_embed_dis_scenario': False,
        'w_l2_norm': 0,
        'w_loss_recon': 10.0,
        'w_loss_embed_recon': 10.0,
        'w_loss_cycle': 0.0,
        'w_loss_domain_dis': 0.0,
        'w_loss_latent_dis': 0.0,
        'w_loss_distr_dis': 0.0,
        'w_loss_retrieval': 0.0,
        'w_lambda_gp': 0.0,
        'image_enc_grad_clip': 10 ** 10,
        'image_dec_grad_clip': 10 ** 10,
        'text_enc_grad_clip': 1.0,
        'text_dec_grad_clip': 1.0,
        'image_embed_enc_grad_clip': 1.0,
        'image_embed_dec_grad_clip': 1.0,
        'text_embed_enc_grad_clip': 0.1,
        'text_embed_dec_grad_clip': 0.1,
        'distr_grad_clip': 1.0,
        'image_embed_dis_grad_clip': 1.0,
        'text_embed_dis_grad_clip': 1.0,
        'latent_dis_grad_clip': 1.0,
        'distr_dis_grad_clip': 1.0,
        'image_embed_dis_val_clip': 0.01,
        'text_embed_dis_val_clip': 0.01,
        'latent_dis_val_clip': 0.01,
        'distr_dis_val_clip': 0.01,
        'lr': 0.001,
        'loss_retrieval': {
            'name': 'ContrastiveLoss',
            'metric': metric,
            'margin': 0.2
        },
        'recall': {
            'do_it': True,
            'metric': metric,
            'batch_size': 1000,
            'ks': [1, 5, 10],
            'imgs_dupls': 5
        }
    }
}


class Trainer():
    LOG_SAMPLE_STEP = 5
    LOG_IMAGE_NUM = 5

    def __init__(self, model, args, device, summary_writer):
        self.model = model
        self.args = args
        self.device = device
        self.summary_writer = summary_writer

        self.criterion_image_recon = losses.ImageReconstractionLoss()
        self.criterion_text_recon = losses.TextReconstractionLoss()
        self.criterion_image_embed_recon = losses.ImageEmbedReconstractionLoss()
        self.criterion_text_embed_recon = losses.TextEmbedReconstractionLoss()

        self.criterion_retrieval = losses.contrastive_loss_by_name(
            self.args['loss_retrieval']['name'])(
                metric=self.args['loss_retrieval']['metric'],
                margin=self.args['loss_retrieval']['margin']
            )

        self.criterion_gp = losses.DiscriminatorGradientPenalty()

        models_gen = [
            self.model.image_enc, self.model.image_dec,
            self.model.text_enc, self.model.text_dec,
            self.model.image_embed_enc, self.model.image_embed_dec,
            self.model.text_embed_enc, self.model.text_embed_dec,
            self.model.distr
        ]
        self.optimizer_gen = optim.Adam(
            filter(
                lambda p: p.requires_grad,
                [w for m in models_gen if m for w in m.parameters()]
            ),
            lr=self.args['lr'],
            weight_decay=self.args['w_l2_norm']
        )

        models_dis = [
            self.model.image_embed_dis, self.model.text_embed_dis,
            self.model.latent_dis, self.model.distr_dis
        ]
        self.optimizer_dis = optim.Adam(
            filter(
                lambda p: p.requires_grad,
                [w for m in models_dis if m for w in m.parameters()]
            ),
            lr=self.args['lr'],
            weight_decay=self.args['w_l2_norm']
        )

        self.sc = MODEL_SCENARIO(
            IMAGE_EMBED_ENC_PI=0,
            IMAGE_EMBED_ENC_WTCI=1 * int(self.args['use_image_embed_enc_scenario']),
            IMAGE_EMBED_DEC_PIC=0,
            IMAGE_EMBED_DEC_WTC=1 * int(self.args['use_image_embed_dec_scenario']),
            IMAGE_EMBED_DEC_PICTC=2 * int(self.args['use_image_embed_dec_scenario']),
            IMAGE_EMBED_DIS_PI=0,
            IMAGE_EMBED_DIS_WTCI=1 * int(self.args['use_image_embed_dis_scenario'])
        )

        self.all_val_parallel_losses = []
        self.all_val_gen_losses = []
        self.all_val_dis_losses = []

        self.metric_recall = metrics.RetrievalRecall(
            metric=self.args['recall']['metric'],
            batch_size=self.args['recall']['batch_size'],
            ks=self.args['recall']['ks'],
            imgs_dupls=self.args['recall']['imgs_dupls']
        )

    def log_parallel_losses(self, phase, losses, n_iter):
        names = ['image_embed_recon', 'text_embed_recon',
                 'retrieval', 'total']
        for name, loss in zip(names, losses):
            self.summary_writer.add_scalar(f'{phase}_parallel/{name}', loss, n_iter)

    def log_gen_losses(self, phase, losses, n_iter):
        names = ['image_recon', 'text_recon',
                 'image_embed_recon', 'text_embed_recon',
                 'image_embed_cycle_recon', 'text_embed_cycle_recon',
                 'retrieval', 'total']
        for name, loss in zip(names, losses):
            self.summary_writer.add_scalar(f'{phase}_gen/{name}', loss, n_iter)

    def log_dis_losses(self, phase, losses, n_iter):
        names = ['image_embed_dis', 'text_embed_dis',
                 'latent_dis', 'distr_dis',
                 'total']
        for name, loss in zip(names, losses):
            self.summary_writer.add_scalar(f'{phase}_dis/{name}', loss, n_iter)

    def log_grads(self, phase, gen_grads, dis_grads, n_iter):
        names = ['image_enc', 'image_dec',
                 'text_enc', 'text_dec',
                 'image_embed_enc', 'image_embed_dec',
                 'text_embed_enc', 'text_embed_dec',
                 'distr',
                 'image_embed_dis', 'text_embed_dis',
                 'latent_dis', 'distr_dis']
        for name, loss in zip(names, [*gen_grads, *dis_grads]):
            self.summary_writer.add_scalar(f'{phase}_grad/{name}', loss, n_iter)

    def log_images(self, phase, x_p, n_iter):
        x_p = x_p[::Trainer.LOG_SAMPLE_STEP][:Trainer.LOG_IMAGE_NUM]

        x_pi = self.model.image_enc(x_p)
        x_pip = self.model.image_dec(x_pi)

        img_grid = torchvision.utils.make_grid(
            torch.cat((x_p, x_pip)),
            normalize=True,
            nrow=x_p.size(0)
        )
        self.summary_writer.add_image(f'{phase}_recon', img_grid, n_iter)

    def calc_parallel_losses(self, x_p, x_w_ids, x_w_mask):
        x_pi = self.model.image_enc(x_p)
        x_wt = self.model.text_enc(x_w_ids, x_w_mask)

        x_pic = self.model.image_embed_enc(x_pi, self.sc.IMAGE_EMBED_ENC_PI)
        x_wtc = self.model.text_embed_enc(x_wt)

        x_pict = self.model.text_embed_dec(x_pic)
        x_wtci = self.model.image_embed_dec(x_wtc, self.sc.IMAGE_EMBED_DEC_WTC)

        loss_image_embed_recon = self.criterion_image_embed_recon(x_wtci, x_pi)
        loss_text_embed_recon = self.criterion_text_embed_recon(x_pict, x_wt)

        loss_retrieval = self.criterion_retrieval(x_pic, x_wtc)

        loss_total = loss_image_embed_recon * self.args['w_loss_embed_recon'] + \
                     loss_text_embed_recon * self.args['w_loss_embed_recon'] + \
                     loss_retrieval * self.args['w_loss_retrieval']

        losses = np.array([
            loss_image_embed_recon.item(), loss_text_embed_recon.item(),
            loss_retrieval.item(), loss_total.item()
        ])

        return losses, loss_total

    def calc_gen_losses(self, x_p, x_w_ids, x_w_mask):
        x_pi = self.model.image_enc(x_p)
        x_wt = self.model.text_enc(x_w_ids, x_w_mask)

        x_pip = self.model.image_dec(x_pi)
        x_wtw = self.model.text_dec(x_wt, max_len=x_w_ids.shape[1])

        x_pic = self.model.image_embed_enc(x_pi, self.sc.IMAGE_EMBED_ENC_PI)
        x_wtc = self.model.text_embed_enc(x_wt)

        x_pict = self.model.text_embed_dec(x_pic)
        x_wtci = self.model.image_embed_dec(x_wtc, self.sc.IMAGE_EMBED_DEC_WTC)

        x_pici = self.model.image_embed_dec(x_pic, self.sc.IMAGE_EMBED_DEC_PIC)
        x_wtct = self.model.text_embed_dec(x_wtc)

        x_pictc = self.model.text_embed_enc(x_pict)
        x_wtcic = self.model.image_embed_enc(x_wtci, self.sc.IMAGE_EMBED_ENC_WTCI)

        x_pictci = self.model.image_embed_dec(x_pictc, self.sc.IMAGE_EMBED_DEC_PICTC)
        x_wtcict = self.model.text_embed_dec(x_wtcic)

        x_c = self.model.distr(x_p.size(0) * 2, self.device)

        x_pict_dis = self.model.text_embed_dis(x_pict)
        x_wtci_dis = self.model.image_embed_dis(x_wtci, self.sc.IMAGE_EMBED_DIS_WTCI)

        x_pic_dis = self.model.latent_dis(x_pic)
        x_wtc_dis = self.model.latent_dis(x_wtc)

        x_pic_distr_dis = self.model.distr_dis(x_pic)
        x_wtc_distr_dis = self.model.distr_dis(x_wtc)
        x_c_distr_dis = self.model.distr_dis(x_c)

        loss_image_recon = self.criterion_image_recon(x_pip, x_p)
        loss_text_recon = self.criterion_text_recon(x_wtw, x_w_ids, x_w_mask)

        loss_image_embed_recon = self.criterion_image_embed_recon(x_pici, x_pi)
        loss_text_embed_recon = self.criterion_text_embed_recon(x_wtct, x_wt)

        loss_image_embed_cycle_recon = self.criterion_image_embed_recon(x_pictci, x_pi)
        loss_text_embed_cycle_recon = self.criterion_text_embed_recon(x_wtcict, x_wt)

        loss_image_embed_dis = -x_wtci_dis.mean()
        loss_text_embed_dis = -x_pict_dis.mean()

        loss_latent_dis = x_pic_dis.mean() - x_wtc_dis.mean()

        loss_distr_dis = x_c_distr_dis.mean() - \
                         (x_pic_distr_dis.mean() + x_wtc_distr_dis.mean()) / 2

        loss_retrieval = self.criterion_retrieval(torch.cat((x_pic, x_wtcic)),
                                                  torch.cat((x_pictc, x_wtc)))

        loss_total = loss_image_recon * self.args['w_loss_recon'] + \
                     loss_text_recon * self.args['w_loss_recon']  + \
                     loss_image_embed_recon * self.args['w_loss_embed_recon'] + \
                     loss_text_embed_recon * self.args['w_loss_embed_recon'] + \
                     loss_image_embed_cycle_recon * self.args['w_loss_cycle'] + \
                     loss_text_embed_cycle_recon * self.args['w_loss_cycle'] + \
                     loss_image_embed_dis * self.args['w_loss_domain_dis'] + \
                     loss_text_embed_dis * self.args['w_loss_domain_dis'] + \
                     loss_latent_dis * self.args['w_loss_latent_dis'] + \
                     loss_distr_dis * self.args['w_loss_distr_dis'] + \
                     loss_retrieval * self.args['w_loss_retrieval']

        losses = np.array([
            loss_image_recon.item(), loss_text_recon.item(),
            loss_image_embed_recon.item(), loss_text_embed_recon.item(),
            loss_image_embed_cycle_recon.item(), loss_text_embed_cycle_recon.item(),
            loss_retrieval.item(), loss_total.item()
        ])

        return losses, loss_total

    def calc_dis_losses(self, x_p, x_w_ids, x_w_mask):
        x_pi = self.model.image_enc(x_p)
        x_wt = self.model.text_enc(x_w_ids, x_w_mask)

        x_pic = self.model.image_embed_enc(x_pi, self.sc.IMAGE_EMBED_ENC_PI)
        x_wtc = self.model.text_embed_enc(x_wt)

        x_pict = self.model.text_embed_dec(x_pic)
        x_wtci = self.model.image_embed_dec(x_wtc, self.sc.IMAGE_EMBED_DEC_WTC)

        x_c = self.model.distr(x_p.size(0) * 2, self.device)

        x_pi_dis = self.model.image_embed_dis(x_pi, self.sc.IMAGE_EMBED_DIS_PI)
        x_wt_dis = self.model.text_embed_dis(x_wt)

        x_pict_dis = self.model.text_embed_dis(x_pict)
        x_wtci_dis = self.model.image_embed_dis(x_wtci, self.sc.IMAGE_EMBED_DIS_WTCI)

        x_pic_dis = self.model.latent_dis(x_pic)
        x_wtc_dis = self.model.latent_dis(x_wtc)

        x_pic_distr_dis = self.model.distr_dis(x_pic)
        x_wtc_distr_dis = self.model.distr_dis(x_wtc)
        x_c_distr_dis = self.model.distr_dis(x_c)

        w_gp = self.args['w_lambda_gp']

        loss_image_embed_dis = -x_pi_dis.mean() + x_wtci_dis.mean() + \
                               w_gp * self.criterion_gp(self.model.image_embed_dis,
                                                        x_pi, x_wtci, device)
        loss_text_embed_dis = -x_wt_dis.mean() + x_pict_dis.mean() + \
                              w_gp * self.criterion_gp(self.model.text_embed_dis,
                                                       x_wt, x_pict, device)

        loss_latent_dis = -x_pic_dis.mean() + x_wtc_dis.mean() + \
                          w_gp * self.criterion_gp(self.model.latent_dis,
                                                   x_pic, x_wtc, device)

        loss_distr_dis = -x_c_distr_dis.mean() + \
                         (x_pic_distr_dis.mean() + x_wtc_distr_dis.mean()) / 2 + \
                         w_gp * self.criterion_gp(self.model.distr_dis,
                                                  x_c, torch.cat((x_pic, x_wtc)), device)

        loss_total = loss_image_embed_dis * self.args['w_loss_domain_dis'] + \
                     loss_text_embed_dis * self.args['w_loss_domain_dis'] + \
                     loss_latent_dis * self.args['w_loss_latent_dis'] + \
                     loss_distr_dis * self.args['w_loss_distr_dis']

        losses = np.array([
            loss_image_embed_dis.item(), loss_text_embed_dis.item(),
            loss_latent_dis.item(), loss_distr_dis.item(),
            loss_total.item()
        ])

        return losses, loss_total

    def clip_grads(self, models, arg_keys):
        grads = []
        for model, key in zip(models, arg_keys):
            grads += [nn.utils.clip_grad.clip_grad_norm_(model.parameters(), self.args[key])]
        return grads

    def clip_gen_grads(self):
        models = [self.model.image_enc, self.model.image_dec,
                  self.model.text_enc, self.model.text_dec,
                  self.model.image_embed_enc, self.model.image_embed_dec,
                  self.model.text_embed_enc, self.model.text_embed_dec,
                  self.model.distr]
        arg_keys = ['image_enc_grad_clip', 'image_dec_grad_clip',
                    'text_enc_grad_clip', 'text_dec_grad_clip',
                    'image_embed_enc_grad_clip', 'image_embed_dec_grad_clip',
                    'text_embed_enc_grad_clip', 'text_embed_dec_grad_clip',
                    'distr_grad_clip']
        return self.clip_grads(models, arg_keys)

    def clip_dis_grads(self):
        models = [self.model.image_embed_dis, self.model.text_embed_dis,
                  self.model.latent_dis, self.model.distr_dis]
        arg_keys = ['image_embed_dis_grad_clip', 'text_embed_dis_grad_clip',
                    'latent_dis_grad_clip', 'distr_dis_grad_clip']
        return self.clip_grads(models, arg_keys)

    def clip_weights(self, models, arg_keys):
        for model, key in zip(models, arg_keys):
            val = self.args[key]
            for p in model.parameters():
                p.data.clamp_(-val, val)

    def clip_dis_weights(self):
        models = [self.model.image_embed_dis, self.model.text_embed_dis,
                  self.model.latent_dis, self.model.distr_dis]
        arg_keys = ['image_embed_dis_val_clip', 'text_embed_dis_val_clip',
                    'latent_dis_val_clip', 'distr_dis_val_clip']
        self.clip_weights(models, arg_keys)

    def step(self, batch, n_iter):
        x_p, (x_w_ids, x_w_mask), is_batch_parallel = batch

        x_p = x_p.to(self.device, non_blocking=True)
        x_w_ids = x_w_ids.to(self.device, non_blocking=True)
        x_w_mask = x_w_mask.to(self.device, non_blocking=True)

        if is_batch_parallel:
            self.optimizer_gen.zero_grad()
            losses, loss_total = self.calc_parallel_losses(x_p, x_w_ids, x_w_mask)
            loss_total.backward()
            self.optimizer_gen.step()

            self.log_parallel_losses('Train', losses, n_iter)
        else:
            self.optimizer_gen.zero_grad()
            gen_losses, gen_loss_total = self.calc_gen_losses(x_p, x_w_ids, x_w_mask)
            gen_loss_total.backward()
            gen_grads = self.clip_gen_grads()
            self.optimizer_gen.step()

            self.optimizer_dis.zero_grad()
            dis_losses, dis_loss_total = self.calc_dis_losses(x_p, x_w_ids, x_w_mask)
            dis_loss_total.backward()
            dis_grads = self.clip_dis_grads()
            self.optimizer_dis.step()
            self.clip_dis_weights()

            self.log_gen_losses('Train', gen_losses, n_iter)
            self.log_dis_losses('Train', dis_losses, n_iter)
            self.log_grads('Train', gen_grads, dis_grads, n_iter)

    def eval(self, batch, train_n_iter, val_n_iter):
        x_p, (x_w_ids, x_w_mask), _ = batch

        x_p = x_p.to(self.device, non_blocking=True)
        x_w_ids = x_w_ids.to(self.device, non_blocking=True)
        x_w_mask = x_w_mask.to(self.device, non_blocking=True)

        with torch.no_grad():
            parallel_losses, _ = self.calc_parallel_losses(x_p, x_w_ids, x_w_mask)
            gen_losses, _ = self.calc_gen_losses(x_p, x_w_ids, x_w_mask)
            dis_losses, _ = self.calc_dis_losses(x_p, x_w_ids, x_w_mask)

            x_pi = self.model.image_enc(x_p)
            x_wt = self.model.text_enc(x_w_ids, x_w_mask)
            x_pic = self.model.image_embed_enc(x_pi)
            x_wtc = self.model.text_embed_enc(x_wt)
            x_pic = x_pic.cpu().numpy()
            x_wtc = x_wtc.cpu().numpy()

            self.all_val_parallel_losses += [parallel_losses]
            self.all_val_gen_losses += [gen_losses]
            self.all_val_dis_losses += [dis_losses]

            self.metric_recall.add(x_pic, x_wtc)

            if val_n_iter >= self.args['val_iter_count']:
                self.log_parallel_losses(
                    'Validation',
                    np.stack(self.all_val_parallel_losses, axis=1).mean(axis=1),
                    train_n_iter
                )
                self.log_gen_losses(
                    'Validation',
                    np.stack(self.all_val_gen_losses, axis=1).mean(axis=1),
                    train_n_iter
                )
                self.log_dis_losses(
                    'Validation',
                   np.stack(self.all_val_dis_losses, axis=1).mean(axis=1),
                   train_n_iter
                )

                self.all_val_parallel_losses = []
                self.all_val_gen_losses = []
                self.all_val_dis_losses = []

                recall_res = self.metric_recall.get_res()
                for i, k in enumerate(self.args["recall"]["ks"]):
                    if recall_res[0][i] is not None:
                        self.summary_writer.add_scalar(
                            f'Recall/search_text_by_image_k={k}', recall_res[0][i], train_n_iter)
                    if recall_res[1][i] is not None:
                        self.summary_writer.add_scalar(
                            f'Recall/search_image_by_text_k={k}', recall_res[1][i], train_n_iter)

                self.log_images('Validation', x_p, train_n_iter)


if __name__ == '__main__':
    summary_writer = get_summary_writer('um')
    write_arguments(summary_writer, args)

    image_size = args['train']['image_size']
    batch_size = args['train']['batch_size']
    num_workers = args['train']['num_workers']
    train_data_loader = get_loader_unimodel(
        dataset_images_name=args['train']['dataset_image_embeds_name'],
        dataset_texts_name=args['train']['dataset_text_embeds_name'],
        dataset_parallel_name=args['train']['dataset_parallel_name'],
        dataset_parallel_ratio=args['train']['dataset_parallel_ratio'],
        sset='TRAIN',
        do_image_aug=True,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_data_loader = get_loader_unimodel(
        dataset_parallel_name=args['train']['val_dataset_name'],
        sset='VAL',
        do_image_aug=False,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = UniModelWrapper(args['model'], name=args['model_name'], load=args['load_model'])
    if args['train']['freeze_image_aut']:
        freeze_model(model.image_enc)
        freeze_model(model.image_dec)
    if args['train']['freeze_text_embeds']:
        freeze_model(model.text_enc.embeds)
    if args['train']['freeze_text_aut']:
        freeze_model(model.text_enc)
        freeze_model(model.text_dec)
    model.to(device)

    trainer = Trainer(model, args['train'], device, summary_writer)

    model.train()
    iter_last_save = 0
    iter_last_val = 0
    for batch_num, batch in enumerate(train_data_loader, 1):
        batch_size = args['train']['batch_size']
        iter_num = batch_num * batch_size

        trainer.step(batch, iter_num)

        iter_last_save += batch_size
        iter_last_val += batch_size

        if iter_last_save >= args['train']['save_iter']:
            model.save()
            iter_last_save = 0

        if iter_last_val >= args['train']['val_iter']:
            model.eval()

            for val_batch_num, val_batch in enumerate(val_data_loader, 1):
                val_iter_num = val_batch_num * batch_size

                trainer.eval(val_batch, iter_num, val_iter_num)

                if val_iter_num >= args['train']['val_iter_count']:
                    break

            model.train()
            iter_last_val = 0