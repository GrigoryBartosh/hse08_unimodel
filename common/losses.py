import torch
import torch.nn as nn
import torch.autograd as autograd

__all__ = [
    'ImageReconstractionLoss', 'TextReconstractionLoss',
    'ImageEmbedReconstractionLoss', 'TextEmbedReconstractionLoss',
    'ContrastiveLoss', 'HardNegativeContrastiveLoss',
    'DiscriminatorGradientPenalty'
]


class ImageReconstractionLoss(nn.Module):
    def __init__(self):
        super(ImageReconstractionLoss, self).__init__()

    def forward(self, input, target):
        return torch.mean((input - target) ** 2)


class TextReconstractionLoss(nn.Module):
    def __init__(self):
        super(TextReconstractionLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target, mask):
        input = input.reshape(-1, input.shape[2])
        target = target.reshape(-1)
        mask = mask.reshape(-1)
        loss = self.ce_loss(input, target)
        loss = (loss * mask).sum() / mask.sum()
        return loss


class ImageEmbedReconstractionLoss(nn.Module):
    def __init__(self):
        super(ImageEmbedReconstractionLoss, self).__init__()

    def forward(self, input, target):
        return torch.mean((input - target) ** 2)


class TextEmbedReconstractionLoss(nn.Module):
    def __init__(self):
        super(TextEmbedReconstractionLoss, self).__init__()

    def forward(self, input, target):
        return torch.mean((input - target) ** 2)


class ContrastiveLoss(nn.Module):
    def __init__(self, metric='cos', margin=0.2, use_caps=True, use_imgs=True):
        super(ContrastiveLoss, self).__init__()

        self.metric = metric
        self.cos = nn.CosineSimilarity(dim=1)
        self.l2 = nn.PairwiseDistance(p=2)

        self.margin = margin
        self.use_caps = use_caps
        self.use_imgs = use_imgs

    def get_scores_diag(self, imgs, caps):
        if self.metric == 'dot':
            scores = torch.mm(imgs, caps.t())
        elif self.metric == 'cos':
            bs, dim = imgs.shape
            imgs = imgs[:, None, :].expand(bs, bs, dim).reshape(-1, dim)
            caps = caps[None, :, :].expand(bs, bs, dim).reshape(-1, dim)
            scores = self.cos(imgs, caps).reshape(bs, bs)
        elif self.metric == 'l2':
            bs, dim = imgs.shape
            imgs = imgs[:, None, :].expand(bs, bs, dim).reshape(-1, dim)
            caps = caps[None, :, :].expand(bs, bs, dim).reshape(-1, dim)
            scores = self.l2(imgs, caps).reshape(bs, bs) ** 2
            scores = -scores
        else:
            assert False, f"Unsupported metric: {self.metric}"

        diag = scores.diag()

        return scores, diag

    def forward(self, imgs, caps):
        scores, diag = self.get_scores_diag(imgs, caps)

        cost_c = torch.clamp(self.margin - diag.expand_as(scores) + scores, min=0)
        cost_i = torch.clamp(self.margin - diag.view(-1, 1).expand_as(scores) + scores, min=0)

        diag_c = torch.diag(cost_c.diag())
        diag_i = torch.diag(cost_i.diag())

        cost_c = torch.mean(cost_c - diag_c)
        cost_i = torch.mean(cost_i - diag_i)

        cost = 0
        if self.use_caps:
            cost = cost + cost_c
        if self.use_imgs:
            cost = cost + cost_i

        return cost


class HardNegativeContrastiveLoss(ContrastiveLoss):
    def __init__(self, nmax=1, metric='cos', margin=0.2, use_caps=True, use_imgs=True):
        super().__init__(metric, margin, use_caps, use_imgs)
        self.nmax = nmax

    def get_negs(self, imgs, caps):
        scores, diag = super().get_scores_diag(imgs, caps)

        scores = (scores - 2 * torch.diag(scores.diag()))

        sorted_cap, _ = torch.sort(scores, dim=0, descending=True)
        sorted_img, _ = torch.sort(scores, dim=1, descending=True)

        max_c = sorted_cap[:self.nmax, :]
        max_i = sorted_img[:, :self.nmax]

        neg_cap = self.margin - diag.view(1, -1).expand_as(max_c) + max_c
        neg_img = self.margin - diag.view(-1, 1).expand_as(max_i) + max_i
        neg_cap = torch.clamp(neg_cap, min=0)
        neg_img = torch.clamp(neg_img, min=0)
        neg_cap = torch.sum(neg_cap, dim=0)
        neg_img = torch.sum(neg_img, dim=1)

        return neg_cap, neg_img

    def forward(self, imgs, caps):
        neg_cap, neg_img = self.get_negs(imgs, caps)

        neg_cap = torch.mean(neg_cap)
        neg_img = torch.mean(neg_img)

        loss = 0
        if self.use_caps:
            loss = loss + neg_cap
        if self.use_imgs:
            loss = loss + neg_img

        return loss


def contrastive_loss_by_name(name):
    if name == 'ContrastiveLoss':
        return ContrastiveLoss
    elif name == 'HardNegativeContrastiveLoss':
        return HardNegativeContrastiveLoss
    else:
        assert False, f"Unsupported contrastive loss '{name}'"


class DiscriminatorGradientPenalty(nn.Module):
    def __init__(self):
        super(DiscriminatorGradientPenalty, self).__init__()

    def forward(self, model_dis, real_data, fake_data, device='cpu'):
        batch_size = real_data.shape[0]
        alpha = torch.rand(batch_size, *([1] * (len(real_data.shape) - 1)), device=device)
        alpha = alpha.expand_as(real_data)

        with torch.enable_grad():
            interpolates = alpha * real_data + (1 - alpha) * fake_data
            interpolates = autograd.Variable(interpolates, requires_grad=True)

            disc_interpolates = model_dis(interpolates)

            ones = torch.ones(disc_interpolates.size(), device=device)
            gradients = autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

        gradients = gradients.reshape(batch_size, -1)
        loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return loss