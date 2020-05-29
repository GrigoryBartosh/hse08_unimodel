import pickle

import torch
import torch.nn as nn

import numpy as np

from bpemb import BPEmb

from common.config import PATH

VOCAB_PATH = PATH['DATASETS']['COCO']['VOCAB']
EMBEDS_PATH = PATH['MODELS']['WORD_EMBEDS']


def save_vocab(vocab, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(vocab, file, pickle.HIGHEST_PROTOCOL)


def save_embeds(embeds, file_path):
    torch.save(embeds.state_dict(), file_path)


if __name__ == '__main__':
    dim = 300
    vs = 5000
    bpemb_en = BPEmb(lang="en", dim=dim, vs=vs)

    vocab = dict((i, i + 1) for i in range(vs))

    embeds = bpemb_en.vectors
    embeds = np.concatenate((np.zeros((1, dim)), embeds))
    embeds = torch.FloatTensor(embeds)
    embeds = nn.Embedding.from_pretrained(embeds)

    save_vocab(vocab, VOCAB_PATH)
    save_embeds(embeds, EMBEDS_PATH)