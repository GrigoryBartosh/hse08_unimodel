import os
import pickle

import numpy as np

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from transformers import BertTokenizer
from bpemb import BPEmb

from PIL import Image
from pycocotools.coco import COCO

from common.config import PATH

__all__ = ['get_loader_unimodel']


def get_image_transform(image_size, do_image_aug):
    AUG_MAX_PERCENT = 0.2

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    def aug_image(x):
        x = transforms.RandomHorizontalFlip()(x)
        size = int(image_size * (1 + np.random.rand() * AUG_MAX_PERCENT))
        x = transforms.Resize((size, size))(x)
        return x

    return transforms.Compose([
        transforms.Lambda(lambda x: aug_image(x) if do_image_aug else x),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])


class ImageGetter():
    def __init__(self, transform):
        self.transform = transform

    def get(self, path):
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image


class TextEncoderBert():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)

    def encode(self, text):
        tokens = [self.tokenizer.cls_token]
        tokens += self.tokenizer.tokenize(text)
        tokens += [self.tokenizer.sep_token]
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens


class TextEncoder():
    def __init__(self, vocab_path):
        with open(vocab_path, 'rb') as file:
            self.vocab = pickle.load(file)

        self.bpemb_en = BPEmb(lang="en", dim=300, vs=1000)

    def encode(self, text):
        token_ids = self.bpemb_en.encode_ids(text)
        ids = np.array([self.vocab[t] for t in token_ids])
        return ids


class DatasetImages(data.Dataset):
    def _check_filename(filename):
        return filename.endswith(('.png', '.jpg'))

    def __init__(self, imgs_dir, transform=None):
        self.paths = []
        for (dirpath, dirnames, filenames) in os.walk(imgs_dir):
            filenames = filter(DatasetImages._check_filename, filenames)
            filenames = map(lambda f: (os.path.join(dirpath, f), f), filenames)
            self.paths += list(filenames)

        self.image_getter = ImageGetter(transform)

    def __getitem__(self, index):
        path, name = self.paths[index]
        img = self.image_getter.get(path)
        return img

    def __len__(self):
        return len(self.paths)


class DatasetTextCOCO(data.Dataset):
    def __init__(self, captions_path):
        self.coco = COCO(captions_path)
        self.ids = list(self.coco.anns.keys())

    def __getitem__(self, index):
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']

        return caption

    def __len__(self):
        return len(self.ids)


class DatasetTextCOCOW2V(data.Dataset):
    def __init__(self, captions_path, vocab_path):
        self.dataset = DatasetTextCOCO(captions_path)
        self.text_encoder = TextEncoder(vocab_path)

    def __getitem__(self, index):
        text = self.dataset[index]
        ids = self.text_encoder.encode(text)
        return ids

    def __len__(self):
        return len(self.dataset)


class DatasetDictEmbeds(data.Dataset):
    def __init__(self, embeds_path):
        with open(embeds_path, 'rb') as file:
            self.embeds = pickle.load(file)

        self.embeds = list(self.embeds.values())

    def __getitem__(self, index):
        embed = self.embeds[index]
        embed = torch.FloatTensor(np.array(embed))
        return embed

    def __len__(self):
        return len(self.embeds)


class DatasetCOCO(data.Dataset):
    def __init__(self, imgs_dir, captions_path, vocab_path, transform=None):
        self.coco = COCO(captions_path)
        self.ids = list(self.coco.anns.keys())

        image_id_to_ids = {}
        for i in self.ids:
            image_id = self.coco.anns[i]['image_id']
            if image_id not in image_id_to_ids:
                image_id_to_ids[image_id] = []
            image_id_to_ids[image_id].append(i)

        image_ids_gen = filter(lambda i: len(image_id_to_ids[i]) >= 5, image_id_to_ids.keys())
        self.ids = []
        for image_id in image_ids_gen:
            self.ids += image_id_to_ids[image_id][:5]

        self.imgs_dir = imgs_dir
        self.image_getter = ImageGetter(transform)
        self.text_encoder = TextEncoder(vocab_path)

    def __getitem__(self, index):
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        img_path = coco.loadImgs(img_id)[0]['file_name']

        img_path = os.path.join(self.imgs_dir, img_path)
        image = self.image_getter.get(img_path)

        caption_ids = self.text_encoder.encode(caption)

        return image, caption_ids

    def __len__(self):
        return len(self.ids)


def collate_images(images):
    images = torch.stack(images, axis=0)
    return images


def collate_texts(ids):
    lens = [len(i) for i in ids]
    max_len = max(lens)

    ids = [np.concatenate((i, np.zeros(max_len - l))) for i, l in zip(ids, lens)]
    masks = [[1] * l + [0] * (max_len - l) for l in lens]

    ids = torch.LongTensor(ids)
    masks = torch.LongTensor(masks)

    return ids, masks


def collate_embeds(embeds):
    embeds = torch.stack(embeds, axis=0)
    return embeds


def collate_parallel(data):
    images, text_ids = zip(*data)

    images = collate_images(images)
    text_ids, text_masks = collate_texts(text_ids)

    # last argument - is data parallel
    return images, (text_ids, text_masks), True


def infinit_data_loader(data_loader):
    while True:
        for x in data_loader:
            yield x


def union_data_loaders(data_loader_a, data_loader_b):
    for (a, b) in zip(data_loader_a, data_loader_b):
        # last argument - is data parallel
        yield a, b, False


def merge_data_loaders(data_loader_a, data_loader_b, ratio=0.5):
    try:
        while True:
            if np.random.random() < ratio:
                yield next(data_loader_a)
            else:
                yield next(data_loader_b)
    except:
        pass


def get_data_loader_images(dataset_name, sset, transform, batch_size, shuffle, num_workers):
    assert dataset_name in PATH['DATASETS'], f"Unknown dataset_name value '{dataset_name}'"

    dataset_images = DatasetImages(
        imgs_dir=PATH['DATASETS'][dataset_name][sset]['IMAGES_DIR'],
        transform=transform
    )

    data_loader_images = data.DataLoader(
        dataset=dataset_images,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_images,
        drop_last=True
    )

    return infinit_data_loader(data_loader_images)


def get_data_loader_texts(dataset_name, sset, batch_size, shuffle, num_workers):
    if dataset_name == 'COCO':
        dataset_text = DatasetTextCOCOW2V(
            captions_path=PATH['DATASETS'][dataset_name][sset]['CAPTIONS'],
            vocab_path=PATH['DATASETS'][dataset_name]['VOCAB']
        )
    else:
        assert False, f"Unknown dataset_name value '{dataset_name}'"

    data_loader_texts = data.DataLoader(
        dataset=dataset_text,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_texts,
        drop_last=True
    )

    return infinit_data_loader(data_loader_texts)


def get_data_loader_parallel(dataset_name, sset, transform,
                             batch_size, shuffle, num_workers):
    if dataset_name == 'COCO':
        dataset_parallel = DatasetCOCO(
            imgs_dir=PATH['DATASETS'][dataset_name][sset]['IMAGES_DIR'],
            captions_path=PATH['DATASETS'][dataset_name][sset]['CAPTIONS'],
            vocab_path=PATH['DATASETS'][dataset_name]['VOCAB'],
            transform=transform
        )
    else:
        assert False, f"Unknown dataset_name value '{dataset_name}'"

    data_loader_parallel = data.DataLoader(
        dataset=dataset_parallel,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_parallel,
        drop_last=True
    )

    return infinit_data_loader(data_loader_parallel)


def get_loader_unimodel(dataset_images_name=None, dataset_texts_name=None,
                        dataset_parallel_name=None, dataset_parallel_ratio=0.5,
                        sset='VAL', do_image_aug=False, image_size=256,
                        batch_size=16, shuffle=False, num_workers=8):
    assert not ((dataset_images_name is None) != (dataset_texts_name is None))

    image_transform = get_image_transform(image_size, do_image_aug)

    data_loader = None
    if dataset_images_name or dataset_texts_name:
        data_loader_images = get_data_loader_images(
            dataset_images_name,
            sset,
            image_transform,
            batch_size,
            shuffle,
            max(1, num_workers // 2)
        )

        data_loader_texts = get_data_loader_texts(
            dataset_texts_name,
            sset,
            batch_size,
            shuffle,
            max(1, num_workers // 2)
        )

        data_loader = union_data_loaders(data_loader_images, data_loader_texts)

    if dataset_parallel_name == 'COCO':
        data_loader_parallel = get_data_loader_parallel(
            dataset_parallel_name,
            sset,
            image_transform,
            batch_size,
            shuffle,
            num_workers
        )

        if data_loader:
            data_loader = merge_data_loaders(
                data_loader_parallel,
                data_loader,
                ratio=dataset_parallel_ratio
            )
        else:
            data_loader = data_loader_parallel

    return data_loader
