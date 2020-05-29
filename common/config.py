import os

__all__ = ['PATH']

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, '..')

PATH = {
    'DATASETS': {
        'COCO': {
            'TRAIN': {
                'IMAGES_DIR': os.path.join(ROOT_DIR, 'data', 'datasets', 'coco', 'images', 'train2014'),
                'CAPTIONS': os.path.join(ROOT_DIR, 'data', 'datasets', 'coco', 'annotations', 'captions_train2014.json')
            },
            'VAL': {
                'IMAGES_DIR': os.path.join(ROOT_DIR, 'data', 'datasets', 'coco', 'images', 'val2014'),
                'CAPTIONS': os.path.join(ROOT_DIR, 'data', 'datasets', 'coco', 'annotations', 'captions_val2014.json')
            },
            'VOCAB': os.path.join(ROOT_DIR, 'data', 'datasets', 'coco', 'annotations', 'vocab.pkl')
        }
    },
    'TF_LOGS': os.path.join(ROOT_DIR, 'data', 'logs_tf'),
    'MODELS': {
        'DIR': os.path.join(ROOT_DIR, 'data', 'models'),
        'WORD_EMBEDS': os.path.join(ROOT_DIR, 'data', 'models', 'word_embeddings.pth')
    }
}
