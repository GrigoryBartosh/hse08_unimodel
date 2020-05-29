import numpy as np

__all__ = ['RetrievalRecall']


class RetrievalRecall:
    def __init__(self, metric='cos', batch_size=1000, ks=[1, 5, 10], imgs_dupls=1):
        self.metric = metric

        self.batch_size = batch_size
        self.ks = ks
        self.imgs_dupls = imgs_dupls
        self.batch_items = self.batch_size * self.imgs_dupls

        self.imgs_enc = []
        self.caps_enc = []

        self.res = []

    def dot(self, imgs, caps):
        return np.dot(imgs, caps.T)

    def cos(self, imgs, caps):
        imgs_norm = np.linalg.norm(imgs, axis=1)
        caps_norm = np.linalg.norm(caps, axis=1)

        scores = np.dot(imgs, caps.T)

        norms = np.dot(np.expand_dims(imgs_norm, 1),
                       np.expand_dims(caps_norm.T, 1).T)

        scores = (scores / norms)

        return scores

    def l2(self, imgs, caps):
        imgs = np.expand_dims(imgs, 1)
        caps = np.expand_dims(caps, 0)

        scores = np.linalg.norm(imgs - caps, axis=-1) ** 2
        scores = -scores

        return scores

    def similarity(self, imgs, caps):
        if self.metric == 'dot':
            return self.dot(imgs, caps)
        elif self.metric == 'cos':
            return self.cos(imgs, caps)
        elif self.metric == 'l2':
            return self.l2(imgs, caps)
        else:
            assert False, f"Unsupported metric '{self.metric}'"

    def recall_by_ranks(self, ranks):
        recall_search = list()
        for k in self.ks:
            recall_search.append(
                len(np.where(ranks < k)[0]) / ranks.shape[0])
        return recall_search

    def recall_for_all(self, imgs_enc, caps_enc):
        imgs_dupls = self.imgs_dupls

        scores = self.similarity(imgs_enc[::imgs_dupls], caps_enc)

        ranks = np.array([np.nonzero(np.in1d(row, np.arange(x * imgs_dupls, x * imgs_dupls + imgs_dupls, 1)))[0][0]
                          for x, row in enumerate(np.argsort(scores, axis=1)[:, ::-1])])
        recall_caps_search = self.recall_by_ranks(ranks)

        ranks = np.array([np.nonzero(row == x // imgs_dupls)[0][0]
                          for x, row in enumerate(np.argsort(scores.T, axis=1)[:, ::-1])])
        recall_imgs_search = self.recall_by_ranks(ranks)

        return recall_caps_search, recall_imgs_search

    def add(self, imgs, caps):
        self.imgs_enc += list(imgs)
        self.caps_enc += list(caps)

        while len(self.imgs_enc) >= self.batch_items:
            imgs, self.imgs_enc = self.imgs_enc[:self.batch_items], self.imgs_enc[self.batch_items:]
            caps, self.caps_enc = self.caps_enc[:self.batch_items], self.caps_enc[self.batch_items:]
            imgs, caps = np.stack(imgs, axis=0), np.stack(caps, axis=0)
            self.res.append(self.recall_for_all(imgs, caps))

    def get_res(self):
        res = self.res
        if len(res) > 0:
            ans = [np.mean([x[i] for x in res], axis=0) for i in range(len(res[0]))]
        else:
            ans = [[None] * len(self.ks)] * 2

        self.res = []

        return ans
