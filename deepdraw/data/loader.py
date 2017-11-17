import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset

from .unpack import load


class BaseDataset(Dataset):
    def __init__(self, path, index):
        self.path = path
        self.index = index

    @classmethod
    def load(self, path):
        index = pd.read_csv(join(path, 'index.csv'))
        return self(path, index)

    def reduce(self, n, seed=None):
        """Reduce dataset to n drawings per class

        Args:
            n (int): number of drwaings per class
            seed (int): random seed

        Returns:
            {Base,Stroke,Raster}Dataset: new reduced dataset
        """

        rand = np.random.RandomState(seed)
        indices = self.index.groupby('word').indices
        res = []

        for word in sorted(self.index.word.unique()):
            v = rand.choice(indices[word], n, replace=False)
            res.append(v)

        res = np.concatenate(res)
        index = self.index.iloc[res]

        # return new instance of same class
        return type(self)(self.path, index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        word, offset = self.index.word[i], self.index.offset[i]
        return load(self.path, word, offset), word


class StrokeDataset(BaseDataset):
    def __getitem__(self, i):
        drawing, word = super().__getitem__(i)
        return drawing['drawing'], word


class RasterDataset(BaseDataset):
    # TODO: add transformation
    def __init__(self, path, index, center=True):
        super().__init__(path, index)
        self.center = center

    def __getitem__(self, i):
        drawing, word = super().__getitem__(i)
        return rasterize(drawing, self.center), word


def rasterize(drawing, center=False):
    """Rasterize drawing using Bresenham's line algorithm.

    Args:
        drawing (dict): drawing to rasterize, as returned by unpack_drawing()
        center (bool): move drawing to center resulting image
    Returns:
        numpy.ndarray: a 256x256 boolean numpy array
    """

    # TODO: resize image?

    drawing = drawing['drawing']
    if center:
        x_max = max(x for X, _ in drawing for x in X)
        y_max = max(y for _, Y in drawing for y in Y)

    image = np.zeros((256, 256), dtype=np.bool)
    for stroke in drawing:
        # stroke = [(x0, x1, ...), (y0, y1, ...)]
        # -> [(x0, y0), (x1, y1), ...]
        stroke = [*zip(*stroke)]
        # -> [((x0, y0), (x1, y1)),  ((x1, y1), (x2, y2)), ...]
        stroke = [*zip(stroke[:-1], stroke[1:])]

        for (x0, y0), (x1, y1) in stroke:
            if center:
                dx = 256 - x_max
                dy = 256 - y_max
                x0 += dx // 2
                x1 += dx // 2
                y0 += dy // 2
                y1 += dy // 2

            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = -1 if x0 > x1 else 1
            sy = -1 if y0 > y1 else 1

            x, y = x0, y0
            if dx > dy:
                err = dx / 2
                while x != x1:
                    image[x, y] = True
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
            else:
                err = dy / 2
                while y != y1:
                    image[x, y] = True
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy

            image[x, y] = True

    image = image.T  # Transpose to switch x and y axes
    return image
