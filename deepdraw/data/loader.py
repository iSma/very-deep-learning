import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from .unpack import load


class DrawDataset(Dataset):
    def __init__(self, path, index, transform=None):
        self.path = path
        self.index = index.reset_index(drop=True)
        self.transform = transform
        self.classes = LabelEncoder().fit_transform(index.word)

    @classmethod
    def load(self, path, transform=None):
        index = pd.read_csv(join(path, 'index.csv'))
        return self(path, index, transform)

    @property
    def words(self):
        """Returns a sorted list of all words appearing in this dataset."""
        return [*sorted(self.index.word.unique())]

    def select(self, classes):
        """Creates a new dataset containing only drawings of given classes.

        Args:
            classes (list of str): classes to keep

        Returns:
            DrawDataset: new reduced dataset
        """
        map = self.index.word.map(lambda w: w in classes)
        index = self.index.loc[map]

        return type(self)(self.path, index, self.transform)

    def reduce(self, n, seed=None):
        """Reduce dataset to `n` drawings per class

        Args:
            n (int): number of drawings per class
            seed (int): random seed

        Returns:
            DrawDataset: new reduced dataset
        """

        rand = np.random.RandomState(seed)
        indices = self.index.groupby('word').indices
        res = []

        for word in self.words:
            v = rand.choice(indices[word], n, replace=False)
            res.append(v)

        res = np.concatenate(res)
        index = self.index.iloc[res]

        return type(self)(self.path, index, self.transform)

    def split(self, sizes):
        """Splits the dataset into subsets of relative size given by `sizes`.

        Can be used to split into test/train/validation sets.

        Example:
            dataset.split([0.75, 0.15, 0.1]) splits the dataset into 3 parts,
            the first containing 75% of the drawings, the second 15% and the
            third 10%.
            Sizes are normalized to sum up to 1, so [0.75, 0.15, 0.1],
            [75, 15, 10] or [15, 3, 2] are all equivalent.

        Args:
            sizes (list of float): sizes of the splits

        Returns:
            list of list of int: A list containing one list of indices for each
                element of `sizes`. Can be used with `SubsetRandomSampler`.

        """
        sizes = np.array(sizes) / np.sum(sizes)
        sizes = np.cumsum(sizes)

        indices = self.index.groupby('word').indices
        res = [[] for _ in sizes]

        for word in self.words:
            i = indices[word]
            s = sizes[:-1] * len(i)
            s = s.round().astype(np.int)
            s = np.split(i, s)

            for r, s in zip(res, s):
                r += list(s)

        return res

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        word, offset = self.index.word[i], self.index.offset[i]
        drawing = load(self.path, word, offset)

        if self.transform:
            drawing = self.transform(drawing)

        return drawing, self.classes[i]
