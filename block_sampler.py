import os
import math
import random
import numpy as np
from typing import Generator
from dataset import Dataset


class BlockInstance:
    def __init__(self, block, category):
        self.block = block
        self.category = category


class BlockSampler:
    def __init__(self, dataset: Dataset, group_by='by_file'):
        self.dataset = dataset
        assert group_by in ['by_file', 'by_sector']
        self.group_by = group_by

    def __iter__(self) -> Generator[BlockInstance, None, None]:
        filenames = list(self.dataset.filenames)
        assert len(filenames) > 0
        sectors = {}
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        while True:
            if self.group_by == 'by_file':
                files = random.sample(filenames, len(filenames))
            if self.group_by == 'by_sector':
                files = random.choices(*zip(*sectors.items()), k=1000)
            for f in files:
                sector = random.randrange(sectors[f])
                block = get_sector(f, sector)
                yield BlockInstance(block, self.dataset.category_from(f))


class BlockSamplerFirstBlock:
    def __init__(self, dataset: Dataset, **kwargs):
        self.dataset = dataset

    def __iter__(self) -> Generator[BlockInstance, None, None]:
        filenames = list(self.dataset.filenames)
        assert len(filenames) > 0
        sectors = {}
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        while True:
            files = random.sample(filenames, len(filenames))
            for f in files:
                sector = 0
                block = get_sector(f, sector)
                yield BlockInstance(block, self.dataset.category_from(f))


class BlockSamplerByCategory:
    def __init__(self, dataset: Dataset, ratio=0.9, **kwargs):
        self.dataset = dataset
        self.ratio = ratio
        self.category_prob = {}
        for cat in self.dataset.categories:
            self.category_prob[cat] = 1

    def __iter__(self):
        assert len(self.dataset.filenames) > 0
        sectors = {}
        filenames = self.dataset.filenames
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        datasets = self.dataset.by_category()
        while True:
            files = random.sample(filenames, len(filenames))
            assert len(files) > 0
            while len(files) > 0:
                if random.random() < self.ratio:
                    cats, probs = zip(*self.category_prob.items())
                    cat = random.choices(cats, probs)[0]
                    f = random.sample(datasets[cat].filenames, 1)[0]
                else:
                    f = files.pop()
                sector = random.randrange(sectors[f])
                block = get_sector(f, sector)
                yield BlockInstance(block, self.dataset.category_from(f))




class BlockSamplerByFile:
    def __init__(self, dataset: Dataset, **kwargs):
        self.dataset = dataset

    def __iter__(self) -> Generator[BlockInstance, None, None]:
        filenames = list(self.dataset.filenames)
        assert len(filenames) > 0
        sectors = {}
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        while True:
            files = random.sample(filenames, len(filenames))
            for f in files:
                sector = random.randrange(sectors[f])
                block = get_sector(f, sector)
                yield BlockInstance(block, self.dataset.category_from(f))


class BlockSamplerBySector:
    def __init__(self, dataset: Dataset, **kwargs):
        self.dataset = dataset

    def __iter__(self) -> Generator[BlockInstance, None, None]:
        filenames = list(self.dataset.filenames)
        assert len(filenames) > 0
        sectors = {}
        for filename in filenames:
            sectors[filename] = count_sectors(filename)
        while True:
            files = random.choices(*zip(*sectors.items()), k=1000)
            for f in files:
                sector = random.randrange(sectors[f])
                block = get_sector(f, sector)
                yield BlockInstance(block, self.dataset.category_from(f))


def count_sectors(filename):
    stat = os.stat(filename)
    return math.ceil(stat.st_size/512)


def get_sector(filename, sector):
    with open(filename, 'rb') as f:
        f.seek(sector*512, 0)
        b = f.read(512)
        assert len(b) > 0
        n = np.zeros((512), dtype='int')
        n[:len(b)] = [int(x) for x in b]
        return n


class RandomSampler:
    def __init__(self, blksampler, ratio=0.5, rnd_cat='random', not_rnd_cat='not_random', **kwargs):
        self.blksampler = blksampler
        self.ratio = ratio
        self.rnd_cat = rnd_cat
        self.not_rnd_cat = not_rnd_cat
        self.dataset = blksampler.dataset.clone(
            categories=[not_rnd_cat, rnd_cat])

    def __iter__(self):
        blkiter = iter(self.blksampler)
        while True:
            if random.random() > self.ratio:
                inst = next(blkiter)
                inst.category = self.not_rnd_cat
                yield inst
            else:
                blk = np.random.randint(0, 256, (512,), dtype='int')
                yield BlockInstance(blk, self.rnd_cat)
