import numpy as np
import threading
from dataset import Dataset
from block_sampler import BlockSampler

_bitmap = np.array([128, 64, 32, 16, 8, 4, 2, 1],
                  dtype='int').reshape((1, 8)).repeat(512, 0)

class BatchEncoder:
    def __init__(self, sampler: BlockSampler, batch_size, xs_encoder='one_hot'):
        self.lock = threading.Lock()
        self.sampler = iter(sampler)
        self.batch_size = batch_size
        if type(xs_encoder) == str:
            assert xs_encoder in ['one_hot', '264bits', '8bits01', "8bits_11", "16bits"]
            xs_encoder = globals()['xs_encoder_' + xs_encoder]
        self.xs_encoder = xs_encoder
        self.ys_encoder = mk_ys_encoder(sampler.dataset.cat_to_ix)

    def __iter__(self):
        while True:
            xs, ys = next(self)
            yield xs, ys

    def __next__(self):
        with self.lock:
            batch = []
            for _ in range(self.batch_size):
                sample = next(self.sampler)
                batch.append(sample)
            xs = self.xs_encoder([s.block for s in batch])
            ys = self.ys_encoder([s.category for s in batch])
            return xs, ys


def xs_encoder_one_hot(blocks):
    xs = np.zeros((len(blocks), 512, 256), dtype='int')
    for i, block in enumerate(blocks):
        block = np.array(block, dtype='int')
        xs[i] = one_hot(block, 256)
    return xs


def xs_encoder_264bits(blocks):
    xs = np.zeros((len(blocks), 512, 264), dtype='int')
    xs[:, :, :256] = xs_encoder_one_hot(blocks)
    xs[:, :, 256:] = xs_encoder_8bits_11(blocks)
    return xs

def xs_encoder_8bits01(blocks):
    xs = np.zeros((len(blocks), 512, 8), dtype='int')
    for i, block in enumerate(blocks):
        blk = block.reshape((512, 1)).repeat(8, 1)
        bits = np.bitwise_and(blk, _bitmap)/_bitmap
        xs[i] = bits
    return xs


def xs_encoder_8bits_11(blocks):
    xs = xs_encoder_8bits01(blocks)
    xs = xs * 2 - 1
    return xs

def decode_8bits_11(blocks):
    return np.sum((blocks + 1) // 2 * 2**np.arange(8)[::-1], axis=-1)

def xs_encoder_16bits(blocks):
    xs = np.zeros((len(blocks), 512, 16), dtype='int')
    xs8 = xs_encoder_8bits01(blocks)
    xs[:, :, :8] = xs8
    xs[:, :, 8:] = 1 - xs8
    return xs


def mk_ys_encoder(cat_to_ix):
    len_cats = len(cat_to_ix.keys())
    def ys_encoder(cats):
        ys = np.zeros((len(cats), len_cats), dtype='int')
        for i, cat in enumerate(cats):
            y = cat_to_ix[cat]
            ys[i] = one_hot(y, len_cats)
        return ys
    return ys_encoder


def one_hot(arr, num_categories):
    arr_shape = np.shape(arr)
    flatten = np.reshape(arr, -1)
    r = np.zeros((len(flatten), num_categories))
    r[np.arange(len(flatten)), flatten] = 1
    return r.reshape((*arr_shape, num_categories))
