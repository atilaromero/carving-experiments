import numpy as np
from block_sampler import count_sectors, BlockSamplerByFile
from batch_encoder import xs_encoder_8bits_11, BatchEncoder
from dataset import Dataset
from report import Reporter
import report
import os
from collections import OrderedDict
import models
import tensorflow
from trainer import Trainer


def filter_dataset(model, dset):
    filtered = {}
    for filename in dset.filenames:
        sector_count = count_sectors(filename)
        filtered[filename] = np.zeros((sector_count,))
        with open(filename, 'rb') as infile:
            for sector in range(sector_count):
                b = infile.read(512)
                assert len(b) > 0
                blk = np.zeros((512,), dtype='int')
                blk[:len(b)] = [int(x) for x in b]
                ys = model.predict(xs_encoder_8bits_11([blk]))
                y = ys[0]
                filtered[filename][sector] = np.argmax(y, axis=-1)
    return filtered


def save_filtered(filtered, filepath):
    with open(filepath, 'ab') as f:
        for k, v in filtered.items():
            f.write('%s\t' % k)
            for n in v:
                f.write(n)
            f.write('\n')


def datasets_X_random(raw_dset, rnd_dset):
    by_category = raw_dset.by_category()
    for cat, dataset in by_category.items():
        dataset = dataset.join(rnd_dset, categories=[cat, 'zzz'])
        tset, vset = dataset.rnd_split_fraction_by_category(0.5)
        yield cat, tset, vset


def gen_rndchk_models(raw_dataset_folder,
                      random_dataset_folder,
                      minimum,
                      maximum,
                      result_dir):
    raw_dset = Dataset.new_from_folders(raw_dataset_folder)
    raw_dset = raw_dset.filter_min_max(minimum, maximum)

    rnd_dset = Dataset.new_from_folders(random_dataset_folder)
    rnd_dset = rnd_dset.filter_min_max(minimum, maximum)

    r = Reporter()
    for cat, tset, vset in datasets_X_random(raw_dset, rnd_dset):
        print(cat)
        model = models.C64_16_2pr_C32_4_2pr_C64_32_2pr_F_D(
            2, 8, 'softmax', 'categorical_crossentropy')
        result = Trainer(model).train(tset, vset)
        h5_path = os.path.join(result_dir, '%s_random.h5' % cat)
        tensorflow.keras.Model.save(model, h5_path)
        r.add(result,
              category=cat,
              **report.report_epochs(**result._asdict()),
              **report.report_elapsed(**result._asdict()),
              **report.report_metrics(**result._asdict()),
              )
    r.save_report(result_dir + "/experiments.tsv")


def evaluate_rnd_model(cat, model, dset):
    assert len(dset.categories) == 2
    bs = BlockSamplerByFile(dset)
    benc = BatchEncoder(
        bs, batch_size=400, xs_encoder=xs_encoder_8bits_11)
    result = model.evaluate_generator(iter(benc), steps=10)
    return OrderedDict(
        category=cat,
        **dict(zip(model.metrics_names, result)),
    )
