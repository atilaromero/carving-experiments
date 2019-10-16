import models
from block_sampler import BlockSampler
from batch_encoder import BatchEncoder
from collections import namedtuple
import callbacks
from tensorflow.keras.callbacks import EarlyStopping

TrainResults = namedtuple(
    'TrainResults', ['model', 'history', 'metrics', 'elapsed'])


class Trainer:
    def __init__(self,
                 model,
                 group_by='by_file',
                 xs_encoder='8bits_11',
                 validation_steps=20,
                 steps_per_epoch=10,
                 epochs=10000000,
                 max_seconds=10*60,
                 batch_size=400,
                 min_delta=1e-03,
                 patience=4,
                 blockSampler=BlockSampler,
                 batchEncoder=BatchEncoder):
        self.model = model
        self.group_by = group_by
        self.xs_encoder = xs_encoder
        self.validation_steps = validation_steps
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.max_seconds = max_seconds
        self.batch_size = batch_size
        self.min_delta = min_delta
        self.patience = patience
        self.blockSampler = blockSampler
        self.batchEncoder = batchEncoder

    def train(self, tset, vset):
        tsampler = self.blockSampler(tset, group_by=self.group_by)
        tbenc = self.batchEncoder(tsampler, self.batch_size,
                                  xs_encoder=self.xs_encoder)

        vsampler = self.blockSampler(vset, group_by=self.group_by)
        vbenc = self.batchEncoder(vsampler, self.batch_size,
                                  xs_encoder=self.xs_encoder)

        model = self.model

        timeIt = callbacks.TimeIt()

        history = model.fit_generator(iter(tbenc),
                                      validation_data=iter(vbenc),
                                      validation_steps=self.validation_steps,
                                      steps_per_epoch=self.steps_per_epoch,
                                      epochs=self.epochs,
                                      verbose=0,
                                      callbacks=[
            timeIt,
            # callbacks.SaveModel(os.path.join(result_dir, model.name + '.h5')),
            callbacks.TimeLimit(self.max_seconds),
            EarlyStopping(monitor='val_categorical_accuracy',
                          min_delta=self.min_delta, patience=self.patience),
            # TensorBoard(
            #     log_dir=os.path.join(log_dir, model.name),
            #     # update_freq=3100,
            # ),
        ],
            use_multiprocessing=False,
            workers=0,
        )
        return TrainResults(
            model=model,
            history=history,
            metrics=['val_binary_accuracy', 'val_categorical_accuracy'],
            elapsed=timeIt.elapsed,
        )
