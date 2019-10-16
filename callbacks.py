import time
from tensorflow.keras.callbacks import Callback


class TimeIt(Callback):
    def on_train_begin(self, logs):
        self.start_time = time.time()

    def on_train_end(self, logs):
        self.elapsed = time.time()-self.start_time


class TimeLimit(Callback):
    def __init__(self, seconds_limit):
        self.seconds_limit = seconds_limit

    def on_train_begin(self, logs):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs):
        elapsed = time.time()-self.start_time
        if self.seconds_limit and elapsed > self.seconds_limit:
            self.model.stop_training = True


class SaveModel(Callback):
    def __init__(self, save_file, period):
        self.save_file = save_file
        self.period = period

    def on_epoch_end(self, epoch, logs):
        if epoch % self.period != 0:
            return
        self.model.save(self.save_file)


class MetricLimit(Callback):
    def __init__(self, metric, limit):
        self.limit = limit
        self.metric = metric

    def on_epoch_end(self, epoch, logs):
        if logs[self.metric] > self.limit:
            self.model.stop_training = True
