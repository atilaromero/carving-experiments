import os
import datetime
from collections import OrderedDict


def report_elapsed(elapsed, **kwargs):
    m, s = divmod(elapsed, 60)
    return {
        'Time': "{:d}m{:02d}s".format(int(m), int(s)),
    }


def report_epochs(history, **kwargs):
    epochs_count = len(history.epoch)
    return {
        'Epochs': epochs_count,
    }


def report_metrics(history, metrics, **kwargs):
    result = {}
    for metric in metrics:
        result[metric] = history.history[metric][-1]
    return result


def report_name(model, **kwargs):
    return {
        'Name': model.name,
    }


def save_report(data, tsv_path):
    save_reports([data], tsv_path)


def save_reports(many_data, tsv_path):
    keys = many_data[0].keys()
    with open(tsv_path, 'a') as f:
        f.write('\t'.join(keys))
        f.write('\n')
        for r in many_data:
            values = [str(r[k]) for k in keys]
            f.write('\t'.join(values))
            f.write('\n')


class Reporter:
    def __init__(self):
        self.reports = []

    def add(self, funcs, **kwargs):
        report_data = OrderedDict()
        report_data.update(kwargs)
        self.reports.append(report_data)

    def save_report(self, filepath):
        save_reports(self.reports, filepath)


class Reporter2:
    def __init__(self, outpath):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        self.f = open(outpath, 'ab')
        self.first_line = True

    def line(self, **kwargs):
        if self.first_line:
            self.f.write(str.encode('\t'.join(kwargs.keys())))
            self.f.write(str.encode('\n'))
            self.first_line = False
        self.f.write(str.encode('\t'.join([str(x) for x in kwargs.values()])))
        self.f.write(str.encode('\n'))
        self.f.flush()

    def close(self):
        self.f.close()


def mk_result_dir(exp_number):
    time_dir = datetime.datetime.now().isoformat()[:19].replace(':', '-')
    model_dir = os.path.join('results', 'exp%s' % exp_number, time_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir
