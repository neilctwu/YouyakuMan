import pickle
import os
import random
import torch

from src.models.preprocess import Preprocess


def lazy_dataset(data_path, shuffle):
    data_path = data_path if data_path[-1] is '/' else data_path+'/'

    def _lazy_load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    pts = [data_path + x for x in os.listdir(data_path) if '.pickle' in x]
    if pts:
        if shuffle:
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_load(pt)
    else:
        raise IOError('Data not exist in {}'.format(data_path))


class Batch(object):
    def __init__(self, dataset=None, device=None,):
        """Create a Batch from a list of examples."""
        if dataset is not None:
            self.batch_size = len(dataset)
            pre_src = [x['src'] for x in dataset]
            pre_labels = [x['labels'] for x in dataset]
            pre_segs = [x['segs'] for x in dataset]
            pre_clss = [x['clss'] for x in dataset]
            pre_srctxt = [x['src_str'] for x in dataset]

            src = torch.tensor(pre_src)
            segs = torch.tensor(pre_segs)
            mask = ~(src == 0)

            labels = torch.tensor(self._pad(pre_labels, -1))
            clss = torch.tensor(self._pad(pre_clss, -1))
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0

            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src', src.to(device))
            setattr(self, 'labels', labels.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask', mask.to(device))
            setattr(self, 'src_str', pre_srctxt)

    def __len__(self):
        return self.batch_size

    @staticmethod
    def _pad(data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data


class DataLoader:
    def __init__(self, data_path, input_length, batch_size,
                 device='cuda', shuffle=True):
        self.data_path = data_path if data_path[-1] is '/' else data_path+'/'
        self.shuffle = shuffle
        self.preprocessor = Preprocess()
        self.input_len = input_length
        self.batch_size = batch_size
        self.device = device
        self.iterer = self._lazy_loader()

    def __next__(self):
        return self._next_batch()

    def _lazy_loader(self):
        def _lazy_load(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        # shuffle data in folders and load one per yield
        pts = [self.data_path + x for x in os.listdir(self.data_path) if '.pickle' in x]
        if pts:
            if self.shuffle:
                random.shuffle(pts)
            for pt in pts:
                data_dic = _lazy_load(pt)
                if (data_dic['summary'] != '')&(data_dic['body'] != ''):
                    yield _lazy_load(pt)
        else:
            raise IOError('Data not exist in {}'.format(self.data_path))

    def _next_batch(self):
        while True:
            try:
                dataset = []
                try:
                    # Drop the current dataset for decreasing memory
                    for i in range(self.batch_size):
                        rawdata = next(self.iterer)
                        data = self.preprocessor(rawdata, self.input_len)
                        dataset.append(data)
                    batched_dataset = Batch(dataset=dataset, device=self.device)
                    return batched_dataset
                except Exception as e:
                    print('DataWarning with data has error {}'.format(e))
            except StopIteration:
                self.iterer = self._lazy_loader()
                return None
