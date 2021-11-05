import numpy as np

from collections.abc import Sequence

from utils import preprocess
from utils.slicer import train_test_slice
from utils.radar import affine_transform, normalize_db as norm_db
from utils.visualization import spec_plot

import torch.utils.data as util_data


class _TimeInfo(list):
    def __init__(self, dataset, time_labels=("date", "offset"), dim="batch_index", drop_attrs=True):
        self._ds = dataset[list(time_labels)]
        if drop_attrs:
            self._ds.attrs = {}
        self.iter_dim = dim
        super(_TimeInfo, self).__init__()

    def __getitem__(self, index):
        return self._ds[{self.iter_dim: index}]

    def __setitem__(self, key, value):
        raise TypeError("Assignment operation not permitted for TimeInfo")

    def __len__(self):
        return self._ds.sizes[self.iter_dim]

    def __str__(self):
        return self._ds.__str__()


class RadarDataset(util_data.Dataset):
    """
    Base class to use as a dataset.
    Sequence superclass is straightforwardly exchangeable with the appropriate Tensorflow/Pytorch superclass
    """
    def __init__(self, recordings, slices=None, normalize_db=True, clip=(-40, 0), add_channel_dim=-3,
                 dim="batch_index", class_attr="activities", label_var="label", ignore_dims=None,
                 norm_values=None, feature_dtype=np.single, label_dtype=np.longlong):

        self.label = label_var
        self.iter_dim = dim
        self.class_attr = class_attr

        self.ftype = feature_dtype
        self.ltype = label_dtype

        process_funcs = []
        if normalize_db:
            preprocess.proc_register(process_funcs, norm_db, axis=(-1, -2))
        if clip is not None:
            preprocess.proc_register(process_funcs, np.clip, clip[0], clip[1])
            if norm_values is not None:  # Change the range of the data from `clip` to `norm_values`
                a = (norm_values[1] - norm_values[0])/(clip[1] - clip[0])
                b = norm_values[0] - a * clip[0]
                preprocess.proc_register(process_funcs, affine_transform, a, b)
        self._process_funcs = process_funcs

        if ignore_dims is True:
            self.ignore_dims = tuple(c for c in recordings[0].coords)
        else:
            self.ignore_dims = ignore_dims
        self._slices = slices
        self.channel_dim = add_channel_dim
        self._recordings = recordings
        self.attrs = dict_intersection([r.attrs for r in recordings])
        self.total_bytes = preprocess.get_size(recordings, loaded_only=False)

        self.dataset = self.Iterable(self)

        super(RadarDataset, self).__init__()

    def _access_item(self, index):
        if isinstance(index, int):
            item = preprocess.slice_recording(self._recordings, self._slices[index])
        elif isinstance(index, slice):
            item = preprocess.recs2dataset(self._recordings, self._slices[index], ignore_dims=self.ignore_dims)
        elif isinstance(index, np.ndarray):
            item = preprocess.recs2dataset(self._recordings, [self._slices[i] for i in index],
                                           ignore_dims=self.ignore_dims)
        else:
            raise IndexError(f"Only ints, np.int arrays and slices are accepted, got {type(index)}")
        return item

    def __getitem__(self, index):
        item = self._access_item(index)
        labels = self._get_labels(item).astype(self.ltype).values
        features_ds = preprocess.apply_processing(self._get_features(item), self._process_funcs)
        if self.channel_dim is not None:
            features_ds = features_ds.expand_dims(dim="channel", axis=self.channel_dim)

        features = [f.astype(self.ftype).values for f in features_ds.values()]
        return features, labels

    def plot(self, index, axes=None, **kwargs):
        features = self.dataset[index]
        spec_plot(features, axes=axes, **kwargs)

    def __len__(self):
        return len(self._slices)

    def _get_features(self, dataset):
        return dataset.drop(self.label)

    def _get_labels(self, dataset):
        return dataset[self.label]

    @property
    def feature_shapes(self):
        features, _ = self.__getitem__(0)
        length = (self.__len__(),)
        shapes = [length + f.shape for f in features]
        return shapes

    def scope(self, unit):
        item = self._access_item(0)
        return preprocess.get_scope(item, unit)

    @property
    def loaded_bytes(self):
        return preprocess.get_size(self._recordings, loaded_only=True)

    @property
    def class_num(self):
        return len(self.attrs[self.class_attr])

    @property
    def branches(self):
        return len(self.feature_shapes)

    class Iterable(Sequence):
        def __init__(self, radar_dataset):
            self._rds = radar_dataset
            super(RadarDataset.Iterable, self).__init__()

        def __len__(self):
            assert isinstance(self._rds, RadarDataset)
            return len(self._rds._slices)

        def __getitem__(self, index):
            assert isinstance(self._rds, RadarDataset)
            item = self._rds._access_item(index)
            ds = preprocess.apply_processing(self._rds._get_features(item), self._rds._process_funcs)
            return ds


def load_rd(config, preprocessed_path, spec_length, stride,
            range_length=None, doppler_length=None, train_load=0.8, gpu=False, split=None):
    """
    Split, slice and load preprocessed data as range/Doppler spectrograms to be used with Pytorch/Tensorflow
    :param config: Radar configuration to load, e.g. "E"
    :param preprocessed_path: Path to the preprocessed data
    :param spec_length: Length of the spectrograms in bins
    :param stride: Spectrogram stride or hop length for the slicing
    :param range_length: If provided, data will be cropped on the range axis from 0 to range_length
    :param doppler_length: If provided, data will be cropped on the doppler axis from 0 to doppler_length
    :param train_load: Target train load to perform train-test split
    :param gpu: If True, load the xarray.Datasets. This is normally desirable if training in the GPU cluster.
    :param split: Type of split.
    It can be None (no split), "deterministic", single", "double" (see train_test_slice) or a dictionary
    containing already split segments (useful for reproducibility)
    :return: RadarDataset objects with the data. Segments are also returned for reproducibility
    """
    recordings = preprocess.open_recordings(config, preprocessed_path, load=gpu,
                                            range_length=range_length, doppler_length=doppler_length)

    if split is None:
        slices = train_test_slice(recordings, spec_length, stride, train_load, split=split)
        rds = RadarDataset(recordings, slices=slices)
        return rds
    elif isinstance(split, dict):
        slices = train_test_slice(recordings, spec_length, stride, train_load,
                                  split=split, return_segments=False)
        rd_datasets = [RadarDataset(recordings, slices=s) for s in slices]
        return rd_datasets
    else:
        slices = train_test_slice(recordings, spec_length, stride, train_load,
                                  split=split, return_segments=True)
        segments = slices.pop(-1)
        rd_datasets = [RadarDataset(recordings, slices=s) for s in slices]
        return rd_datasets, segments


def dict_intersection(dictionaries):
    intersection = set() | dictionaries[0].items()
    for d in dictionaries:
        intersection &= d.items()
    return dict(intersection)
