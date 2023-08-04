import keras
from typing import Dict, List, Optional

from datasets.dataset import SignalModulationDataset
from tools.utils import is_str, some_are_nones, is_tuple
import numpy as np
import pickle
import h5py
import ast


class RadioML2018(SignalModulationDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._modulations = None
        self._snrs = None
        self._ds_file = None
        self._classes = None
        self._groups = None

    def load(self, val, *args, force=False, **kwargs):
        """
        Additional parameters:
            to_1024 = False      - merge 8 consecutive frames into 1, legacy, works badly
            transpose = True     - swap last 2 dimensions
            minimum_snr = -10000 - filter result by minimum snr
        """
        if not force and not some_are_nones(
            self._labels, self._data, self._modulations, self._snrs, self._ds_file, self._classes
        ):
            return

        if not is_tuple(val, 2, str):
            raise TypeError(
                f"load argument has bad type: {type(val)}. Expected path to h2py file and "
            )

        data, classes = val

        self._ds_file = h5py.File(data, "r")
        self._groups = list(self._ds_file.keys())
        self._snrs = self._ds_file[self._groups[self.Detail.SNR_IDX]][()]
        # Convert one-hot to regular
        self._labels = np.argmax(self._ds_file[self._groups[self.Detail.LABELS_IDX]], axis=1)
        # self._labels = self._ds_file[self._groups[self.Detail.LABELS_IDX]][()]
        self._modulations = self.Detail.load_classes(classes)

    def get_snrs(self, *args, **kwargs) -> np.ndarray:
        assert self._snrs is not None, "Can't get snrs: dataset is not loaded"
        return self._snrs

    def get_modulations(self, *args, **kwargs) -> List:
        assert self._modulations is not None, "Can't get modulations: dataset is not loaded"
        return self._modulations

    def get_data_element(self, idx: int):
        return self._ds_file[self._groups[self.Detail.DATA_IDX]][idx]

    def dump(self, how_or_where, *args, **kwargs):
        raise NotImplementedError("Dataset doesn't fit in memory, 'dump' impossible to implement")

    def get_data(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            "Dataset doesn't fit in memory, 'get_data' impossible to implement"
        )
    
    def get_dim(self):
        return (1024, 2)

    def split_train_val_test(
        self, train_perc: float, val_perc: float, force_resplit=False, *args, **kwargs
    ):
        raise NotImplementedError(
            "Dataset doesn't fit in memory, 'split_train_val_test' impossible to implement"
        )

    def get_split_indecies(self, *args, **kwargs):
        raise NotImplementedError(
            "Dataset doesn't fit in memory, 'get_split_indecies' impossible to implement"
        )

    def to_keras_generator(self, list_IDs: np.ndarray, batch_size=32, shuffle=True):
        return _RadioML2018Generator(list_IDs, self, batch_size, shuffle)

    class Detail:
        @staticmethod
        def load_classes(path: str) -> List[str]:
            str_repr = ""
            for line in open(path, "r"):
                line = line.strip()
                if line.startswith("classes"):
                    line = line.split(" ")[-1]
                str_repr += line
            return ast.literal_eval(str_repr)

        SNR_IDX = 2
        LABELS_IDX = 1
        DATA_IDX = 0


class _RadioML2018Generator(keras.utils.Sequence):
    def __init__(self, list_IDs: np.ndarray, RML2018_ds: RadioML2018, batch_size=32, shuffle=True):
        "Initialization"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = len(RML2018_ds.get_modulations())
        self.list_IDs = list_IDs
        self.n_samples = len(self.list_IDs)
        self.indexes = None
        self.ds = RML2018_ds
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_IDs_temp = self.list_IDs[indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.ds.get_dim()))
        # y = np.empty((self.batch_size, self.n_classes), dtype=int)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.ds.get_data_element(ID)

            # Store class
            y[i] = self.ds.get_labels()[ID]

        return X, y
