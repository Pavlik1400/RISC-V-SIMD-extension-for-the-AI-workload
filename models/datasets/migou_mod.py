from typing import Dict, List, Optional

from datasets.dataset import SignalModulationDataset
from sklearn.preprocessing import Normalizer
from tools.utils import is_str, some_are_nones
import numpy as np
import pickle


class MigouMod_Dataset(SignalModulationDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._modulations = None
        self._snrs = None

    def load(self, val, *args, force=False, **kwargs):
        """
        Additional parameters:
            to_1024 = False      - merge 8 consecutive frames into 1, legacy, works badly
            transpose = True     - swap last 2 dimensions
            minimum_snr = -10000 - filter result by minimum snr
        """
        if not force and not some_are_nones(
            self._labels, self._data, self._modulations, self._snrs
        ):
            return

        if is_str(val):
            (
                self._labels,
                self._data,
                self._modulations,
                self._snrs,
            ) = self.Detail.load_dataset(val, *args, **kwargs)
            print("Postprocessed and loaded MIGOU-MOD")
            return

        raise TypeError(f"load argument has bad type: {type(val)}")

    def get_snrs(self, *args, **kwargs) -> np.ndarray:
        assert self._snrs is not None, "Can't get snrs: dataset is not loaded"
        return self._snrs

    def get_modulations(self, *args, **kwargs) -> List:
        assert self._modulations is not None, "Can't get modulations: dataset is not loaded"
        return self._modulations

    class Detail:
        @staticmethod
        def extract_data(raw_ds: Dict):
            pass

        @staticmethod
        def load_dataset(ds_path: str, transpose=True, shuffle=False, normalize=False):
            with open(ds_path, "rb") as fp:
                raw_dataset = pickle.load(fp, encoding="latin1")

            # Raw dataset has the following representation:
            # Dict[(modulation name, snr(1m=37dB, 6m=22dB)) : [40k (2, 128) shaped lists]]

            modulations = np.unique(
                list(map(lambda x: x[0], raw_dataset.keys()))
            )  # All unique types of modulations
            # uniq_snrs = np.unique(
            #     list(map(lambda x: x[1], raw_dataset.keys()))
            # )  # All unique values of SNR
            snr_mapping = {"ota_1m": 37, "ota_6m": 22}

            # extract labels + data + snrs
            n_samples = sum([len(data) for data in raw_dataset.values()])
            sample_shape = raw_dataset[list(raw_dataset.keys())[0]][0].shape
            assert len(sample_shape) == 2, "1d data is only supported"
            print(f"Loaded {n_samples} samples with shape {sample_shape}")

            labels = np.empty(n_samples, dtype=np.uint8)
            if not transpose:
                data = np.empty((n_samples, *sample_shape), dtype=np.float32)
            else:
                data = np.empty((n_samples, sample_shape[1], sample_shape[0]), dtype=np.float32)
            snrs = np.empty_like(labels)
            modulation_name_to_label = {mod: i for i, mod in enumerate(modulations)}
            print(modulation_name_to_label)

            idx = 0
            for k in raw_dataset:
                (mod, s) = k
                cur_data = raw_dataset[k]
                curr_n_samples = len(cur_data)

                label = modulation_name_to_label[mod]
                labels[idx : idx + curr_n_samples] = np.ones(curr_n_samples) * label

                snr = snr_mapping[s]
                snrs[idx : idx + curr_n_samples] = np.ones(curr_n_samples) * snr

                if transpose:
                    cur_data = cur_data.transpose(0, 2, 1)
                data[idx : idx + curr_n_samples, :, :] = cur_data
                # for sample_i in range(curr_n_samples):
                #     sample = cur_data[sample_i]
                #     data[sample_i + idx, :, :] = np.transpose(sample) if transpose else sample

                if normalize:
                    assert transpose
                    # data[idx : idx + curr_n_samples, :, 0] = Normalizer(copy=False).fit_transform(
                    #     data[idx : idx + curr_n_samples, :, 0]
                    # )  # I samples
                    # data[idx : idx + curr_n_samples, :, 1] = Normalizer(copy=False).fit_transform(
                    #     data[idx : idx + curr_n_samples, :, 1]
                    # )  # Q samples
                    data[idx : idx + curr_n_samples, :, 0] -= data[idx : idx + curr_n_samples, :, 0].min()
                    data[idx : idx + curr_n_samples, :, 0] /= np.max(data[idx : idx + curr_n_samples, :, 0])
                    
                    data[idx : idx + curr_n_samples, :, 1] -= data[idx : idx + curr_n_samples, :, 1].min()
                    data[idx : idx + curr_n_samples, :, 1] /= np.max(data[idx : idx + curr_n_samples, :, 1])


                idx += curr_n_samples

            if shuffle:
                shuffle_index = np.arange(n_samples)
                np.random.shuffle(shuffle_index)
                data = data[shuffle_index]
                labels = labels[shuffle_index]
                snrs = snrs[shuffle_index]

            # if normalize:
            #     if transpose:
            #         data[:, :, 0] = (data[:, :, 0] - np.mean(data[:, :, 0])) / np.std(data[:, :, 0])
            #         data[:, :, 1] = (data[:, :, 1] - np.mean(data[:, :, 1])) / np.std(data[:, :, 1])
            #         # data[:, :, 0] = Normalizer(copy=False).fit_transform(data[:, :, 0])  # I samples
            #         # data[:, :, 1] = Normalizer(copy=False).fit_transform(data[:, :, 1])  # Q samples
            #     else:
            #         raise NotImplementedError
            #         data[:, 0, :] = Normalizer(copy=False).fit_transform(data[:, 0, :])  # I samples
            # data[:, 1, :] = Normalizer(copy=False).fit_transform(data[:, 1, :])  # Q samples

            return labels, data, modulations, snrs
