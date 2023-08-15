from datasets.dataset import SignalModulationDataset
from datasets.matlab_v2_dataset import MatlabV2
from datasets.radio_ml_2016 import RadioML2016
from datasets.radio_ml_2018 import RadioML2018
from datasets.migou_mod import MigouMod_Dataset
from enum import Enum


class DatasetName(Enum):
    MATLAB_V2 = 0
    RADIOML_2016 = 1
    RADIOML_2018 = 2
    MIGOU_MOD = 3


__datasets_map = {
    DatasetName.MATLAB_V2: MatlabV2,
    DatasetName.RADIOML_2016: RadioML2016,
    DatasetName.RADIOML_2018: RadioML2018,
    DatasetName.MIGOU_MOD: MigouMod_Dataset,
}


def make_sigmod_ds(name: DatasetName, *args, **kwargs) -> SignalModulationDataset:
    if name not in __datasets_map:
        raise ValueError(f"Unknown dataset: {name}")
    return __datasets_map[name](*args, **kwargs)


def list_available_datasets():
    return list(__datasets_map.keys())
