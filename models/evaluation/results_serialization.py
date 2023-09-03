from typing import Any, Optional, Dict
from tensorflow.keras import Model
import json
import numpy as np
import os
import sys
import dataclasses

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from models.fabric import ModelName, load_model_configuration, make_sigmod_model
from datasets.fabric import DatasetName


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
    
def _dump_results(where: str, results: Dict[str, Any]):
    with open(where, "w") as results_file:
        json.dump(results, results_file, indent=4, cls=EnhancedJSONEncoder)


def get_model_parameters(model: Model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams
    return int(totalParams)


def dump_results(
    where: str,
    model: Model,
    model_config: Any,
    model_name: ModelName,
    dataset_name: DatasetName,
    dataset_path: str,
    dataset_params: Optional[Dict] = None,
    train_history: Optional[Dict] = None,
    cm_val: Optional[np.ndarray] = None,
    cls_to_acc_val: Optional[Dict] = None,
    snr_to_acc_val: Optional[Dict] = None,
    cm_test: Optional[np.ndarray] = None,
    cls_to_acc_test: Optional[Dict] = None,
    snr_to_acc_test: Optional[Dict] = None,
    short_description="",
    dump_model=False,
    is_tflite=False,
):
    results = {
        "model_configuration": model_config.to_dict(),
        "model_name": model_name.name,
        "dataset_name": dataset_name.name,
        "dataset_path": dataset_path,
        "short_description": short_description,
    }

    def add_value(name, value):
        if value is not None:
            results[name] = value

    add_value("dataset_params", dataset_params)
    add_value("train_history", train_history)
    
    add_value("cm_val", cm_val if cm_val is None else cm_val.tolist())
    add_value("cls_to_acc_val", cls_to_acc_val)
    add_value("snr_to_acc_val", snr_to_acc_val)

    add_value("cm_test", cm_test if cm_test is None else cm_test.tolist())
    add_value("cls_to_acc_test", cls_to_acc_test)
    add_value("snr_to_acc_test", snr_to_acc_test)

    if dump_model:
        if is_tflite:
            os.makedirs(where, exist_ok=True)
            with open(where + "/model_tflite", 'wb') as model_file:
                model_file.write(model)
        else:
            model_weights_path = os.path.abspath(os.path.join(where, "model_weights"))
            model.save_weights(model_weights_path)
            results["path_to_weights"] = model_weights_path
            results["n_parameters"] = get_model_parameters(model)
    results["is_tflite"] = is_tflite

    _dump_results(os.path.join(where, "results.json"), results)


def load_results(where: str, load_model=False):
    with open(os.path.join(where, "results.json"), "r") as results_file:
        results = json.load(results_file)
    results["model_configuration"] = load_model_configuration(results["model_configuration"])
    if load_model:
        if results["is_tflite"]:
            with open(where + "/model_tflite", 'rb') as model_file:
                results["model"] = model_file.read()
        else:
            assert "path_to_weights" in results, "No path to model weights found!"
            model = make_sigmod_model(ModelName[results["model_name"]], results["model_configuration"])
            model.load_weights(results["path_to_weights"])
            results["model"] = model
    return results
