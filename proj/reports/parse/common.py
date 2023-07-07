import json
from typing import Any
import os


def add_filename_to_path_if_needed(path_or_dir: str, filename: str):
    if os.path.isdir(path_or_dir):
        path_or_dir += "/" + filename
    return path_or_dir

def save_update_report(report_filepath: str, key: str, value: Any):
    report = {}
    try:
        with open(report_filepath, "r") as res_file:
            report = json.load(res_file)
    except Exception:
        print("[warn] Looks like report doesn't exist or is not json file")

    with open(report_filepath, "w") as res_file:
        report.update({key: value})
        print(f"Add experiment '{key}' to result file '{value}'")
        json.dump(report, res_file, indent=4)