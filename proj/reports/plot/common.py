import matplotlib
import matplotlib.pyplot as plt
from enum import Enum
import json
from typing import Dict


class SaveMethod(Enum):
    Show = 0
    Png = 1
    Pgf = 2
    Pdf = 3

    def from_str(s: str):
        return {
            "show": SaveMethod.Show,
            "png": SaveMethod.Png,
            "pgf": SaveMethod.Pgf,
            "pdf": SaveMethod.Pdf,
        }[s]


def get_save_func(save_method: SaveMethod):
    if save_method == SaveMethod.Show:
        return lambda _: plt.show()
    if save_method == SaveMethod.Png:
        return lambda name: plt.savefig(name + ".png")
    if save_method == SaveMethod.Pgf:
        matplotlib.use("pgf")
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "font.size": 7,
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
        return lambda name: plt.savefig(name + ".pgf")
    if save_method == SaveMethod.Pdf:
        return lambda name: plt.savefig(name + ".pdf", format="pdf", bbox_inches="tight")
    raise ValueError("Unknown save method")


def load_report_or(path: str, default_value: Dict) -> Dict:
    try:
        with open(path, "r") as report_file:
            report = json.load(report_file)
        return report
    except Exception:
        return default_value
