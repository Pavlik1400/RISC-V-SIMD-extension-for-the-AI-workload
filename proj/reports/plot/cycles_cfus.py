import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from common import SaveMethod, get_save_func, load_report_or
from typing import Dict
import os

# text_additional_height = 5e7
# text_additional_height = 1e6
# text_additional_height = 3e5
text_additional_height = 0.3
down_scale = 1e6


def main(
    cfus_filepath: str,
    v5x_filepath: str,
    v6x_filepath: str,
    save_method: SaveMethod,
    output_dir: str,
):
    # 0. Data gathering
    v5x_report: Dict = load_report_or(v5x_filepath, {})
    v6x_report: Dict = load_report_or(v6x_filepath, {})
    cfus_report: Dict = load_report_or(cfus_filepath, {})

    meaned_v5_data = {}
    for k, v5_cycles in v5x_report.items():
        meaned_v5_data[k] = np.mean(v5_cycles)

    meaned_v6x_data = {}
    for k in v5x_report:
        meaned_v6x_data[k] = np.mean(v6x_report[k])

    meaned_cfus_data = {}
    for k in cfus_report:
        meaned_cfus_data[k] = np.mean(cfus_report[k])

    v5x_versions = list(meaned_v5_data.keys())
    v5_cycles = np.array(list(meaned_v5_data.values())) / down_scale

    v6x_versions = list(meaned_v6x_data.keys())
    v6x_cycles = np.array(list(meaned_v6x_data.values())) / down_scale

    cfus_versions = list(meaned_cfus_data.keys())
    cfus_cycles = list(meaned_cfus_data.values())

    assert v5x_versions == v6x_versions

    # fig_vx, (ax_cfus, ax_vx) = plt.subplots(2)
    fig_vx, ax_vx = plt.subplots()

    # 1. Bar plot of cfus
    # ax_cfus.set_ylim(min(cfus_cycles) - 1e6, max(cfus_cycles) + 1e6)
    # cfus_bars = ax_cfus.bar(cfus_versions, cfus_cycles, color="#3891A6", width=0.8)

    # ax_cfus.spines["top"].set_visible(False)
    # ax_cfus.spines["right"].set_visible(False)
    # ax_cfus.spines["left"].set_visible(False)
    # ax_cfus.spines["bottom"].set_color("#DDDDDD")
    # ax_cfus.tick_params(bottom=False, left=False)

    # ax_cfus.set_axisbelow(True)
    # ax_cfus.yaxis.grid(True, color="#DDDDDD")
    # ax_cfus.xaxis.grid(False)
    # # ax_cfus.set_xlabel("CFU versions")
    # ax_cfus.set_ylabel("Cycles (M)")
    # ax_cfus.legend()

    # for bars in [cfus_bars]:
    #     bar_color = bars[0].get_facecolor()
    #     for bar in bars:
    #         ax_vx.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             bar.get_height() + text_additional_height,
    #             # "%.1fM" % (bar.get_height() / 1e6),
    #             "%.1f" % (bar.get_height() / 1e6),
    #             horizontalalignment="center",
    #             color=bar_color,
    #             weight="bold",
    #         )

    # 2. Bar plot of V5 vs V6
    # creating the bar plot -- V5 under V6
    # width = 0.8
    # hardware_bars = plt.bar(
    #     v6x_versions, v6x_cycles, color="#ebbd34", width=width, label="CFU_V6"
    # )

    # simulation_bars = plt.bar(
    #     v5x_versions, v5_cycles, color="#63A8D3", width=width, label="CFU_V5"
    # )

    # V5 near V6
    width = 0.3
    x = np.arange(len(v6x_cycles))
    hardware_bars = ax_vx.bar(
        # x + width + .1, v6x_cycles, color="#ebbd34", width=width, label="CFU_V6"
        # x + width + .1, v6x_cycles, color="#DB5461", width=width, label="CFU_V6"
        # x + width + 0.1,
        x,
        v6x_cycles,
        color="#DB5461",
        width=width,
        label="Hardware",
    )

    simulation_bars = ax_vx.bar(
        # x, v5_cycles, color="#63A8D3", width=width, label="CFU_V5"
        # x, v5_cycles, color="#329F5B", width=width, label="CFU_V5"
        # x,
        x + width + 0.1,
        v5_cycles,
        color="#329F5B",
        width=width,
        label="Simulation",
    )
    # ax_vx.set_xticklabels(x + width / 2 + .05, v5x_versions)
    ax_vx.set_xticks(x + width / 2 + 0.05, v5x_versions)

    # plt.xticks(ind + width / 2, ('Xtick1', 'Xtick3', 'Xtick3'))

    ax_vx.spines["top"].set_visible(False)
    ax_vx.spines["right"].set_visible(False)
    ax_vx.spines["left"].set_visible(False)
    ax_vx.spines["bottom"].set_color("#DDDDDD")

    ax_vx.tick_params(bottom=False, left=False)

    ax_vx.set_axisbelow(True)
    ax_vx.yaxis.grid(True, color="#DDDDDD")
    ax_vx.xaxis.grid(False)

    # ax_vx.set_xlabel("CFU versions")
    ax_vx.set_xlabel("CFU version")
    # plt.xlabel("CFU versions (number after ''X'' - number of multiply-accumulate operations per cycle)")
    ax_vx.set_ylabel("Cycles$\cdot 10^6$")
    # plt.title("CNN1:generated_0-30 inference acceleration (clock cycles)")
    # plt.title("CNN1:generated\\_0-30 inference acceleration (clock cycles)")
    ax_vx.legend()

    # Code below for V5 under V6
    # bar_color = simulation_bars[0].get_facecolor()
    # for bar in simulation_bars:
    #     ax_vx.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height() + text_additional_height,
    #         "%.1fM" % (bar.get_height() / 1e6),
    #         horizontalalignment="center",
    #         color=bar_color,
    #         weight="bold",
    #     )

    for bars in [simulation_bars, hardware_bars]:
        bar_color = bars[0].get_facecolor()
        for bar in bars:
            ax_vx.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + text_additional_height,
                # "%.1fM" % (bar.get_height() / 1e6),
                # "%.1f" % (bar.get_height() / 1e6),
                "%.1f" % (bar.get_height()),
                horizontalalignment="center",
                color=bar_color,
                weight="bold",
            )

    x_left, x_right = ax_vx.get_xlim()
    y_low, y_high = ax_vx.get_ylim()
    ax_vx.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.5)

    os.makedirs(output_dir, exist_ok=True)
    get_save_func(save_method)(output_dir + "/cycle_bars_cfus")
    # matplotlib.use("pgf")
    # matplotlib.rcParams.update(
    #     {
    #         "pgf.texsystem": "pdflatex",
    #         "font.family": "serif",
    #         "font.size": 7,
    #         "text.usetex": True,
    #         "pgf.rcfonts": False,
    #     }
    # )
    # plt.savefig(output_dir + "/cycle_bars_computation_software.pgf")
    # plt.savefig(output_dir + "/cycle_bars_computation_software.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfus", required=True)
    parser.add_argument("--v5x", required=True)
    parser.add_argument("--v6x", required=True)
    parser.add_argument("--output_dir", "-o", default=".")
    parser.add_argument(
        "--save_method", "-m", default="show", help="Supported: [show, png, pgf, pdf]"
    )
    args = parser.parse_args()

    save_method = SaveMethod.from_str(args.save_method)
    main(args.cfus, args.v5x, args.v6x, save_method, args.output_dir)
