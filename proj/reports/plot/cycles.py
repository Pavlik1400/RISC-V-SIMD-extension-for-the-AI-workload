import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from common import SaveMethod, get_save_func, load_report_or
from typing import Dict
import os

# text_additional_height = 5e7
text_additional_height = 2e7


def main(
    simulation_filepath: str, hardware_filepath: str, save_method: SaveMethod, output_dir: str
):
    simulation_report: Dict = load_report_or(simulation_filepath, {})
    hardware_report: Dict = load_report_or(hardware_filepath, {})

    meaned_simulation_data = {}
    for k, simulations_cycles in simulation_report.items():
        meaned_simulation_data[k] = np.mean(simulations_cycles)

    meaned_hardware_data = {}
    for k in simulation_report:
        if k in hardware_report:
            meaned_hardware_data[k] = np.mean(hardware_report[k])
        else:
            meaned_hardware_data[k] = 0

    simulation_versions = list(meaned_simulation_data.keys())
    simulations_cycles = list(meaned_simulation_data.values())

    hardware_versions = list(meaned_hardware_data.keys())
    hardware_cycles = list(meaned_hardware_data.values())

    fig, ax = plt.subplots()

    # creating the bar plot
    width = 0.8
    hardware_bars = plt.bar(
        hardware_versions, hardware_cycles, color="#ebbd34", width=width, label="Hardware"
    )

    simulation_bars = plt.bar(
        simulation_versions, simulations_cycles, color="#63A8D3", width=width, label="Simulation"
    )
    # ax.set_xticklabels(meaned_simulation_data.keys())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")

    ax.tick_params(bottom=False, left=False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD")
    ax.xaxis.grid(False)

    plt.xlabel("CFU versions")
    # plt.xlabel("CFU versions (number after ''X'' - number of multiply-accumulate operations per cycle)")
    plt.ylabel("Cycles (M)")
    # plt.title("CNN1:generated_0-30 inference acceleration (clock cycles)")
    # plt.title("CNN1:generated\\_0-30 inference acceleration (clock cycles)")
    plt.legend()

    bar_color = simulation_bars[0].get_facecolor()
    for bar in simulation_bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + text_additional_height,
            "%.1fM" % (bar.get_height() / 1e6),
            horizontalalignment="center",
            color=bar_color,
            weight="bold",
        )

    # bar_color = hardware_bars[0].get_facecolor()
    # for bar in hardware_bars:
    #     if bar.get_height() == 0:
    #         continue
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height() + text_additional_height,
    #         "%.1fM" % (bar.get_height() / 1e6),
    #         horizontalalignment="center",
    #         color=bar_color,
    #         weight="bold",
    #     )

    os.makedirs(output_dir, exist_ok=True)
    get_save_func(save_method)(output_dir + "/cycle_bars_computation_software")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simulation_report", "-soft", required=True)
    parser.add_argument("--hardware_report", "-hard", required=True)
    parser.add_argument("--output_dir", "-o", default=".")
    parser.add_argument("--save_method", "-m", default="show", help="Supported: [show, png, pgf]")
    args = parser.parse_args()

    save_method = SaveMethod.from_str(args.save_method)
    main(args.simulation_report, args.hardware_report, save_method, args.output_dir)
