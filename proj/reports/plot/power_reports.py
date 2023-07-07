import os
from typing import Dict
import matplotlib.pyplot as plt
from common import SaveMethod, get_save_func, load_report_or
from argparse import ArgumentParser


def main(report_filepath: str, save_method: SaveMethod, output_dir: str):
    design_to_power: Dict = load_report_or(report_filepath, {})
    versions = list(design_to_power.keys())
    powers = list(design_to_power.values())

    fig, ax = plt.subplots()

    # creating the bar plot
    width = 0.8
    bars = plt.bar(
        versions, powers, color="#ebbd34", width=width, label="Hardware"
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

    ax.set_ylim(0.4, 0.6)

    plt.xlabel("CFU versions")
    # plt.xlabel("CFU versions (number after ''X'' - number of multiply-accumulate operations per cycle)")
    plt.ylabel("Power consumption (Watt)")
    # plt.title("CNN1:generated_0-30 inference acceleration (clock cycles)")
    # plt.title("CNN1:generated\\_0-30 inference acceleration (clock cycles)")
    # plt.legend()

    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1e6,
            "%.1fM" % (bar.get_height() / 1e6),
            horizontalalignment="center",
            color=bar_color,
            weight="bold",
        )

    # Display the plot
    os.makedirs(output_dir, exist_ok=True)
    get_save_func(save_method)(output_dir + "/power_consumption")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--report", "-r", required=True)
    parser.add_argument("--output_dir", "-o", default=".")
    parser.add_argument("--save_method", "-m", default="show", help="Supported: [show, png, pgf]")
    args = parser.parse_args()

    save_method = SaveMethod.from_str(args.save_method)
    main(args.report, save_method, args.output_dir)
