import os
from pprint import pprint
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from common import SaveMethod, get_save_func, load_report_or


def parse_times(times_filepath: str):
    # state = "looking_for_cycles"
    state = "looking_for_anchor"

    results = {}
    cur_cfu_name = None

    for line in open(times_filepath, "r"):
        line = line.strip()
        if line.startswith("// CFU V") and state == "looking_for_anchor":
            state = "looking_for_cycles"
            cur_cfu_name = line.strip("//").strip()
            results[cur_cfu_name] = {}
        if line.startswith("Mean (sec)") and state == "looking_for_cycles":
            state = "looking_for_anchor"
            results[cur_cfu_name] = float(line.split(" ")[-1])

    pprint(results)


inference = {
    # "ORIGINAL\\_NO\\_QUANT": 200.65990871191025,
    "ORIGINAL": 24.935958051681517,
    "SIMPLIFIED": 16.178125858306885,
    "V4.0": 3.1903757095336913,
    "V5.0": 2.938006329536438,
    "V5.1": 2.9603942394256593,
    "V6.1": 2.206445503234863,
    "v7.0": 2.2571829080581667,
    "V8.0": 2.333394765853882,
    "V8.0\nx1": 11.51913013458252,
    "V8.0\nx2": 6.25622456073761,
    "V8.0\nx4": 3.652141880989075,
    "V8.0\nx8": 2.333394765853882,
    "V8.0\nx16": 1.6747270345687866,
    "V8.0\nx24": 1.4663979291915894,
    "V8.0\nx32": 1.3512747764587403,
}


def main(report_filepath: str, save_method: SaveMethod, output_dir: str):
    # TODO: load report from report file
    
    versions = list(inference.keys())
    times = list(inference.values())

    fig, ax = plt.subplots()

    # creating the bar plot
    width = 0.8
    bars = plt.bar(
        versions, times, color="#ebbd34", width=width, label="Hardware"
    )

    # ax.set_xticklabels(meaned_simulation_data.keys())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")

    ax.tick_params(bottom=False, left=False)

    plt.xticks(rotation=30)
    ax.set_yticks(np.arange(0, inference["ORIGINAL"] + 0.5, 1.0))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD")
    ax.xaxis.grid(False)

    # ax.set_ylim(0.4, 0.6)

    plt.xlabel("Implementation versions")
    # plt.xlabel("CFU versions (number after ''X'' - number of multiply-accumulate operations per cycle)")
    plt.ylabel("Time (s)")
    # plt.title("CNN1:generated_0-30 inference acceleration (clock cycles)")
    # plt.title("CNN1:generated\\_0-30 inference acceleration (clock cycles)")
    # plt.legend()

    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            "%.2f" % (bar.get_height()),
            horizontalalignment="center",
            color=bar_color,
            weight="bold",
        )

    # Display the plot
    os.makedirs(output_dir, exist_ok=True)
    get_save_func(save_method)(output_dir + "/inference_time")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--report", "-r", required=False)
    parser.add_argument("--output_dir", "-o", default=".")
    parser.add_argument("--save_method", "-m", default="show", help="Supported: [show, png, pgf]")
    args = parser.parse_args()

    save_method = SaveMethod.from_str(args.save_method)
    main(args.report, save_method, args.output_dir)
