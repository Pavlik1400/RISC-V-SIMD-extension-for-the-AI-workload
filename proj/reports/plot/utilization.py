from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from common import SaveMethod, get_save_func, load_report_or
from argparse import ArgumentParser
import os


def main(report_filepath: str, save_method: SaveMethod, output_dir: str):
    design_to_utilization: Dict = load_report_or(report_filepath, {})
    # Extract row and column labels
    row_labels = list(design_to_utilization.keys())
    column_labels = list(design_to_utilization[row_labels[0]].keys())

    # Extract values as a numpy array and transpose
    values = np.array(
        [[design_to_utilization[row][col] for col in column_labels] for row in row_labels]
    )
    values = values.T  # Transpose the values

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the heatmap
    # im = ax.imshow(values, cmap='YlOrRd')
    im = ax.imshow(values, cmap="Blues")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set the ticks and labels for x-axis and y-axis
    ax.set_xticks(np.arange(len(row_labels)))
    ax.set_yticks(np.arange(len(column_labels)))
    ax.set_xticklabels(row_labels)
    ax.set_yticklabels(column_labels)

    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(column_labels)):
        for j in range(len(row_labels)):
            color = "black"
            value = values[i, j]
            if value > 50:
                color = "white"
            text = ax.text(j, i, values[i, j], ha="center", va="center", color=color)

    # Set title and labels
    # ax.set_title("Design Utilization Heatmap (Transposed)")
    ax.set_xlabel("CFU version")
    ax.set_ylabel("Resource")

    # Display the plot
    os.makedirs(output_dir, exist_ok=True)
    get_save_func(save_method)(output_dir + "/resources_utilization")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--report", "-r", required=True)
    parser.add_argument("--output_dir", "-o", default=".")
    parser.add_argument("--save_method", "-m", default="show", help="Supported: [show, png, pgf]")
    args = parser.parse_args()

    save_method = SaveMethod.from_str(args.save_method)
    main(args.report, save_method, args.output_dir)
