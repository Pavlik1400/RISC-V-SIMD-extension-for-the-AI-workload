import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
import sys
from utils import get_modulations, save_plot
from argparse import ArgumentParser
import seaborn as sns

# TODO: ugly
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from evaluation.results_serialization import load_results


# Don't change, use --save_png
# EXT = "pgf"
EXT = "pdf"
scale = 100.0

experiment = "experiments/cnn_1d_v013_small_radio_ml16b_quant_results/"
# label = "Quantized CNN Confusion matrix"

def confusion_values_to_acc(confusion_matrix: np.ndarray):
    confusion_matrix = confusion_matrix.astype(np.float64)
    for r in range(confusion_matrix.shape[0]):
        row = confusion_matrix[r]
        confusion_matrix[r] = row / np.sum(row) * scale
    return confusion_matrix

def confusion_matrix_plot(save_path: str):
    fig, ax = plt.subplots()

    results = load_results(experiment)
    
    # modulations = get_modulations(results)
    modulations = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    # print(modulations)

    confusion_matrix = np.array(results["cm_test"])
    confusion_matrix = confusion_values_to_acc(confusion_matrix)
    # print(confusion_matrix)

    # Create a heatmap using seaborn
    # ax[i // 2, i % 2].set_title(labels[i])

    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=modulations).plot(
        include_values=False,
        cmap="Blues",
        ax=ax,
        colorbar=True,
        # values_format="0.0f",
        xticks_rotation="vertical",
    )
    
    # sns.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax, vmax=scale, vmin=0);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # Add labels to the x-axis and y-axis
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    save_filepath = (
        os.path.join(save_path, f"confusion_matrix.{EXT}") if save_path is not None else None
    )
    save_plot(save_filepath)
    # plt.show()


# def confusion_matrices_n_plots(save_path: str):
#     for i, experiment in enumerate(experiments):
#         results = load_results(experiment)
#         modulations = get_modulations(results)

#         confusion_matrix = np.array(results["cm_test"])

#         # fig = plt.figure(figsize=(5, 4))
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#         # ax.set_title(labels_latex[i])

#         sns.heatmap(
#             confusion_matrix,
#             # annot=False,
#             annot=True,
#             cmap="Blues",
#             xticklabels=modulations,
#             yticklabels=modulations,
#             fmt="g",
#         )

#         plt.xticks(rotation=45)

#         # Add labels to the x-axis and y-axis
#         # ax.set_xlabel("Predicted")
#         # ax.set_ylabel("True")
#         save_filepath = (
#             os.path.join(save_path, f"confusion_matrix_{labels[i]}.{EXT}")
#             if save_path is not None
#             else None
#         )
#         save_plot(save_filepath)


# plt.show()
# plt.savefig()

if __name__ == "__main__":
    parser = ArgumentParser("Make model plots")
    parser.add_argument("--save_path", "-s", required=False, default=None)
    parser.add_argument("--save_png", action="store_const", const=True, default=False)
    args = parser.parse_args()

    save_path, save_png = args.save_path, args.save_png

    if save_path is not None:
        if not save_png:
            # matplotlib.use("pgf")
            matplotlib.rcParams.update(
                {
                    "pgf.texsystem": "pdflatex",
                    "font.family": "serif",
                    "font.size": 14,
                }
            )
        else:
            EXT = "png"
        os.makedirs(save_path, exist_ok=True)

    confusion_matrix_plot(save_path)
