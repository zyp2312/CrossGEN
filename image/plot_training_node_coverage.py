"""Draw the PEMS03 coverage results as a vector PDF.

Run with Python and Matplotlib. Values are from the workbook
ST-FiT_不同训练节点比例_Figure3数值.xlsx, cells D1:I1 and D5:I6 of
不同训练节点比例结果. The published ST-FiT values were read from Fig. 3(a).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COVERAGE = [5, 10, 25, 50, 75, 100]
CROSSGEN_MAE = [20.51, 18.027, 17.3212, 17.5121, 17.2739, 17.38]
PUBLISHED_STFIT_MAE = [20.69, 18.53, 17.44, 17.63, 17.71, 17.84]


def main():
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })
    figure, axes = plt.subplots(figsize=(2.31, 1.65), layout="constrained")
    axes.plot(
        COVERAGE, CROSSGEN_MAE,
        label="CrossGEN", color="#005F92", linestyle="-", linewidth=1.1,
        marker="o", markersize=3.5, markeredgewidth=0.6,
    )
    axes.plot(
        COVERAGE, PUBLISHED_STFIT_MAE,
        label="ST-FiT", color="#A2401E", linestyle=(0, (4, 2)),
        linewidth=1.1, marker="s", markersize=3.4,
        markerfacecolor="white", markeredgewidth=0.7,
    )

    axes.set(
        xlabel="Target training-node coverage (%)",
        ylabel="MAE",
        xlim=(1, 104),
        ylim=(16.9, 21.05),
        xticks=COVERAGE,
        yticks=[17, 18, 19, 20, 21],
    )
    axes.get_xticklabels()[0].set_horizontalalignment("right")
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_axisbelow(True)
    axes.grid(axis="y", color="#DDDDDD", linewidth=0.45)
    axes.legend(loc="upper right", frameon=False, handlelength=2.2)

    output = Path(__file__).with_name("training-node-coverage.pdf")
    figure.savefig(output)
    plt.close(figure)


if __name__ == "__main__":
    main()
