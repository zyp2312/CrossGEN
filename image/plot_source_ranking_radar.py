"""Plot candidate-source MAE for each traffic-flow target under CrossGEN.

The five-seed mean MAEs come from 五种子_四位小数_20261115.xlsx
(column 四位均值). The plotted experiments match Table 2 in main.tex.
Only the CrossGEN variant is plotted; the radial coordinate is the actual MAE.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DIRECTORY = Path(__file__).resolve().parent
TARGETS = ["PEMS03", "PEMS04", "PEMS08", "CA03", "CA06"]
SOURCE_COLORS = {
    "PEMS03": "#b43d1f",
    "PEMS04": "#6844a9",
    "PEMS08": "#973465",
    "CA03": "#08649d",
    "CA06": "#00745c",
}
SOURCE_MARKERS = {
    "PEMS03": "o",
    "PEMS04": "s",
    "PEMS08": "D",
    "CA03": "^",
    "CA06": "v",
}

# Mean MAE across five seeds for CrossGEN, per target and candidate source.
MAE = {
    "PEMS03": {"CA06": 18.3548, "PEMS04": 18.0279, "PEMS08": 18.4344},
    "PEMS04": {"PEMS03": 24.8293, "PEMS08": 24.5500, "CA06": 24.5691},
    "PEMS08": {"PEMS03": 23.0026, "PEMS04": 23.0487, "CA06": 23.5522},
    "CA03": {"PEMS04": 21.3422, "CA06": 21.1475},
    "CA06": {"PEMS03": 16.3558, "PEMS04": 16.4928, "CA03": 16.0685},
}


def plot():
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "font.size": 8,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    })

    figure, axis = plt.subplots(
        figsize=(2.9, 2.55), subplot_kw={"projection": "polar"}
    )
    angles = np.linspace(0, 2 * np.pi, len(TARGETS), endpoint=False)

    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_xticks(angles)
    axis.set_xticklabels(TARGETS)
    axis.tick_params(axis="x", pad=3, labelsize=7.5)
    axis.set_ylim(15.5, 25.5)
    axis.set_yticks([16, 18, 20, 22, 24])
    axis.set_yticklabels(["16", "18", "20", "22", "24"], fontsize=7)
    axis.set_rlabel_position(25)
    axis.grid(color="#D8DDE2", linewidth=0.55)
    axis.spines["polar"].set_color("#AAB2B8")
    axis.spines["polar"].set_linewidth(0.6)

    for angle, target in zip(angles, TARGETS):
        for source, mae in MAE[target].items():
            axis.scatter(
                angle, mae, s=26, marker=SOURCE_MARKERS[source],
                color=SOURCE_COLORS[source], edgecolors="white", linewidths=0.65,
                zorder=3,
            )

    legend_handles = [
        Line2D(
            [], [], linestyle="none", marker=SOURCE_MARKERS[source],
            markersize=5.5, markerfacecolor=SOURCE_COLORS[source],
            markeredgecolor="white", label=source,
        )
        for source in SOURCE_COLORS
    ]
    figure.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=5, frameon=False, columnspacing=1.25, handletextpad=0.45, fontsize=8,
    )
    figure.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.22)
    figure.savefig(DIRECTORY / "source-ranking-radar.pdf")
    plt.close(figure)


if __name__ == "__main__":
    plot()
