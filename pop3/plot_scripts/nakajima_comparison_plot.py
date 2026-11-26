"""

For comparing models to the Nakajima+2022 paper figures.

"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

import polars as pl
from matplotlib import pyplot as plt

from pop3 import OutputCatalogue

from ..utils import instantiate_console, load_config

HEII_LABEL = r"$HeII 1640 / H\beta$"
OIII_LABEL = r"$OIII 1665 / H\beta$"
CIII_LABEL = r"$CIII 1909 / H\beta$"

HEII_KEY = "he 2 1640.41A"
CIII_KEY = "blnd 1909.00A"

NAKAJIMA_COLOURS: dict[str, str] = {
    "PopIII_sal1-100": "tab:blue",
    "PopIII_sal1-500": "tab:green",
    "PopIII_sal50-500": "tab:orange",
}


def read_nakajima_table(nakajima_path: Path) -> pl.DataFrame:
    """
    Loads the .dat table and parses it into a dataframe.
    """
    labels = [
        "population",
        "gas density (log(nH))",
        "gas metallicity (absolute)",
        "ionisation (log(U))",
        "EW HII",
        "CIV 1549 (doublet)",
        "HeII 1640",
        "OIII 1665",
        "CIII 1909",
    ]
    df = pl.read_csv(nakajima_path, separator=" ", comment_prefix="#", new_columns=labels)

    return df


def parse_plot_args() -> argparse.Namespace:
    """
    Parses the catalogue path and optional output path.
    """
    parser = argparse.ArgumentParser(description="Quick comparison plot maker for Nakajima+2022")
    parser.add_argument("-cp", "--catalogue-path", type=Path, help="Path to the Cloudy HDF5 file.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Figure save path (leave blank for none)")

    return parser.parse_args()


def draw_nakajima(ax: Axes, nakajima: pl.DataFrame, x_column: str, y_column: str) -> None:
    """
    Draw the Nakajima+2022 models on `ax`.
    """
    for (population,), group in nakajima.partition_by("population", as_dict=True).items():
        ax.scatter(
            group[x_column],
            group[y_column],
            marker="s",
            s=10,
            color=NAKAJIMA_COLOURS[population],
            label=f"Nakajima+2022 {population}",
        )


def draw_duminy(ax: Axes, lines: pl.DataFrame, x_label: str, y_label: str) -> None:
    """
    Draw the Duminy+2027 models on `ax`.
    """
    ax.scatter(lines[x_label], lines[y_label], marker="^", s=10, label="Us")


def heii_against_ciii(catalogue_path: Path, nakajima: pl.DataFrame, output_path: Path | None = None) -> None:
    """
    Plot HeII against CIII line intensities relative to Hbeta.
    """
    cat = OutputCatalogue(catalogue_path)
    lines = cat.read_lines()
    fig, axes = plt.subplots()

    draw_nakajima(axes, nakajima, x_column="HeII 1640", y_column="CIII 1909")
    draw_duminy(axes, lines, HEII_KEY, CIII_KEY)
    axes.set(xscale="log", yscale="log", xlabel=HEII_LABEL, ylabel=CIII_LABEL, title="Nakajima+2022 Comparison Plot")
    axes.legend()

    if output_path is not None:
        fig.savefig(output_path)


if __name__ == "__main__":
    MNRAS_STYLE = Path.home() / "mnras.mplstyle"
    plt.style.use(MNRAS_STYLE)

    args = parse_plot_args()
    console = instantiate_console()
    config = load_config()

    nakajima = read_nakajima_table(nakajima_path=config.cloudy.misc.nakajima_path)
    heii_against_ciii(catalogue_path=args.catalogue_path, nakajima=nakajima, output_path=args.output)
    console.log(f"Saved Nakajima comparison plot to {args.output}")
