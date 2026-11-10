"""

Physical tests for the Cloudy pipeline.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

    from ..utils import CloudyConfig, Pop3Config

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import polars as pl

from ..utils import instantiate_console, load_config, parse_args
from .data_storage import get_output_filename
from .ray_extraction import create_cloudy_tasks
from .run_cloudy_models import get_star

PANEL_LABELS: list[tuple[str, str]] = [
    ("temperature", r"$T$ [K]"),
    ("h_density", r"$n_\mathrm{H}$ [cm$^{-3}$]"),
    ("metallicity", r"$Z$ []"),
]
CLOUDY_PANEL_LABELS: list[tuple[str, str]] = [
    ("Te", r"$T$ [K]"),
    ("eden", r"$n_\mathrm{e}$ [cm$^{-3}$]"),
    ("hden", r"$n_\mathrm{H}$ [cm$^{-3}$]"),
]
X_LIM = 5e17  # radius [cm] where xlim starts (Cloudy interpolates a large range)

MNRAS_STYLE = Path.home() / "mnras.mplstyle"
plt.style.use(MNRAS_STYLE)


def plot_rays(config: Pop3Config, console: Console) -> None:
    """
    Plots the ray objects from a particular run (duplicates the block in the pipeline).
    """
    ledger = pl.read_parquet(config.stars.ledger_path)
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    star_pos, _, snapshot = get_star(ledger=ledger, config=config.cloudy, console=console)
    snap_path: Path = config.cloudy.snap_dir / snapshot / snapshot
    h5_path = get_output_filename(config=config)
    output_path = config.cloudy.plot_dir / f"{h5_path.stem}_rays.png"

    rays = list(
        create_cloudy_tasks(  # turn iterator into a held list
            snap_path=snap_path,
            star_pos_unitary=star_pos,
            ray_length_kpc=config.cloudy.ray_length_kpc,
            homogeneous=False,
            n_side=config.cloudy.n_side,
        )
    )

    for run_id, ray in enumerate(rays):
        for axis, (field_name, _) in zip(axes, PANEL_LABELS, strict=True):
            axis.plot(ray.depth, getattr(ray, field_name), label=f"{run_id!s}", lw=0.8, alpha=0.7)

    for axis, (_, axis_label) in zip(axes, PANEL_LABELS, strict=True):
        axis.set_yscale("log")
        axis.set_ylabel(axis_label)
        axis.set_xlim(X_LIM)

    fig.suptitle(
        f"YT Rays on {config.cloudy.star} at {config.cloudy.star_age_myr}Myr ({config.cloudy.ray_length_kpc}kpc)"
    )
    axes[-1].set_xscale("log")  # sharex is on
    axes[-1].set_xlabel(r"Depth [cm]")
    axes[0].legend(title="Run IDs", ncols=3, fontsize="x-small")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    console.log(f"Saved YT ray plot to {output_path}")


def plot_cloudy_overview(h5_path: Path, console: Console, config: CloudyConfig) -> None:
    """
    Plots Cloudy's overview profiles for every run with overview data in a run HDF5 file.
    """
    fig, axes = plt.subplots(nrows=len(CLOUDY_PANEL_LABELS), ncols=1, sharex=True)

    # just load this all in for now
    with h5py.File(h5_path, "r") as f:
        overview_group = f["cloudy_data/ovr"]
        starts = overview_group["starts"][:]
        lengths = overview_group["lengths"][:]
        depth = overview_group["depth"][:]
        columns = {name: overview_group[name][:] for name, _ in CLOUDY_PANEL_LABELS}

        # for reconstructing paths
        metadata = f["metadata"]
        star = metadata.attrs["star"]
        age_myr = metadata.attrs["star_age_myr"]

    # some pathlib wizardry
    h5_path = get_output_filename(config=config)
    output_path = config.plot_dir / f"{h5_path.stem}_cloudy.png"

    for run_id, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        if length == 0:  # failed ray
            continue
        run_slice = slice(start, start + length)
        for axis, (name, _) in zip(axes, CLOUDY_PANEL_LABELS, strict=True):
            values = columns[name][run_slice]
            axis.plot(depth[run_slice], values, label=f"{run_id}", lw=0.8, alpha=0.6)

    for axis, (_, axis_label) in zip(axes, CLOUDY_PANEL_LABELS, strict=True):
        axis.set_yscale("log")
        axis.set_ylabel(axis_label)
        axis.set_xlim(X_LIM)

    axes[-1].set_xscale("log")
    axes[-1].set_xlabel(r"Depth [cm]")
    axes[0].legend(title="Run IDs", ncols=3, fontsize="x-small")
    fig.suptitle(f"Cloudy Solution: {star} at {age_myr}Myr")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    console.log(f"Saved Cloudy .ovr file plot to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    console = instantiate_console()
    config = load_config(args=args)

    if args.force:
        plot_rays(
            config=config,
            console=console,
        )

    h5_path = get_output_filename(config=config.cloudy)
    plot_cloudy_overview(
        h5_path=h5_path,
        console=console,
        config=config.cloudy,
    )
