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
import numpy as np
import polars as pl

from ..utils import instantiate_console, load_config, parse_args
from .commands import DlawTable, TlawTable
from .data_storage import get_output_filename
from .parsers import read_nomoto_table
from .ray_extraction import create_cloudy_tasks
from .rays_to_commands import build_cloudy_cmds
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


def plot_cloudy_residual(config: Pop3Config, console: Console) -> None:
    """
    Plots a residual for the yt ray against what Cloudy calculated.
    """
    ledger = pl.read_parquet(config.stars.ledger_path)
    star_pos, redshift, snapshot = get_star(ledger=ledger, config=config.cloudy, console=console)
    snap_path: Path = config.cloudy.snap_dir / snapshot / snapshot
    h5_path = get_output_filename(config=config.cloudy)
    output_path = config.cloudy.plot_dir / f"{h5_path.stem}_residual.png"
    yields = read_nomoto_table(nomoto_path=config.cloudy.datafile_dir / "pop3_metals.tsv", config=config)

    ray = next(  # take first ray
        create_cloudy_tasks(
            snap_path=snap_path,
            star_pos_unitary=star_pos,
            ray_length_kpc=config.cloudy.ray_length_kpc,
            homogeneous=False,
            n_side=config.cloudy.n_side,
        )
    )

    cmds = build_cloudy_cmds(
        ray=ray,
        yields=yields,
        redshift=redshift,
        config=config.cloudy,
    )

    # what Cloudy solved for
    with h5py.File(h5_path, "r") as f:
        overview_group = f["cloudy_data/ovr"]
        starts = overview_group["starts"][:]
        lengths = overview_group["lengths"][:]

        zero_slice = slice(starts[0], starts[0] + lengths[0])

        # all come  out linear
        temp = overview_group["Te"][zero_slice]
        hden = overview_group["hden"][zero_slice]
        cloudy_depth = overview_group["depth"][zero_slice]

    cloudy_depth = align_cloudy_depth(cloudy_depth=cloudy_depth)
    # what was fed in
    dlaw = next(cmd for cmd in cmds if isinstance(cmd, DlawTable))
    tlaw = next(cmd for cmd in cmds if isinstance(cmd, TlawTable))

    log_cloudy_depth = np.log10(cloudy_depth)

    # Cloudy interpolates in log-log; reproduce the same effect
    reproduced_t_interp = np.interp(
        log_cloudy_depth,
        tlaw.log_path_lengths,
        tlaw.log_temperatures,
    )
    tlaw_residual = reproduced_t_interp - np.log10(temp)

    reproduced_d_interp = np.interp(
        log_cloudy_depth,
        dlaw.log_path_lengths,
        dlaw.log_densities,
    )
    dlaw_residual = reproduced_d_interp - np.log10(hden)

    fig, (ax_temp, ax_dens) = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax_temp.plot(cloudy_depth, tlaw_residual)
    ax_dens.plot(cloudy_depth, dlaw_residual)
    for ax in (ax_temp, ax_dens):
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_ylim(-0.15, +0.15)

    ax_dens.set_xlabel("Depth [cm]")
    ax_dens.set_xscale("log")
    ax_dens.set_xlim(X_LIM)
    ax_temp.set_ylabel(r"$T$ [K]")
    ax_dens.set_ylabel(r"$n_\mathrm{H}$ [cm$^{-3}$]")
    annotation = f"Cloudy zones: {lengths[0]}\nRay cells: {ray.depth.size}"
    ax_temp.text(
        0.02,
        0.95,
        annotation,
        transform=ax_temp.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "grey", "alpha": 0.8},
    )
    fig.suptitle("Cloudy and YT Ray Tlaw and Dlaw Residuals")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    console.log(f"Saved residual plot to {output_path}")


def align_cloudy_depth(
    cloudy_depth: np.ndarray,
) -> np.ndarray:
    """
    Cloudy evaluates outwards from r0. The stated depth in the .ovr file is the midpoint, but the law is evaluated
    at the outer edge of each zone; this function realigns the depth array to what Cloudy evaluated.
    """
    # cloudy_depth is the midpoint -> 0.5 * (x_i + x_(i+1)) gives approximate right edge
    edges = 0.5 * (cloudy_depth[:-1] + cloudy_depth[1:])
    # recreate the edge of the final bin
    outer_edges = np.concatenate((edges, [2.0 * cloudy_depth[-1] - edges[-1]]))

    return outer_edges


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

    plot_cloudy_residual(
        config=config,
        console=console,
    )
