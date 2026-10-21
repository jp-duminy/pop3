"""

Creates profile and phase plots for stars across their snapshots.

 NOTE: the type annotations are slightly ugly because of how you access yt data. 

"""

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from yt.data_objects.static_output import Dataset
    from yt.data_objects.profiles import ProfileND
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap

import gc
import argparse
from pathlib import Path
from concurrent.futures import as_completed

import yt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpi4py.futures import MPIPoolExecutor
from rich.progress import Progress

from ..utils import instantiate_logger, parse_args, load_config

MNRAS_STYLE = Path.home() / "mnras.mplstyle"
plt.style.use(MNRAS_STYLE)

# NOTE: quantities change to grid when stored
RADIAL_FIELDS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("grid", "radius"), ("grid", "density")),
    (("grid", "radius"), ("grid", "temperature")),
    (("grid", "radius"), ("grid", "metallicity3")),
    (("grid", "radius"), ("grid", "H_p0_number_density")),
    (("grid", "radius"), ("grid", "H_p1_number_density")),
]
PHASE_FIELDS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("grid", "density"), ("grid", "temperature")),
    (("grid", "density"), ("grid", "metallicity3")),
    (("grid", "temperature"), ("grid", "metallicity3")),
]
LABELS: dict[str, str] = {
    "density": r"$\rho\ [\mathrm{g\,cm^{-3}}]$",
    "temperature": r"$T\ [\mathrm{K}]$",
    "metallicity3": r"$Z\ [Z_{\odot}]$",
    "H_p0_number_density": r"$n_{\rm HI}\ [\mathrm{cm^{-3}}]$",
    "H_p1_number_density": r"$n_{\rm HII}\ [\mathrm{cm^{-3}}]$",
    "radius": r"$R\ [\mathrm{cm}]$"
}


class SphereTask(NamedTuple):
    """
    Contains:

    - label
    - snapshot
    - sphere_path
    """
    label: str
    snapshot: str
    sphere_path: Path


def process_sphere(task: SphereTask, outdir: Path) -> str:
    """
    Takes a snapshot sphere and produces profiles and plots.
    """
    yt.set_log_level("error")
    sp_ds = yt.load(task.sphere_path)

    # a lot of tedious directory making
    task_outdir = outdir / task.label / task.snapshot
    task_outdir.mkdir(parents=True, exist_ok=True)

    # both profiles can reuse the same function
    with plt.rc_context({"figure.figsize": (7.0, 7.4)}):
    
        radial_profiles = build_profiles(sp_ds, field_pairs=RADIAL_FIELDS)
        plot_radial_figure(
            profiles=radial_profiles, 
            title=f"Radial Profiles for {task.label} at z = {sp_ds.current_redshift:.2f}",
            output_path=task_outdir / f"radial_plot.png", 
        )
    del radial_profiles  # free up memory
    gc.collect()

    with plt.rc_context({"figure.figsize": (7.0, 2.6)}):
        phase_profiles = build_profiles(sp_ds,field_pairs=PHASE_FIELDS)
        plot_phase_figure(
            profiles=phase_profiles, 
            title=f"Phase Profiles for {task.label} at z = {sp_ds.current_redshift:.2f}",
            output_path=task_outdir / f"phase_plot.png", 
        )
    del phase_profiles
    gc.collect()

    return f"{task.label}_{task.snapshot}"


def build_plot_tasks(spheres_dir: Path, labels: list[str]) -> list[SphereTask]:
    """
    Builds the plot tasks via pure globbing over expected names rather than the ledger.
    """
    tasks: list[SphereTask] = []

    for label in labels:
        for sphere_path in sorted((spheres_dir / label).glob(f"{label}_*.h5")):
            # is in form POP3-***_DDxxxx.h5
            snapshot = sphere_path.stem.removeprefix(f"{label}_")  
            tasks.append(SphereTask(label, snapshot, sphere_path))

    return tasks


def build_profiles(
    sp_ds: Dataset, 
    field_pairs: list[tuple[tuple[str, str], tuple[str, str]]], 
) -> dict[str, ProfileND]:
    """
    Constructs radial profiles for requested fields, weighted by cell mass as per Britton. 
    Currently aimed at the following fields:

    - density
    - metallicity
    - temperature
    - HII density
    - HI density
    """
    profiles: dict[str, ProfileND] = {}

    sphere_data = sp_ds.all_data()
    sphere_data.set_field_parameter("center", sp_ds.parameters["center"].to("unitary"))

    for pair in field_pairs:

        profile = sphere_data.profile(  # default 64 bins
            bin_fields=list(pair),  # convert the pair back to a list (2n for 2D profile)
            fields=[("grid", "cell_mass")], 
            weight_field=None,
        )
        # unpack the pair and key by both fields to get the name
        # otherwise phase diagrams overwrite each other
        x_field, y_field = pair
        name = f"{x_field[1]}_{y_field[1]}"

        profiles[name] = profile

    return profiles


def plot_phase_figure(
    profiles: dict[str, ProfileND], 
    title: str, 
    output_path: Path, 
    dims: tuple[int, int] = (1, 3),
) -> None:
    """
    Creates a phase profile figure for an input dict of profiles. 
    """
    if len(profiles.keys()) != dims[0] * dims[1]:
        raise ValueError(f"Passed dims ({dims[0] * dims[1]}) disagree with the number of profiles ({len(profiles.keys())}) in the dict.")

    nrows, ncols = dims

    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols,
        sharex=False,
        squeeze=False,
    )

    mass_extrema = []

    for profile in profiles.values():
        mass_vals = profile["grid", "cell_mass"].to("Msun")
        mass_extrema.append(np.min(mass_vals[mass_vals > 0]))
        mass_extrema.append(np.max(mass_vals))

    mass_floor = np.min(mass_extrema)
    mass_ceiling = np.max(mass_extrema)
    mass_norm = LogNorm(vmin=mass_floor, vmax=mass_ceiling)  # avoid zero mass biasing plot

    for axis, profile in zip(axes.flat, profiles.values()):

        mass_values = profile["grid", "cell_mass"].to("Msun").T  # pcolormesh expects (y, x)

        # similar to yt complex plots example but with Britton's code
        mesh = axis.pcolormesh(profile.x, profile.y, mass_values, norm=mass_norm, cmap="cmyt.dusk")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(f"{LABELS[profile.x_field[1]]}")
        axis.set_ylabel(f"{LABELS[profile.y_field[1]]}")

    fig.colorbar(
        mesh,  # this technically works
        ax=axes, 
        label=r"$M_{\rm gas}\ [M_\odot]$", 
        pad=0.01, 
        aspect=10,
    )

    fig.suptitle(title)  # shared title for all figures
    fig.savefig(output_path)
    plt.close(fig)


def plot_radial_figure(
    profiles: dict[str, ProfileND], 
    title: str, 
    output_path: Path, 
    colour: str = "C0",  # just pick the first colour
    dims: tuple[int, int] = (5, 1),
) -> None:
    """
    Creates a radial profile figure for an input dict of profiles. 
    """
    if len(profiles.keys()) != dims[0] * dims[1]:
        raise ValueError(f"Passed dims ({dims[0] * dims[1]}) disagree with the number of profiles ({len(profiles.keys())}) in the dict.")

    nrows, ncols = dims

    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols,
        sharex=True,
        squeeze=False,
    )

    for axis, profile in zip(axes.flat, profiles.values()):

        plot_profile_distribution(
            axis=axis,
            profile=profile,
            colour=colour,
        )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(f"{LABELS[profile.x_field[1]]}")
        axis.set_ylabel(f"{LABELS[profile.y_field[1]]}")
        axis.set_xlim(np.min(profile.x.d), np.max(profile.x.d))
        axis.label_outer()

    fig.suptitle(title)  # shared title for all figures
    fig.savefig(output_path)
    plt.close(fig)


def plot_profile_distribution(
    axis: Axes,
    profile: ProfileND,
    colour: Colormap,
    step: float = 0.01,
    alpha_scale: float = 1.0,
) -> None:
    """
    Plots a profile distribution with quantile bands (via fill_between, from alpha_scale param).
    """
    x_data = profile.x.d
    y_data = profile.y.d
    z_data = profile["grid", "cell_mass"].d

    populated_columns = profile.used.any(axis=1)
    x_data = x_data[populated_columns]
    z_data = z_data[populated_columns, :]

    z_cdf = np.cumsum(z_data, axis=1) / np.sum(z_data, axis=1, keepdims=True)  # cumulative distribution function in z
    z_cdf[:, -1] = 1.0  # if you don't do this roundoff variance can make it 0.999 and the outermost bin gets nuked
    y_med = y_data[np.argmax(z_cdf >= 0.5, axis=1)]  # median at 0.5, argmax finds first bin where this is exceeded

    axis.plot(
        x_data[~np.isnan(y_med)], 
        y_med[~np.isnan(y_med)],
        alpha=0.9,
        color=colour,
    )

    n_bands = int(np.round(0.5 / step))  # np.round returns float even if rounded to integer

    offsets = np.linspace(step, 0.5, n_bands)
    # separate low and high points for shaded quantiles
    lower_levels = 0.5 - offsets
    upper_levels = 0.5 + offsets
    plot_alpha = 2.0 * alpha_scale * step

    lower_vals = _profile_quantiles(z_cdf, y_data, lower_levels)
    upper_vals = _profile_quantiles(z_cdf, y_data, upper_levels)

    band_mask = np.logical_and.accumulate(lower_vals != np.min(y_data), axis=1)

    for j in range(len(offsets)):
        mask_j = band_mask[:, j]
        lo, hi = lower_vals[mask_j, j], upper_vals[mask_j, j]

        axis.fill_between(
            x_data[mask_j], 
            lo, 
            hi,
            linewidth=0, 
            alpha=plot_alpha, 
            color=colour,
        )


def _profile_quantiles(
    cdf: np.ndarray,   
    y_bin_centres: np.ndarray,         
    quantile_levels: np.ndarray,      
) -> np.ndarray:
    """
    Gets the bin centres from y_bin_centres corresponding to where the specified cumulative fractions are crossed.
    """                     
    crossed = cdf[:, :, np.newaxis] >= quantile_levels[np.newaxis, np.newaxis, :]
    quantile_idx = np.argmax(crossed, axis=1)

    return y_bin_centres[quantile_idx]

if __name__ == "__main__":

    args = parse_args()
    config = load_config(args)
    console = instantiate_logger()
    yt.set_log_level("error")

    all_tasks = build_plot_tasks(
        spheres_dir=config.spheres.sp_dir,
        labels=config.stars.process_stars,
    )

    filtered_tasks: list[SphereTask] = []

    for sphere in all_tasks:
        outpath = config.plots.profile_plot_dir / sphere.label / sphere.snapshot
        if outpath.exists() and not args.force:
            continue
        filtered_tasks.append(SphereTask(
            label=sphere.label, snapshot=sphere.snapshot, sphere_path=sphere.sphere_path
        ))

    console.log(f"{len(filtered_tasks)} tasks to run; {len(all_tasks) - len(filtered_tasks)} skipped.")

    with Progress(console=console) as progress, MPIPoolExecutor() as executor:
        task_id = progress.add_task("Making Profile Plots", total=len(filtered_tasks))
        futures = [executor.submit(process_sphere, task, config.plots.profile_plot_dir) for task in filtered_tasks]

        for future in as_completed(futures):
            console.log(f"Done {future.result()}")
            progress.advance(task_id)