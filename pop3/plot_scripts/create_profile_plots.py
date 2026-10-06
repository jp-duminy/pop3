"""

Creates profile and phase plots for stars across their snapshots.

 NOTE: the type annotations are slightly ugly because of how you access yt data. 

"""

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from yt.data_objects.static_output import Dataset
    from yt.data_objects.profiles import ProfileND

import gc
import argparse
from pathlib import Path
from concurrent.futures import as_completed

import yt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpi4py.futures import MPIPoolExecutor
from rich.progress import Progress

from ..utils import common_parser, instantiate_logger

# NOTE: quantities change to grid when stored
RADIAL_FIELDS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("index", "radius"), ("grid", "density")),
    (("index", "radius"), ("grid", "temperature")),
    (("index", "radius"), ("grid", "metallicity3")),
    (("index", "radius"), ("grid", "H_p0_number_density")),
    (("index", "radius"), ("grid", "H_p1_number_density")),
]
PHASE_FIELDS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("grid", "density"), ("grid", "temperature")),
    (("grid", "density"), ("grid", "metallicity3")),
    (("grid", "temperature"), ("grid", "metallicity3")),
]

def parse_args() -> argparse.Namespace:
    """
    Parses the command-line arguments; returns the corresponding Namespace object.
    """
    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        prog="star-profiles",
        description="Routine for making diagnostic plots of stars across their lifetimes.",
        suggest_on_error=True,
    )
    parser.add_argument(
        "-sp",
        "--spheres",
        type=Path, 
        default=Path.home() / "mphys" / "data" / "spheres", 
        help="Path to sphere directory."
    )
    parser.add_argument(
        "-pd",
        "--profiledir",
        type=Path, 
        required=False,
        help="(Optional) path to profile directory for saving intermediates."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrites existing figures."
    )

    return parser.parse_args()


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


def process_sphere(task: SphereTask, outdir: Path, profile_root_dir: Path | None = None) -> str:
    """
    Takes a snapshot sphere and produces profiles and plots.
    """
    sp_ds = yt.load(task.sphere_path)

    # a lot of tedious directory making
    task_outdir = outdir / task.label / task.snapshot
    task_outdir.mkdir(parents=True, exist_ok=True)
    profile_dir = profile_root_dir / task.label / task.snapshot if profile_root_dir is not None else None
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)

    # both profiles can reuse the same function
    radial_profiles = build_profiles(sp_ds, field_pairs=RADIAL_FIELDS, save_dir=profile_dir)
    plot_profile_figure(
        profiles=radial_profiles, 
        title=f"Radial Profiles for {task.label} in {task.snapshot}",
        output_path=task_outdir / "radial_plot.png", 
        share_x=True, 
        dims=(5, 1),
    )
    del radial_profiles  # free up memory
    gc.collect()

    phase_profiles = build_profiles(sp_ds,field_pairs=PHASE_FIELDS, save_dir=profile_dir)
    plot_profile_figure(
        profiles=phase_profiles, 
        title=f"Phase Profiles for {task.label} in {task.snapshot}",
        output_path=task_outdir / "phase_plot.png", 
        share_x=False, 
        dims=(1, 3)
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
    save_dir: Path | None = None
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
    centre = sp_ds.center.to("unitary")
    sp_ds.data.set_field_parameter("center", centre)

    for pair in field_pairs:

        profile = sp_ds.data.profile(  # default 64 bins
            bin_fields=list(pair),  # convert the pair back to a list (2n for 2D profile)
            fields=[("grid", "cell_mass")], 
            weight_field=None,
        )
        # unpack the pair and key by both fields to get the name
        # otherwise phase diagrams overwrite each other
        x_field, y_field = pair
        name = f"{x_field[1]}_{y_field[1]}"

        profiles[name] = profile
        if save_dir is not None:
            profile.save_as_dataset(save_dir / f"{name}_profile.h5")

    return profiles


def plot_profile_figure(
    profiles: dict[str, ProfileND], 
    title: str, 
    output_path: Path, 
    share_x: bool,
    dims: tuple[int, int],
) -> None:
    """
    Creates a profile figure for an input dict of profiles.
    """
    if len(profiles.keys()) != dims[0] * dims[1]:
        raise ValueError(f"Passed dims ({dims[0] * dims[1]}) disagree with the number of profiles ({len(profiles.keys())}) in the dict.")

    nrows, ncols = dims

    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols,
        sharex=share_x,
        squeeze=False,
    )

    for axis, profile in zip(axes.flat, profiles.values()):

        mass_values = profile["grid", "cell_mass"].to("Msun").T  # pcolormesh expects (y, x)
        mass_norm = LogNorm(vmin=mass_values[mass_values > 0].min(), vmax=mass_values.max())  # avoid zero mass biasing plot

        # similar to yt complex plots example but with Britton's code
        mesh = axis.pcolormesh(profile.x, profile.y, mass_values, norm=mass_norm)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(f"{profile.x_field[1]} [{profile.x.units}]")
        axis.set_ylabel(f"{profile.y_field[1]} [{profile.y.units}]")
        fig.colorbar(mesh, ax=axis, label=r"$M_{\rm gas}\ [M_\odot]$")

    fig.suptitle(title)  # shared title for all figures
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":

    args = parse_args()
    console = instantiate_logger()

    outdir: Path = args.outdir
    all_tasks = build_plot_tasks(
        spheres_dir=args.spheres,
        labels=args.stars,
    )

    filtered_tasks: list[SphereTask] = []

    for sphere in all_tasks:
        outpath = outdir / sphere.label / sphere.snapshot
        if outpath.exists() and not args.force:
            continue
        filtered_tasks.append(SphereTask(
            label=sphere.label, snapshot=sphere.snapshot, sphere_path=sphere.sphere_path
        ))

    console.log(f"{len(filtered_tasks)} tasks to run; {len(all_tasks) - len(filtered_tasks)} skipped.")

    with Progress(console=console) as progress, MPIPoolExecutor() as executor:
        task_id = progress.add_task("Making Profile Plots", total=len(filtered_tasks))
        futures = [executor.submit(process_sphere, task, outdir, args.profiledir) for task in filtered_tasks]

        for future in as_completed(futures):
            console.log(f"Done {future.result()}")
            progress.advance(task_id)