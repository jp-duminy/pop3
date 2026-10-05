"""

Creates profile and phase plots for stars across their snapshots.

"""

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from yt.data_objects.static_output import Dataset
    from yt.data_objects.profiles import ProfileND

import argparse
from pathlib import Path
from concurrent.futures import as_completed

import numpy as np
import yt
from mpi4py.futures import MPIPoolExecutor
from rich.progress import Progress

from ..utils import common_parser, instantiate_logger, DATA_DIR

def parse_args() -> argparse.ArgumentParser:
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


def load_sphere(sphere_path: Path) -> tuple[Dataset, np.ndarray]:
    """
    Reload; return ds and star centre; register radius derived field.
    """
    pass

def build_radial_profiles(sphere_ds: Dataset, field_names: list[tuple[str, str]]) -> dict[str, ProfileND]:
    """
    Radial profiles for 5 fields weighted by cell mass as per Britton:

    - density
    - metallicity
    - temperature
    - HII density
    - HI density
    """
    pass

def build_phase_profiles(sphere_ds: Dataset) -> dict[str, ProfileND]:
    """
    Phase profiles for the following, binned by cell mass:
    
    - density-temperature
    - density-metallicity
    - temperature-metallicity
    """
    pass

def plot_radial_figure(profiles: dict[str, ProfileND], title: str, output_path: Path) -> None: 
    """
    Makes a plot of the radial profiles for a snapshot.
    """
    pass

def plot_phase_figure(profiles: dict[str, ProfileND], title: str, output_path: Path) -> None: 
    """
    Makes a plot of the phase diagrams for a snapshot.
    """
    pass