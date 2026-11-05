"""

Unorganised functions for the test Cloudy run.

"""

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from rich.console import Console

    from ..utils import Pop3Config
    from .cloudy_model import CloudyResult
    from .commands import CloudyCommand

from pathlib import Path

import numpy as np
import polars as pl
import yt
from unyt import unyt_array

from ..sim.pop3_ledger import add_metallicity3
from ..utils import instantiate_logger, load_config, parse_args
from .cloudy_model import CloudyModel
from .commands import (
    HAZY_FLOOR,
    Abundances,
    Background,
    Blackbody,
    DlawTable,
    ElementTable,
    RadSource,
    SaveCmd,
    SphereGeometry,
    StopCondition,
    TlawTable,
)
from .parsers import read_nomoto_table
from .ray_extraction import cast_rays

FIELDS: dict[str, tuple[str, str]] = {
    "density": ("gas", "density"),
    "temperature": ("gas", "temperature"),
    "metallicity": ("gas", "metallicity3"),
    "h_density": ("gas", "H_nuclei_density"),
}


class RayArrays(NamedTuple):
    """
    Container for the four primary quantities of interest coming out of the rays. All quantities
    are in CGS units.

    - density
    - temperature
    - metallicity
    - h_density
    - depth (midpoints of ray cells)
    """

    density: unyt_array
    temperature: unyt_array
    metallicity: unyt_array
    h_density: unyt_array
    r0: unyt_array
    depth: unyt_array


def test_ray_extraction(snap_path: Path, star_pos: np.ndarray, ray_length: float) -> tuple[RayArrays, float]:
    """
    Extract rays from the specified star. Returns a nicely-packaged RayArrays namedtuple and the redshift.
    """
    ds = yt.load(snap_path)
    add_metallicity3(ds=ds)
    centre = ds.arr(star_pos, "unitary")
    radius = ds.quan(ray_length, "kpc").to("unitary")

    all_coords = cast_rays(centre=centre, radius=radius)
    test_coords = all_coords[:, 0]

    ray = ds.ray(centre, test_coords)
    ray_t = ray["t"]  # where the ray enters each cell
    ray_dts = ray["dts"]  # length of ray segment inside each cell
    ray_sort = np.argsort(ray_t, stable=True)  # depth-order of ray

    ray_t = ray_t[ray_sort]
    ray_dts = ray_dts[ray_sort]

    # I'm defining r0 as the position of the first cell hit by the star cell's ionising radiation
    r0 = radius * (ray_t[0] + ray_dts[0])
    midpoints = ((ray_t + ray_dts / 2)[1:] * radius) - r0  # the midpoint of each cell shifted so r0 is the origin

    arrays = RayArrays(
        **{name: ray[field][ray_sort][1:] for name, field in FIELDS.items()},  # an elegant unpack indeed
        # these should come out in CGS but let's assure ourselves
        r0=r0.to("cm"),
        depth=midpoints.to("cm"),
    )

    return arrays, ds.current_redshift


def ray_to_commands(
    ray: RayArrays,
    yields: dict[str, float],
    redshift: float,
) -> list[CloudyCommand]:
    """
    Converts a RayArrays object to a list of necessary commands for the `CloudyModel`.
    """
    cmds: list[CloudyCommand] = []
    # prepend the tiny float to the depth arrays (hazy stipulates 1e-30 so let's go with 1e-35)
    # concat instead of 0-index then [1:] assignment is safer
    log_depths = np.log10(np.concatenate(([HAZY_FLOOR], ray.depth.to("cm").d)))

    h_density = ray.h_density.to("cm**-3").d
    log_h_density = np.log10(np.concatenate((h_density[:1], h_density)))

    temperature = ray.temperature.to("K").d
    log_temperature = np.log10(np.concatenate((temperature[:1], temperature)))

    metallicity = np.maximum(ray.metallicity.to("dimensionless").d, 1e-34)
    log_metallicity = np.log10(np.concatenate((metallicity[:1], metallicity)))

    cmds.extend(
        [
            DlawTable(log_densities=log_h_density, log_path_lengths=log_depths),
            TlawTable(log_temperatures=log_temperature, log_path_lengths=log_depths),
        ]
    )

    element_tables: list[ElementTable] = []
    for element, log_abundance in yields.items():
        element_tables.append(
            ElementTable(
                element_name=element,
                log_path_lengths=log_depths,
                log_abundances=log_metallicity + log_abundance,
            )
        )
    cmds.append(Abundances(elements=element_tables))

    cmds.append(SphereGeometry(log_inner_radius=np.log10(ray.r0.to("cm").d)))
    cmds.append(
        StopCondition(electron_frac=0.01, log_max_depth=log_depths[-1] - 0.01)
    )  # go 0.01cm back so Cloudy doesn't extrapolate-abort
    cmds.append(Background(redshift=redshift, use_cosmic_rays=True))
    cmds.append(RadSource(shape=Blackbody(log_temperature=4.9), log_q_hydrogen=49.4))  # schaerer+2003 vals

    return cmds


def run_test_model(config: Pop3Config, ledger: pl.DataFrame, console: Console) -> CloudyResult:
    """
    Builds and runs the `CloudyModel`.
    """
    star_pos_unitary = np.asarray(
        (
            ledger.filter(
                (pl.col("label") == config.cloudy.star) & (pl.col("snapshot") == str(config.cloudy.snap_path.name))
            )
            .select(pl.col("positions_unitary"))
            .item()
        ),
        dtype=np.float64,
    )
    nomoto_table = read_nomoto_table(nomoto_path=config.cloudy.datafile_dir / "pop3_metals.tsv", config=config)
    console.log(nomoto_table)
    console.log("Successfully parsed datafiles.")

    rays, redshift = test_ray_extraction(
        snap_path=config.cloudy.snap_path, star_pos=star_pos_unitary, ray_length=config.cloudy.ray_length_kpc
    )
    console.log("Successfully extracted rays.")
    cmd_list = ray_to_commands(ray=rays, yields=nomoto_table, redshift=redshift)
    save_cmd = SaveCmd(run_id=1, overview=True, continuum=True, linelist=config.cloudy.datafile_dir / "linelist.dat")
    console.log("Generated commands.")

    model = CloudyModel(config=config.cloudy, in_commands=cmd_list, save_command=save_cmd)
    console.log("Successfully instantiated a CloudyModel.")

    model.prepare()
    console.log(f"Wrote input files at {model.run_dir}.")
    model.execute()
    console.log("Successfully ran Cloudy! Sending back data.")
    result = model.package_data()

    return result


if __name__ == "__main__":
    args = parse_args()
    console = instantiate_logger()
    config = load_config(args=args)
    ledger = pl.read_parquet(config.stars.ledger_path)

    result = run_test_model(config=config, ledger=ledger, console=console)

    if result.status == "error":
        console.log("Error raised :(")
