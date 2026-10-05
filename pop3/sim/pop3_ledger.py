"""

This file contains the polars DataFrame functionality for storing and querying
the pop3 stars' properties across the simulation. 

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yt.data_objects.particle_filters import ParticleFilter
    from yt.data_objects.data_containers import YTDataContainer
    from yt.data_objects.static_output import Dataset

from pathlib import Path
import argparse

import yt
from yt.data_objects.particle_filters import add_particle_filter
import numpy as np
import polars as pl
from mpi4py.futures import MPIPoolExecutor
from rich.progress import Progress

from ..utils import instantiate_logger, common_parser, DATA_DIR

FIELD_MAP = {
    "positions_unitary": ("particle_position", "unitary"),
    "ptypes": ("particle_type", "dimensionless"),
    "particle_indices": ("particle_index", "dimensionless"),
    "masses_msun": ("particle_mass", "Msun"),
    "metallicities": ("metallicity_fraction", "dimensionless"),
    "creation_times_myr": ("creation_time", "Myr"),
}
MIN_SEP_MYR = 1e-3  # to filter Enzo's invasive star population


def parse_args() -> argparse.Namespace:
    """
    Parses the command-line arguments; returns the corresponding Namespace object.
    """
    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        prog="pop3-ledger",
        description="Creates the POP3 ledger.",
        suggest_on_error=True,
    )
    parser.add_argument(
        "-m",
        "--metal",
        type=float,
        default=1e-10,
        help="Maximum metallicity for a Pop3 star."
    )

    return parser.parse_args()

def age_myr() -> pl.Expr:
    """
    Calculates star age in Myr from current_time - creation_time.
    """
    return (pl.col("current_time_myr") - pl.col("creation_times_myr")).alias("age_myr")

def star_label() -> pl.Expr:
    """
    Names the star based on its formation time.
    """
    return (pl.format("POP3-{}", pl.col("creation_times_myr").floor().cast(pl.Int64))).alias("label")

def is_alive() -> pl.Expr:
    """
    Determines whether or not a star is alive.
    """
    return ((pl.col("ptypes") == 5) & (pl.col("masses_msun") > 1e-3)).alias("is_alive")

def alive_snapshots(ledger: pl.DataFrame) -> pl.DataFrame:
    """
    Finds the snapshots wherein each star is alive.
    """
    return (
        ledger.filter(is_alive())
        .group_by("particle_indices")
        .agg(pl.col("snapshot").sort_by("current_time_myr"))
    )

def star_lifetime_summary(ledger: pl.DataFrame) -> pl.DataFrame:
    """
    Returns lifetime stats for the stars.
    """
    return (
        ledger.group_by("particle_indices", "label")
        .agg(
            pl.col("creation_times_myr").first(),
            pl.col("current_time_myr").filter(pl.col("is_alive")).max().alias("last_alive_myr"),
            pl.col("current_time_myr").filter(~pl.col("is_alive")).min().alias("first_dead_myr"),
        )
        .sort("creation_times_myr")
    )

def filter_duplicates(ledger: pl.DataFrame, min_sep_myr: float = 1e-3) -> pl.DataFrame:
    """
    In the simulation, stars with negligible mass can spawn in straight after a Pop3 stars (let's call them
    invasive stars); this filters them on a min_sep_myr threshold on their ages.
    """
    # get a reduced dataframe of invasive stars (those which come within min_sep_myr of a Pop3)
    invasive_stars =  (
        ledger.select("particle_indices", "creation_times_myr").unique(subset="particle_indices")
        .sort("creation_times_myr", "particle_indices")  # use index as a tiebreak in case stars have ~= formation time
        .filter(pl.col("creation_times_myr").diff() < min_sep_myr)
    )

    return ledger.join(invasive_stars, on="particle_indices", how="anti")  # drop those stars from the ledger

def select_stars(ledger: pl.DataFrame, labels: list[str]) -> pl.DataFrame:
    """
    Selects stars based on their labels.
    """
    typos = set(labels) - set(ledger["label"].unique())

    if typos:  # guard here so you don't find out the following morning you made a typo 
        raise ValueError(f"Typos: {sorted(typos)}")
    
    return ledger.filter(pl.col("label").is_in(labels))

def create_snapshot_dataframe(
    snapshot: Path, 
    pop3_threshold: float, 
    field_map: dict[str, tuple[str, str]], 
) -> pl.DataFrame | None:
    """
    Helper to create a dataframe from a reduced snapshot catalogue such that the loop over the catalogues
    can be parallelised.
    """
    yt.set_log_level("error")  # avoids lots of verbose output (new subprocesses spawned)

    ds = yt.load(snapshot)
    metallicities = ds.data[("pop3", "metallicity_fraction")]
    pop3_mask = np.asarray(metallicities < pop3_threshold).nonzero()[0]

    if pop3_mask.shape[0] == 0:  # skip snaps with no pop3 stars
        return None

    # create pure ndarrays from unyt arrays from the field map
    columns: dict[str, np.ndarray] = {
        name: ds.data[("pop3", field)][pop3_mask].to(unit).d
        for name, (field, unit) in field_map.items()
    }
    columns["positions_unitary"] = pl.Series(columns["positions_unitary"], dtype=pl.Array(pl.Float64, 3))  # treat pos vector

    # snapshot info
    redshift = ds.current_redshift
    current_time = ds.current_time.to("Myr")

    frame = pl.DataFrame(columns).with_columns(
        pl.col("particle_indices", "ptypes").cast(pl.Int64),
        pl.lit(snapshot.stem).alias("snapshot"),
        pl.lit(float(redshift)).alias("redshift"),  # need float() to remove unyt
        pl.lit(float(current_time)).alias("current_time_myr"),
    )

    return frame

# NOTE: from the sim source paper we have one star forming at z ~ 23.7 and another at z ~ 18.2 (first at DD0032)
# NOTE: from Britton we know they live ~3.5 Myr 

def _pop3(pfilter: ParticleFilter, data: YTDataContainer):
    """
    Filters particles to the conditions expected for Pop3 particles.
    """
    # stars (ptype 5): filter to supernova remnants and living stars
    pop3_remnant = (data["particle_type"] == 5) & (data["particle_mass"].in_units("Msun") < 1e-10)
    pop3_star = (data["particle_type"] == 5) & (data["particle_mass"].in_units("Msun") > 1e-3)

    # dm (ptype 1): HACK: enzo stores dead stars as dm particles with nonzero creation time
    pop3_dm = (data["particle_type"] == 1) & (data["creation_time"] > 0) & (data["particle_mass"].in_units("Msun") > 1)

    return pop3_remnant | pop3_dm | pop3_star

add_particle_filter("pop3", function=_pop3, filtered_type="all",
                    requires=["particle_type", "creation_time", "particle_mass"])


def add_metallicity3(ds: Dataset) -> None:
    """
    Adds the metallicity3 field to a yt dataset.
    """
    if ("gas", "metallicity3") in ds.derived_field_list:
        return

    ds.unit_registry.modify("Zsun", ds.parameters["SolarMetalFractionByMass"])

    ds.add_field(
        ("gas", "metallicity3"), 
        function=lambda field, data: data["enzo", "SN_Colour"] / data["gas", "density"],
        units="Zsun",
        sampling_type="cell",
    )


if __name__ == "__main__":
    """
    Constructs the Pop3 ledger.
    """

    yt.set_log_level("warning")  # avoids lots of verbose output
    args = parse_args()
    parquet_path = args.outdir / "pop3_ledger.parquet"
    console = instantiate_logger()

    snapshot_paths = sorted((DATA_DIR / "pop3").glob(pattern="DD*.h5"))  # sort; futures preserves order when writing to list

    with Progress(console=console) as progress, MPIPoolExecutor() as executor:

        task_id = progress.add_task("Making Pop3 Ledger", total=len(snapshot_paths))
        task_args = ((path, args.metal, FIELD_MAP) for path in snapshot_paths)

        frame_list: list[pl.DataFrame] = []

        # NOTE: looping is just for the progress bar, you could do frame_list = executor.starmap(...)
        for frame in executor.starmap(create_snapshot_dataframe, task_args):  # found this cool method on mpi4py.futures
            progress.advance(task_id) 
            if frame is not None:
                frame_list.append(frame)

    ledger = pl.concat(frame_list)  # vertical concat
    ledger = ledger.pipe(filter_duplicates, min_sep_myr=MIN_SEP_MYR).with_columns(age_myr(), is_alive(), star_label())

    console.log(f"Total Pop3 Stars: {ledger.n_unique("particle_indices")}")

    with pl.Config(tbl_rows=-1):  # set this so the whole dataframe prints
        df_diagnostic = (
        ledger.sort("particle_indices")
        .unique("particle_indices", keep="first", maintain_order=True)
        .select("label", "particle_indices", "creation_times_myr", "metallicities")
        )
        console.log(df_diagnostic)

    ledger.write_parquet(parquet_path)
    console.log(f"Wrote parquet file at {parquet_path}")
 
