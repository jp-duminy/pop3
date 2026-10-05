"""

Creates yt spheres around the Pop3 stars to be saved as datasets for analysis.

"""

import argparse
import gc
from pathlib import Path
from concurrent.futures import as_completed

import polars as pl
import yt
from mpi4py.futures import MPIPoolExecutor
from rich.progress import Progress

from ..sim.pop3_ledger import select_stars, star_lifetime_summary, add_metallicity3
from ..utils import common_parser, instantiate_logger, DATA_DIR

SPHERE_FIELDS = [

    # basic gas fields
    ("gas", "density"),
    ("gas", "temperature"),
    ("gas", "metallicity3"),

    # number densities
    ("gas", "H_p0_number_density"),
    ("gas", "H_p1_number_density"),
    ("gas", "H_nuclei_density"),
    ("gas", "H2_number_density"),
    ("gas", "He_p0_number_density"),
    ("gas", "He_p1_number_density"),
    ("gas", "He_p2_number_density"),
    ("gas", "El_number_density"),  # electron number density

    # ionisation rates
    ("gas", "H_p0_ionization_rate"),
    ("gas", "He_p0_ionization_rate"),
    ("gas", "He_p1_ionization_rate"),
    ("gas", "photo_gamma"),
    ("gas", "cell_volume"), 
    ("gas", "cell_mass"),

    # place in raw snapshot
    ("index", "x"),
    ("index", "y"),
    ("index", "z"),
]

def build_sphere_tasks(
    ledger: pl.DataFrame, 
    labels: list[str], 
    snapshot_dir: Path
) -> pl.DataFrame:
    """
    Reduces the ledger to only the selected stars and their lifetimes.
    """
    tasks = (
        ledger.pipe(select_stars, labels)
        .join(star_lifetime_summary(ledger), on=["particle_indices", "label"])  # adds birth/death info
        .filter(pl.col("current_time_myr") <= pl.col("first_dead_myr").fill_null(float("inf")))
        .sort("particle_indices", "snapshot")  # time order
        .with_columns(  # add directory info for loading & saving
            pl.format("{}/{}/{}", pl.lit(str(snapshot_dir)), pl.col("snapshot"), pl.col("snapshot"))  # concatenate directories
                .alias("snapshot_path"),
            pl.format("{}_{}", pl.col("label"), pl.col("snapshot")).alias("base_name"),  # concatenate star name + snap name
        )
    )

    return tasks


def parse_args() -> argparse.Namespace:
    """
    Parses the command-line arguments; returns the corresponding Namespace object.
    """
    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        prog="star-spheres",
        description="Routine for collecting yt spheres around specified Pop3 stars.",
        suggest_on_error=True,
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=1.0,
        help="Fixed radius of spheres (kpc)."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrites existing data files."
    )

    return parser.parse_args()


def create_star_sphere(
    row: dict[str, object], 
    outpath: Path, 
    radius_kpc: float, 
) -> Path:
    """
    Generates a sphere of specified radius_kpc to outpath, and stores the fields
    specified in the module-level SPHERE_FIELDS. This function should be called by workers.
    """
    yt.set_log_level("error")

    ds = yt.load(row["snapshot_path"])
    add_metallicity3(ds=ds)

    centre = ds.arr(row["positions_unitary"], "unitary")
    radius = ds.quan(radius_kpc, "kpc")

    sp = ds.sphere(centre, radius)
    sp.save_as_dataset(filename=str(outpath), fields=SPHERE_FIELDS)

    del ds
    gc.collect()

    return outpath


def compute_hii_region_size() -> None:
    """
    Takes in _ and returns the size of the HII region (under construction).
    """
    pass


if __name__ == "__main__":

    args = parse_args()
    outdir: Path = args.outdir

    console = instantiate_logger()

    ledger = pl.read_parquet(args.ledger)
    labels = args.stars or star_lifetime_summary(ledger)["label"].to_list()

    all_tasks: pl.DataFrame = build_sphere_tasks(
        ledger=ledger, 
        labels=labels,
        snapshot_dir=DATA_DIR,
    )

    # make a directory for each requested star
    for label in all_tasks["label"].unique():
        (outdir / label).mkdir(parents=True, exist_ok=True)

    # reduce tasks if the outputs already exist
    filtered_tasks: list[tuple[dict[str, object], Path]] = []

    for row in all_tasks.iter_rows(named=True):
        outpath = outdir / row["label"] / f"{row['base_name']}.h5"
        if outpath.exists() and not args.force:
            continue
        filtered_tasks.append((row, outpath))

    console.log(f"{len(filtered_tasks)} tasks to run; {all_tasks.height - len(filtered_tasks)} skipped.")

    with Progress(console=console) as progress, MPIPoolExecutor() as executor:
        task_id = progress.add_task("Creating spheres", total=len(filtered_tasks))
        futures = [
            executor.submit(create_star_sphere, row, outpath, args.radius)
            for row, outpath in filtered_tasks
        ]

        for future in as_completed(futures):  # as_completed() gives us the iterator for the progress bar
            outpath = future.result()
            console.log(f"Saved {outpath}.")
            progress.advance(task_id)