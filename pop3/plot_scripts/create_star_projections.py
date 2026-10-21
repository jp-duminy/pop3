"""

This routine generates projection plots for density, metallicity and temperature (weighted
by density) for requested stars. It is an expensive one, taking multiple hours for one star.

"""

import gc
import logging
from concurrent.futures import as_completed
from pathlib import Path

import polars as pl
import yt
from mpi4py.futures import MPIPoolExecutor

from ..sim.pop3_ledger import add_metallicity3, select_stars, star_lifetime_summary
from ..utils import (
    instantiate_logger,
    load_config,
    my_progress,
    parse_args,
)


def build_film_tasks(
    ledger: pl.DataFrame, labels: list[str], post_sn_cutoff: float, snapshot_dir: Path
) -> pl.DataFrame:
    """
    Creates a dataframe containing the following info for making a film:

    - particle_indices: particle IDs
    - label: the label of each star for titles
    - snapshot_path: the paths to each raw snapshot
    - centre_unitary: the positions of the stars in the raw snapshot
    - base_name: the {label}_{snapshot_stem} str.

    The columns are filtered to between when the stars were born and post_sn_cutoff after death.
    """
    tasks = (
        ledger.pipe(select_stars, labels)
        .join(star_lifetime_summary(ledger), on=["particle_indices", "label"])  # adds birth/death info
        .filter(
            pl.col("current_time_myr") <= pl.col("first_dead_myr").fill_null(float("inf")) + post_sn_cutoff
        )  # filter to where we want to visualise
        .sort("particle_indices", "snapshot")  # time order
        .with_columns(  # add global plotting info
            pl.col("positions_unitary")
            .first()
            .over("particle_indices")
            .alias("centre_unitary"),  # position of star in its first snap
            pl.format("{}/{}/{}", pl.lit(str(snapshot_dir)), pl.col("snapshot"), pl.col("snapshot")).alias(
                "snapshot_path"
            ),  # concatenate directories
            pl.format("{}_{}", pl.col("label"), pl.col("snapshot")).alias(
                "base_name"
            ),  # concatenate star name + snap name
        )
        .select("particle_indices", "label", "snapshot_path", "centre_unitary", "base_name")
    )

    if not all(Path(path).exists() for path in tasks["snapshot_path"]):  # quick guard
        logger = logging.getLogger(__name__)
        logger.warning("Warning: not all snapshot paths exist (should not happen by construction).")

    return tasks


def project_star_snapshot(
    row: dict[str, object],
    outpath: Path,
    width_kpc: float,
    axis: str = "x",
) -> Path:
    """
    Generates a single projection plot of the star from a snapshot, saved as a .h5 file
    to outdir. This function should be called by workers.
    """
    yt.set_log_level("error")

    fields = [("gas", "density"), ("gas", "temperature"), ("gas", "metallicity3")]
    weight_field = ("gas", "density")

    ds = yt.load(row["snapshot_path"])
    add_metallicity3(ds=ds)

    centre = ds.arr(row["centre_unitary"], "unitary")
    width = ds.quan(width_kpc, "kpc")

    # britton's dimensions
    region = ds.box(centre - 1.05 * width / 2, centre + 1.05 * width / 2)

    p = yt.ProjectionPlot(ds, axis, fields, weight_field=weight_field, center=centre, width=width, data_source=region)

    data = {field[1]: p.frb[field] for field in fields}  # package the projection plot into a dict

    del p  # free up memory

    extra_attrs = {"centre_unitary": centre.to("unitary"), "width_kpc": width.to("kpc")}
    yt.save_as_dataset(ds, filename=str(outpath), data=data, extra_attrs=extra_attrs)  # yt doesn't like Path objects

    region.clear_data()
    del region  # free up memory again and force garbage collection
    del ds
    gc.collect()

    return outpath


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)

    console = instantiate_logger()

    ledger = pl.read_parquet(config.stars.ledger_path)
    labels = config.stars.process_stars or star_lifetime_summary(ledger)["label"].to_list()

    all_tasks: pl.DataFrame = build_film_tasks(
        ledger=ledger,
        labels=labels,
        post_sn_cutoff=config.plots.post_death,
        snapshot_dir=config.spheres.data_dir,
    )

    # make a directory for each requested star
    for label in all_tasks["label"].unique():
        (config.plots.proj_dir / label).mkdir(parents=True, exist_ok=True)

    # reduce tasks if the outputs already exist
    filtered_tasks: list[tuple[dict[str, object], Path]] = []

    for row in all_tasks.iter_rows(named=True):
        outpath = config.plots.proj_dir / row["label"] / f"{row['base_name']}_{config.plots.axis}.h5"
        if outpath.exists() and not args.force:
            continue
        filtered_tasks.append((row, outpath))

    console.log(f"{len(filtered_tasks)} tasks to run; {all_tasks.height - len(filtered_tasks)} skipped.")

    # pool workers for film making in conjunction with task-based progress bar
    with my_progress(console=console) as progress, MPIPoolExecutor() as executor:
        task_id = progress.add_task("Projecting", total=len(filtered_tasks))
        futures = [
            executor.submit(project_star_snapshot, row, outpath, config.plots.frame_width, config.plots.axis)
            for row, outpath in filtered_tasks
        ]

        for future in as_completed(futures):  # as_completed() gives us the iterator for the progress bar
            outpath = future.result()
            console.log(f"Saved {outpath}.")
            progress.advance(task_id)
