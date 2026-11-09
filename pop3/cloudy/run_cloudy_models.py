"""

Execution functions for running Cloudy models. Handles MPI, config, aftermaths, etc.

"""

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from rich.console import Console

    from ..utils import CloudyConfig

import subprocess
from concurrent.futures import as_completed
from pathlib import Path

import healpy as hp
import numpy as np
import polars as pl
from mpi4py.futures import MPIPoolExecutor

from ..utils import instantiate_console, load_config, my_progress, parse_args
from .cloudy_model import CloudyModel, CloudyResult
from .commands import SaveCmd
from .parsers import read_nomoto_table
from .ray_extraction import RayArrays, RayHomogeneous, create_cloudy_tasks
from .rays_to_commands import build_cloudy_cmds


class ModelInput(NamedTuple):
    """
    Little container for the inputs to a Cloudy model.

    - `ray` object
    - yields dict
    - snapshot redshift
    - run ID
    """

    ray: RayArrays | RayHomogeneous
    yields: dict[str, float]
    redshift: float
    run_ID: int


def get_star(ledger: pl.DataFrame, config: CloudyConfig, console: Console) -> tuple[np.ndarray, float, str]:
    """
    Uses the config and the ledger to find the requested star. Returns:

    - pos: the star's position in unitary units
    - the redshift of the located snapshot
    - the snapshot name
    """
    console.log(f"Finding star {config.star} in the snapshot where its age is closest to {config.star_age_myr} Myr")

    target = pl.DataFrame({"label": [config.star], "target_age_myr": [config.star_age_myr]}).sort(
        "target_age_myr"
    )  # both dfs must be sorted by the on key in join_asof()

    star_rows = (
        ledger.select("label", "age_myr", "snapshot", "redshift", "positions_unitary", "is_alive")
        .filter((pl.col("label") == config.star) & pl.col("is_alive"))  # make sure we select the star and it is alive
        .sort("age_myr")  # see above comment
    )

    # find the star in the ledger and get its properties as a dict
    closest = (
        target.join_asof(
            star_rows,
            left_on=pl.col("target_age_myr"),
            right_on=pl.col("age_myr"),
            strategy="nearest",
        )
        .with_columns((pl.col("age_myr") - pl.col("target_age_myr")).alias("offset_myr"))
        .row(0, named=True)
    )

    matched_age = closest["age_myr"]
    offset = closest["offset_myr"]
    pos = np.asarray(closest["positions_unitary"], dtype=np.float64)
    redshift = closest["redshift"]
    snapshot = closest["snapshot"]

    console.log(f"Found star {closest['label']} in snapshot {snapshot} at redshift {redshift:.3f}")
    console.log(f"  Position (unitary units): {pos}")
    console.log(f"  Age in this snapshot: {matched_age:.3f} Myr, {offset:.3f} Myr from input")

    return pos, redshift, snapshot


def run_ray(
    model_input: ModelInput,
    config: CloudyConfig,
) -> CloudyResult:
    """
    Worker function. Takes the `ModelInput` namedtuple, makes a `CloudyModel`, then runs Cloudy and
    returns its results.
    """
    # turn the ray object and config into commands for the .in file
    cmds = build_cloudy_cmds(
        ray=model_input.ray,
        yields=model_input.yields,
        redshift=model_input.redshift,
        config=config,
    )
    save_cmd = SaveCmd(
        run_id=model_input.run_ID,
        overview=config.store_overview,
        continuum=config.store_continuum,
        linelist=config.datafile_dir / "linelist.dat" if config.store_lines else None,
    )

    # instantiate the CloudyModel object then call its methods to run a Cloudy pipeline
    model = CloudyModel(
        config=config,
        in_commands=cmds,
        save_command=save_cmd,
    )
    # since we're running 768 models (or more) we want to handle a couple errors
    try:
        model.prepare()
        model.execute()
        return model.package_data()  # will return a Cloudy error
    except subprocess.TimeoutExpired:
        return CloudyResult.err(run_id=model_input.run_ID, reason="timeout")
    except OSError as error:
        return CloudyResult.err(run_id=model_input.run_ID, reason=f"io: {error}")


if __name__ == "__main__":
    # set up logging, read inputs, find the star
    args = parse_args()
    console = instantiate_console()
    config = load_config(args=args)
    nomoto_yields = read_nomoto_table(nomoto_path=config.cloudy.datafile_dir / "pop3_metals.tsv", config=config)
    ledger = pl.read_parquet(config.stars.ledger_path)

    star_pos, redshift, snapshot = get_star(ledger=ledger, config=config.cloudy, console=console)
    snap_path: Path = config.cloudy.snap_dir / snapshot / snapshot

    # prepare sightlines -> turn into yt rays -> iterator which workers can take Cloudy tasks from
    console.log("Preparing Sightlines.")
    model_inputs = (
        ModelInput(run_ID=run_id, ray=ray, yields=nomoto_yields, redshift=redshift)
        for run_id, ray in enumerate(
            create_cloudy_tasks(
                snap_path=snap_path,
                star_pos_unitary=star_pos,
                ray_length_kpc=config.cloudy.ray_length_kpc,
                homogeneous=config.cloudy.homogeneous,
                n_side=config.cloudy.n_side,
            )
        )
    )

    n_rays = hp.nside2npix(config.cloudy.n_side)
    results: list[CloudyResult] = []

    # progress bar + mpi worker pool (cloud emoji on the progress bar is rumoured to speed up cloudy)
    with my_progress(console=console, spinner="weather") as progress, MPIPoolExecutor() as executor:
        task_id = progress.add_task("Running Cloudy Models", total=n_rays)
        # workers grab rays as rank 0 delivers them (create_cloudy_tasks() delivers an iterator)
        futures = {executor.submit(run_ray, model_input, config.cloudy) for model_input in model_inputs}

        for future in as_completed(futures):
            result = future.result()
            progress.advance(task_id)
            msg = "Success" if result.success else f"Error: {result.reason}"
            console.log(f"Cloudy Model {result.run_id}: {msg}")  # explicitly catches errors

            results.append(result)

    console.log("All Cloudy Models Finished!")
    console.log(f"{len([res for res in results if res.success])}/{n_rays} models succeeded.")
