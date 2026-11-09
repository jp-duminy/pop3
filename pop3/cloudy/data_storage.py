"""

Functionality for creating a HDF5 file to store the outputs of a single Cloudy run.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils import CloudyConfig
    from .cloudy_model import CloudyResult


import subprocess
from dataclasses import asdict
from pathlib import Path

import h5py
import healpy as hp
import numpy as np


def create_output_file(config: CloudyConfig, force: bool = True) -> Path:
    """
    Creates the output HDF5 file at `output_dir` / `{star}_{age}`.
    """
    output_path = config.output_dir / f"{config.star}_{config.star_age_myr!s}Myr.h5"

    if output_path.exists():
        if force:
            output_path.unlink()
        else:
            raise ValueError(f"Warning: {output_path} already exists!!")

    commit_hash = _get_git_commit()
    n_rays = hp.nside2npix(config.n_side)

    with h5py.File(output_path, "w") as out:
        metadata = out.create_group("metadata")
        metadata.attrs["n_rays"] = n_rays

        for param, value in asdict(config).items():
            if isinstance(value, Path):  # Path objects can't go in directly
                value = str(value)
            elif value is None:  # nor can these
                value = "None"
            metadata.attrs[param] = value

        if commit_hash is not None:
            metadata.attrs["git_commit"] = commit_hash

        cloudy = out.create_group("cloudy_data")
        cloudy.create_dataset("success", shape=n_rays, dtype=np.bool_)

        if config.store_overview:
            ovr = cloudy.create_group("ovr")

            for name in ["starts", "lengths"]:
                ovr.create_dataset(name, shape=n_rays, fillvalue=0, dtype=np.int32)

        if config.store_continuum:
            cloudy.create_group("con")
        if config.store_lines:
            cloudy.create_group("lin")

    return output_path


def write_result(out: h5py.File, result: CloudyResult) -> None:
    """
    Writes a `CloudyResult` object to the output file.
    """
    cloudy = out["cloudy_data"]
    n_rays = out["metadata"].attrs["n_rays"]

    cloudy["success"][result.run_id] = result.success

    if not result.success:  # early exit
        return

    # number of zones differs between runs so this uses start and length arrays with resized datasets
    if result.overview is not None:
        ovr = cloudy["ovr"]
        for column, values in result.overview.items():
            dataset_name = column.replace("/", "_per_")  # annoyingly h5py can't take the / character in a name
            if dataset_name not in ovr:  # must be resizeable (maxshape=(None,) lets the colum be resized)
                dataset = ovr.create_dataset(
                    dataset_name, shape=(0,), maxshape=(None,), dtype=values.dtype, chunks=True
                )  # should be float64
                dataset.attrs["cloudy_name"] = column  # lets names be keyed by Cloudy names later

            dataset = ovr[dataset_name]
            old_size = dataset.shape[0]
            dataset.resize((old_size + values.size,))
            dataset[old_size:] = values
            n_zones = values.size

        # once the loop concludes, since all arrays are aligned, the index arrays the index arrays can be written using the final dataset
        ovr["starts"][result.run_id] = old_size
        ovr["lengths"][result.run_id] = n_zones

    if result.continuum is not None:
        con = cloudy["con"]

        if "wavelength_angstrom" not in con:
            wavelengths = result.continuum["wavelength_angstrom"]
            con.create_dataset("wavelength_angstrom", data=wavelengths)
            con.create_dataset("combined", shape=(n_rays, len(wavelengths)), fillvalue=np.nan, dtype=np.float64)

        con["combined"][result.run_id] = result.continuum["combined"]

    if result.lines is not None:
        lin = cloudy["lin"]

        if "labels" not in lin:
            lin.create_dataset(
                "labels", dtype=h5py.string_dtype(), data=list(result.lines.keys())
            )  # string_dtype() does exist but VS Code can't see it somehow
            lin.create_dataset("values", dtype=np.float64, shape=(n_rays, len(result.lines)), fillvalue=np.nan)

        lin["values"][result.run_id] = np.fromiter(result.lines.values(), dtype=np.float64)

    cloudy["success"][result.run_id] = result.success


# NOTE: copied over from octavius
def _get_git_commit() -> str | None:
    """
    Tries to retrieve the git commit; wraps in try/except to avoid stalling on clusters.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,  # since mpiexec allows running elsewhere
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None
