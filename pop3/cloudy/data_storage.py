"""

Functionality for creating a HDF5 file to store the outputs of a single Cloudy run.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from ..utils import CloudyConfig
    from .cloudy_model import CloudyResult


import subprocess
from dataclasses import asdict
from pathlib import Path

import h5py
import healpy as hp
import numpy as np


class OutputCatalogue:
    """
    Object-oriented interface for a single HDF5 catalogue containing the outputs of Cloudy models from a single
    star in a given snapshot with specified physics parameters.
    """

    def __init__(self, catalogue_path: Path) -> None:

        self.catalogue_path = catalogue_path
        self._file = h5py.File(catalogue_path, "r")
        # eager load this
        self._metadata: dict[str, Any] = dict(self._file["metadata"].attrs)

    def close(self) -> None:
        """
        Closes the HDF5 file.
        """
        self._file.close()

    def __repr__(self) -> str:
        return f"Catalogue for {self.star} in {self.snapshot_path.name} ({self.star_age_myr} Myr old): {self.n_succeeded} successful runs."

    def __len__(self) -> int:
        return self._metadata["n_rays"]

    @property
    def n_succeeded(self) -> int:
        return np.sum(self._file["cloudy_data"]["success"])

    @property
    def ray_length_kpc(self) -> float:
        """
        The configured ray length, in kpc, which produced this catalogue. Keep in mind this does not necessarily reflect
        the stopping length of each ray: this is the upper bound.
        """
        return self._metadata["ray_length_kpc"]

    @property
    def star(self) -> str:
        """
        The label of the star which produced this catalogue.
        """
        return self._metadata["star"]

    @property
    def star_age_myr(self) -> float:
        """
        The true age of the star in the snapshot which this catalogue was produced from.
        """
        return self._metadata["star_age_myr"]

    @property
    def snapshot_path(self) -> Path:
        """
        `Path` object pointing to the snapshot which produced this catalogue.
        """
        return Path(self._metadata["snap_path"])

    def successes(self) -> np.ndarray:
        return self._file["cloudy_data"]["success"][:]

    def read_overviews(self, columns: list[str]) -> list[dict[str, np.ndarray] | None]:
        """
        Helper for getting the overviews back from the file.
        """
        overview_group: h5py.Group = self._file["cloudy_data/ovr"]
        # these aren't very big so can do everything in-memory
        starts = overview_group["starts"][:]
        lengths = overview_group["lengths"][:]
        flat = {name: overview_group[name][:] for name in ["depth", *columns]}

        ovrs: list[dict[str, np.ndarray] | None] = []
        for start, length in zip(starts, lengths, strict=True):
            run_slice = slice(start, start + length)
            ovrs.append({name: values[run_slice] for name, values in flat.items()} if length > 0 else None)

        return ovrs

    def _align_cloudy_depth(
        self,
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


def get_output_filename(config: CloudyConfig, star: str, star_age_myr: float) -> Path:
    """
    Gets the output HDF5 filename of a run.
    """
    suffix = "_hom" if config.physics.homogeneous else ""
    path = config.misc.output_dir / f"{star}_{star_age_myr:.3f}Myr{suffix}.h5"

    return path


def create_output_file(
    config: CloudyConfig, star_name: str, star_age_myr: float, snapshot_path: Path, force: bool = False
) -> Path:
    """
    Creates the output HDF5 file at `output_dir` / `{star}_{age}`.
    """
    output_path: Path = get_output_filename(config=config, star=star_name, star_age_myr=star_age_myr)

    if output_path.exists():
        if force:
            output_path.unlink()
        else:
            raise ValueError(f"Warning: {output_path} already exists!!")

    commit_hash = _get_git_commit()
    n_rays = hp.nside2npix(config.star.n_side)

    with h5py.File(output_path, "w") as out:
        metadata = out.create_group("metadata")
        metadata.attrs["n_rays"] = n_rays
        metadata.attrs["snap_path"] = str(snapshot_path)  # Path object can't go in directly

        run_params = (
            asdict(config.physics)
            | asdict(config.misc)
            | {
                "star": star_name,
                "star_age_myr": star_age_myr,
                "ray_length_kpc": config.star.ray_length_kpc,
            }
        )
        for param, value in run_params.items():
            if isinstance(value, Path):  # Path objects can't go in directly
                value = str(value)
            elif value is None:  # nor can these
                value = "None"
            metadata.attrs[param] = value

        if commit_hash is not None:
            metadata.attrs["git_commit"] = commit_hash

        cloudy = out.create_group("cloudy_data")
        cloudy.create_dataset("success", shape=n_rays, dtype=np.bool_)

        if config.misc.store_overview:
            ovr = cloudy.create_group("ovr")

            for name in ["starts", "lengths"]:
                ovr.create_dataset(name, shape=n_rays, fillvalue=0, dtype=np.int32)

        if config.misc.store_continuum:
            cloudy.create_group("con")
        if config.misc.store_lines:
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
