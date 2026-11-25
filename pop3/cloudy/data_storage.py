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
from functools import cached_property
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import polars as pl


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
        return np.sum(self.successes)

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

    @cached_property
    def _ovr_starts(self) -> np.ndarray:
        """
        Index array for the start of each run in the overview dataset.
        """
        return self._file["cloudy_data"]["ovr"]["starts"][:]

    @cached_property
    def _ovr_lengths(self) -> np.ndarray:
        """
        Index array for the length of each run in the overview dataset.
        """
        return self._file["cloudy_data"]["ovr"]["lengths"][:]

    @cached_property
    def _ovr_datasets(self) -> dict[str, str]:
        """
        List of available datasets in the .ovr file. Translates the `_per_` renaming back into
        the actual Cloudy name.
        """
        ovr_group = self._file["cloudy_data"]["ovr"]
        datasets = {
            dataset.attrs["cloudy_name"]: name for name, dataset in ovr_group.items() if "cloudy_name" in dataset.attrs
        }

        return datasets

    @cached_property
    def continuum_wavelengths(self) -> np.ndarray:
        """
        The full wavelengths array for aligning to the continuum file outputs. This is handled separately because otherwise
        everything continuum-related becomes 3D and a slight memory bomb.
        """
        return self._file["cloudy_data"]["con"]["wavelength_angstrom"][:]

    @cached_property
    def successes(self) -> np.ndarray:
        """
        Boolean array in run ID order of whether runs were successful or not.
        """
        return self._file["cloudy_data"]["success"][:]

    def read_single_overview(self, run_id: int, columns: list[str] | None = None) -> dict[str, np.ndarray] | None:
        """
        Reads a single model's .ovr file from the HDF5 catalogue. This is mainly for analysing a single ray in post.
        """
        ovr_group = self._file["cloudy_data"]["ovr"]
        start = self._ovr_starts[run_id]
        length = self._ovr_lengths[run_id]

        if length == 0:
            return None

        run_slice = slice(start, start + length)
        names = columns if columns is not None else list(self._ovr_datasets)

        run_ovr = {
            cloudy_name: ovr_group[self._ovr_datasets[cloudy_name]][run_slice]
            for cloudy_name in ["depth", *names]  # always return depth
        }
        cloudy_depth = run_ovr["depth"]
        # cloudy_depth is the midpoint -> 0.5 * (x_i + x_(i+1)) gives approximate right edge
        edges = 0.5 * (cloudy_depth[:-1] + cloudy_depth[1:])
        # recreate the edge of the final bin
        outer_edges = np.concatenate((edges, [2.0 * cloudy_depth[-1] - edges[-1]]))
        run_ovr["depth_input"] = outer_edges

        return run_ovr

    def read_overview(self, columns: list[str] | None = None, drop_failed: bool = True) -> pl.DataFrame:
        """
        Reads all models' .ovr files back from the HDF5 catalogue. Returns:

        `pl.DataFrame`, sorted by contiguous run ID, where each row contains depth-ordered quantities from the
        .ovr file for a zone in the given run ID. The values come from the final iteration.
        """
        ovr_group: h5py.Group = self._file["cloudy_data/ovr"]
        names = columns if columns is not None else list(self._ovr_datasets)
        # these aren't very big so can do everything in-memory
        flat_datasets = {name: ovr_group[self._ovr_datasets[name]][:] for name in ["depth", *names]}

        # run IDs aren't contiguously-ordered since they get written as they come back
        order = np.argsort(self._ovr_starts)
        run_ids = np.repeat(order, self._ovr_lengths[order])
        successes = self.successes[run_ids]

        df = (
            pl.DataFrame({"run_id": run_ids, "success": successes, **flat_datasets})
            .filter(
                pl.col("success") | pl.lit(not drop_failed)
            )  # unioning success with the literal bool input drops failed runs
            .sort("run_id", maintain_order=True)
        )

        # add input depth column for aligning Cloudy reported depth to evaluated depth (outer edge of each zone dr)
        next_depth = pl.col("depth").shift(-1).over("run_id")
        previous_depth = pl.col("depth").shift(1).over("run_id")

        df = df.with_columns(
            (0.5 * (pl.col("depth") + next_depth))  # 0.5 * (x_i + x_(i+1)) gives approximate right edge
            .fill_null(1.5 * pl.col("depth") - 0.5 * previous_depth)  # assuming symmetry (zones are tiny anyway)
            .alias("depth_input")
        )

        return df

    def read_continuum(self, drop_failed: bool = True) -> pl.DataFrame:
        """
        Reads all models' .con files back from the HDF5 catalogue. Returns:

        `pl.DataFrame`, sorted by contiguous run ID, where each row contains the currently-read continuum values
        for the given run ID. Values come from the final iteration.

        Access the `continuum_wavelengths` property for the corresponding wavelengths.

        Combined is divided by wavelength.
        """
        con_group: h5py.Group = self._file["cloudy_data"]["con"]
        n_rays = len(self)
        n_wavelengths = con_group["combined"].shape[1]

        combined = pl.Series("combined", con_group["combined"][:], dtype=pl.Array(pl.Float64, n_wavelengths))
        df = (
            pl.DataFrame({"run_id": np.arange(n_rays), "success": self.successes})
            .with_columns(combined)
            .filter(
                pl.col("success") | pl.lit(not drop_failed)
            )  # unioning success with the literal bool input drops failed runs
        )

        return df

    def read_lines(self, drop_failed: bool = True) -> pl.DataFrame:
        """
        Reads all models' .lin files back from the HDF5 catalogue. Returns:

        `pl.DataFrame`, sorted by contiguous run ID, where each row contains a model's line ratios as
        columns.

        Units: normalised intensity relative to H beta.
        """
        lin_group: h5py.Group = self._file["cloudy_data"]["lin"]
        labels = lin_group["labels"].asstr()[:]
        values = lin_group["values"][:]

        # NOTE: if the same line appears twice in the file this does handle it but maybe worth deduping at some point
        df = pl.DataFrame(values, schema=labels.tolist(), orient="row").with_columns(
            pl.Series("success", self.successes)
        )

        # in-place insertion of run IDs
        df.insert_column(0, pl.Series("run_id", np.arange(len(self))))

        # then filter the bad apples
        df = df.filter(pl.col("success") | pl.lit(not drop_failed))

        return df


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
