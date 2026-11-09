"""

Functionality for creating a HDF5 file to store the outputs of a single Cloudy run.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils import CloudyConfig


import subprocess
from dataclasses import asdict
from pathlib import Path

import h5py

# so these aren't always hardcoded
GROUP_NAMES = {
    "ovr": "overview",
    "lin": "lines",
    "con": "continuum",
}


def create_output_file(output_dir: Path, config: CloudyConfig, force: bool = True) -> Path:
    """
    Creates the output HDF5 file at `output_dir` / `{star}_{age}`.
    """
    output_path = output_dir / f"{config.star}_{config.star_age_myr!s}Myr.h5"

    if output_path.exists():
        if force:
            output_path.unlink()
        else:
            raise ValueError(f"Warning: {output_path} already exists!!")

    commit_hash = _get_git_commit()

    with h5py.File(output_path, "w") as out:
        metadata = out.create_group("metadata")

        for param, value in asdict(config).items():
            if isinstance(value, Path):  # Path objects can't go in directly
                value = str(value)
            elif value is None:  # nor can these
                value = "None"
            metadata.attrs[param] = value

        if commit_hash is not None:
            metadata.attrs["git_commit"] = commit_hash

        cloudy = out.create_group("cloudy_data")

        if config.store_overview:
            cloudy.create_group(GROUP_NAMES["ovr"])
        if config.store_continuum:
            cloudy.create_group(GROUP_NAMES["con"])
        if config.store_lines:
            cloudy.create_group(GROUP_NAMES["lin"])

    return output_path


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
