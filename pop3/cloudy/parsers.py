"""

Functionality for writing and parsing .lin files.

"""

from pathlib import Path

import numpy as np
import polars as pl

from ..utils import Pop3Config


def read_line_list(line_path: Path) -> dict[str, float]:
    """
    Reads an output .lin file, returning a dict containing the intensities keyed by species name.
    """
    table = pl.read_csv(line_path, separator="\t")
    table = table.rename({name: name.strip() for name in table.columns})  # strip 4 character minimum

    results = table.row(-1, named=True)  # return as dict
    results.pop("#lineslist")

    return results


def read_continuum(continuum_path: Path) -> dict[str, np.ndarray]:
    """
    Reads an output .con file, returning a dict of wavelengths and the combined spectrum.
    """
    table = pl.read_csv(continuum_path, separator="\t", has_header=False, skip_rows=1, columns=[0, 4])
    wavelength_angstrom = table.to_series(0).to_numpy()
    combined = table.to_series(1).to_numpy() / wavelength_angstrom

    results = {
        "wavelength_angstrom": wavelength_angstrom,
        "combined": combined,
    }

    return results


def read_overview(overview_path: Path) -> dict[str, np.ndarray]:
    """
    Reads an output .ovr file, returning a dict of the result columns.
    """
    table = pl.read_csv(overview_path, separator="\t")
    table = table.rename({name: name.strip().lstrip("#") for name in table.columns})
    results = {name: table[name].to_numpy() for name in table.columns}

    return results


def read_out_log(out_path: Path) -> bool:
    """
    Reads an output .out file, returning whether Cloudy successfully ran.
    """
    last_line = out_path.read_text().rstrip().splitlines()[-1]
    run_complete = "Cloudy exited OK" in last_line

    return run_complete


def read_nomoto_table(nomoto_path: Path, config: Pop3Config) -> dict[str, float]:
    """
    Reads the Nomoto+2006 table into a dict of floats for each element corresponding to the second RHS
    term in my derivation.
    """
    metals_table = pl.read_csv(nomoto_path, separator="\t")
    element_names: list[str] = metals_table["element_name"].to_list()
    log_f_over_a = np.log10(metals_table["mass_fraction_over_mass_number"].to_numpy() / config.cloudy.primordial_XH)

    results: dict[str, float] = {name: val for name, val in zip(element_names, log_f_over_a, strict=True)}

    return results
