"""

Handles PopIII abundance and SED data.

"""

from pathlib import Path

import polars as pl
from rich import Console


def create_abundance_dict(abundance_file_path: Path, console: Console) -> dict[str, float]:
    """
    Uses the tabulated PopIII ejecta file to create an abundance dict keyed by Cloudy
    element name and containing the f_x/A_x values.
    """
    df = pl.read_csv(abundance_file_path, separator="\t")
    abundances = dict(df.select("element_name", "mass_fraction_over_mass_number").iter_rows())
    console.log(abundances)  # pretty prints

    return abundances
