"""

Functionality for writing and parsing .lin files.

"""

from pathlib import Path

import polars as pl


def read_line_list(line_path: Path) -> dict[str, float]:
    """
    Reads an output .lin file, returning a dict containing the intensities keyed by species name.
    """
    table = pl.read_csv(line_path, separator="\t")
    table = table.rename({name: name.strip() for name in table.columns})  # strip 4 character minimum

    results = table.row(-1, named=True)  # return as dict
    results.pop("#lineslist")

    return results
