"""

Minor utilities.

"""

import argparse
import logging
from typing import Generator
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter

import yt
from rich.console import Console
from rich.logging import RichHandler

# top-level data directory
DATA_DIR = Path("/cephfs2/brs/pop2-prime/cc_512_no_dust_continue")
DEFAULT_LEDGER_DIR = Path.home() / "mphys" / "data" / "pop3_ledger.parquet"

def common_parser() -> argparse.ArgumentParser:
    """
    Arguments common to every parser. Wrap this function call in [] and pass to parents= arg on subparsers.
    """
    parser = argparse.ArgumentParser(add_help=False)  # need to set add_help=False for parent parsers
    parser.add_argument(
        "-l", 
        "--ledger", 
        type=Path, 
        default=DEFAULT_LEDGER_DIR, 
        help="Path to ledger parquet file."
    )
    parser.add_argument("-o", "--outdir", type=Path, default=Path("."), help="Path to output directory.")
    parser.add_argument("-s", "--stars", nargs="+", help="Stars to process (list labels).")

    return parser


def instantiate_logger() -> Console:
    """
    Instantiates the terminal UI with parallel-aware rich. Returns a `Console` object.
    """
    yt.set_log_level("error")

    try:  # route output to tty so progress bars and live output appears 
        console = Console(file=open("/dev/tty", "w"), force_terminal=True) 
    except OSError:  # in case that file doesn't exist
        console = Console(force_terminal=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False)],
    )

    return console


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """
    Context manager around perf_counter().
    """
    t0 = perf_counter()
    yield
    elapsed = perf_counter() - t0
    print(f"{label} completed in {elapsed:.1f}s.")