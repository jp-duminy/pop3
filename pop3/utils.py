"""

Minor utilities.

"""

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from yaml import safe_load

import yt
from rich.console import Console
from rich.logging import RichHandler

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

FILEPATHS = frozenset({
    "cloudy_path", 
    "ledger_path", 
    "sp_dir", 
    "profile_plot_dir",
    "proj_dir",
    "film_dir",
    "frame_dir",
    "data_dir",
    "mpl_style", 
})


def load_config(args: argparse.Namespace) -> Pop3Config:
    """
    Loads the config from args.config, and applies any requested overrides.
    """
    config = Pop3Config.from_yaml(args.config)

    if args.stars is not None:  # use replace to override frozen dataclass
        config = replace(config, stars=replace(config.stars, process_stars=tuple(args.stars)))

    return config

@dataclass(frozen=True, slots=True)
class Pop3Config:
    """
    Contains nested dataclasses each with a set of parameters:

    - CloudyConfig
    - PlotConfig
    - StarConfig
    - SphereConfig
    """
    cloudy: CloudyConfig
    plots: PlotConfig
    stars: StarConfig
    spheres: SphereConfig

    @classmethod
    def from_yaml(cls, config_path: Path) -> Pop3Config:

        with open(config_path, "r") as f:
            raw: dict = safe_load(f)

        # make the star list a tuple so it is also immutable
        raw["stars"]["process_stars"] = tuple(raw["stars"].get("process_stars") or ())

        # turn the strs in the yaml into Path objects
        for heading, entries in raw.items():  
            for key in FILEPATHS:
                entries: dict
                if entries.get(key) is not None:
                    entries[key] = Path(entries[key]).expanduser()

        config = Pop3Config(
            cloudy=CloudyConfig(**raw["cloudy"]),
            plots=PlotConfig(**raw["plots"]),
            stars=StarConfig(**raw["stars"]),
            spheres=SphereConfig(**raw["spheres"]),
        )

        return config


@dataclass(frozen=True, slots=True)
class CloudyConfig:
    """
    Contains Cloudy-specific parameters.
    """
    cloudy_path: Path

@dataclass(frozen=True, slots=True)
class PlotConfig:
    """
    Contains all plot and film params.
    """
    profile_plot_dir: Path
    film_dir: Path
    proj_dir: Path
    frame_dir: Path
    axis: str = "x"
    framerate: int  = 60
    frame_width: float = 1.5  # kpc
    post_death: float = 125.0  # Myr


@dataclass(frozen=True, slots=True)
class StarConfig:
    """
    Contains star-specific parameters.
    """
    ledger_path: Path
    process_stars: tuple[str, ...]
    max_metallicity: float = 1.0e-4


@dataclass(frozen=True, slots=True)
class SphereConfig:
    """
    Contains parameters for instantiating spheres and post-processing them.
    """
    sp_dir: Path
    data_dir: Path
    sp_radius: float = 1.0
    sp_radius_unit: str = "kpc"


def parse_args() -> argparse.ArgumentParser:
    """
    Parses common arguments which override their equivalent values in the config.

    - config: config path
    - stars: unseparated list of stars e.g. POP3-240, POP3-379
    - force: overwrites existing outputs
    """
    parser = argparse.ArgumentParser(        
        prog="POP3",
        description="Routine commands for interfacing with the Pop2Prime simulation.",
        suggest_on_error=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML file.",
    )
    parser.add_argument("-s", "--stars", nargs="+", default=None, help="Stars to process (overrides config).")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrites existing outputs.")

    return parser.parse_args()


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

