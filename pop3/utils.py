"""

Minor utilities.

"""

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yt
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from yaml import safe_load

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

FILEPATHS = frozenset(
    {
        "ledger_path",
        "sp_dir",
        "profile_plot_dir",
        "proj_dir",
        "film_dir",
        "frame_dir",
        "data_dir",
        "mpl_style",
        "snap_dir",
        "plot_dir",
    }
)

CLOUDY_FILEPATHS = frozenset(
    {
        "linelist_path",
        "yield_path",
        "sed_path",
        "snap_dir",
        "cloudy_src_dir",
        "output_dir",
        "plot_dir",
        "sed_dir",
        "larkin_40_path",
        "larkin_260_path",
        "schaerer_40_path",
    }
)


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

        with open(config_path) as f:
            raw: dict = safe_load(f)

        # make the star list a tuple so it is also immutable
        raw["stars"]["process_stars"] = tuple(raw["stars"].get("process_stars") or ())

        config = Pop3Config(
            cloudy=CloudyConfig.from_dict(raw["cloudy"]),
            plots=PlotConfig(**_convert_paths(raw["plots"], FILEPATHS)),
            stars=StarConfig(**_convert_paths(raw["stars"], FILEPATHS)),
            spheres=SphereConfig(**_convert_paths(raw["spheres"], FILEPATHS)),
        )

        return config


@dataclass(frozen=True, slots=True)
class CloudyConfig:
    """
    Contains the three Cloudy-related configs.

    - physics: `CloudyPhysicsConfig`
    - target: `CloudyStarConfig`
    - misc: `CloudyMiscConfig`
    """

    physics: CloudyPhysicsConfig
    star: CloudyStarConfig
    misc: CloudyMiscConfig

    @classmethod
    def from_dict(cls, raw: dict[str, dict[str, Any]]) -> CloudyConfig:
        """
        Converts the YAML nested dictionary into a `CloudyConfig` object.
        """
        # targets in the YAML file is a dict keyed by star name of a list of ages
        yaml_targets: dict[str, list] = raw["sim"]["targets"]
        # make the star list a tuple so it is also immutable
        targets = {star: tuple(ages) for star, ages in yaml_targets.items()}

        config = cls(
            physics=CloudyPhysicsConfig(**_convert_paths(raw["physics"], CLOUDY_FILEPATHS)),
            star=CloudyStarConfig(**(raw["sim"] | {"targets": targets})),
            misc=CloudyMiscConfig(**_convert_paths(raw["misc"], CLOUDY_FILEPATHS)),
        )

        return config

    def update_physics_params(self, **changes: Any) -> CloudyConfig:
        """
        Updates the physics section (`CloudyPhysicsConfig`) with any specified keywork arguments.
        Returns a new `CloudyConfig`.
        """
        return replace(self, physics=replace(self.physics, **changes))


@dataclass(frozen=True, slots=True)
class CloudyPhysicsConfig:
    """
    Physics-related parameters for a specific `CloudyModel` run.

    Default notes:


    - use_cosmic_rays = True, for numerical stability
    - efrac_threshold = 0.01, per Fergus/Nakajima
    - density_tolerance = 1.0e-5, Hazy default
    - n_iterations = None, one calculation
    """

    linelist_path: Path
    yield_path: Path
    sed_path: Path
    sed_type: str
    use_cosmic_rays: bool = True
    efrac_threshold: float = 0.01  # fergus' choice
    density_tolerance: float = 1.0e-5
    n_iterations: int | None = None
    homogeneous: bool = False
    iterate_to_convergence: bool = False


@dataclass(frozen=True, slots=True)
class CloudyStarConfig:
    """
    Parameters relating to the star ray extraction.

    Default notes:

    - ray_length_kpc = 1.0, for quick checks
    - n_side = 8 -> 768 rays, same as Enzo
    """

    targets: dict[str, tuple[float, ...]]
    ray_length_kpc: float = 1.0
    n_side: int = 8  # enzo's choice
    use_first_star: bool = False  # lazy option


@dataclass(frozen=True, slots=True)
class CloudyMiscConfig:
    """
    Directories, filepaths, what to store, etc. for `CloudyModel`.
    """

    snap_dir: Path
    cloudy_src_dir: Path
    output_dir: Path
    plot_dir: Path
    timeout_s: float
    sed_dir: Path
    larkin_260_path: Path
    larkin_40_path: Path
    schaerer_40_path: Path
    cleanup_run_dirs: bool = False
    store_overview: bool = True
    store_continuum: bool = True
    store_lines: bool = True


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
    framerate: int = 60
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


def parse_args() -> argparse.Namespace:
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


def instantiate_console() -> Console:
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


def my_progress(console: Console, spinner: str = "earth") -> Progress:
    """
    Modified rich progress bar.
    """
    return Progress(
        SpinnerColumn(spinner_name=spinner),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    )


def _convert_paths(section: dict[str, Any], path_keys: frozenset[str]) -> dict[str, Any]:
    """
    Helper for iterating over a YAML file section and, if am key is in `path_keys`, convert it
    to a `Path` object with the user shortcut expanded so it represents an absolute path.
    """
    result = {
        key: Path(value).expanduser() if key in path_keys and value is not None else value
        for key, value in section.items()
    }

    return result
