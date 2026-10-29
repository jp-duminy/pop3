"""

Dataclasses which represent Cloudy commands and can write directly to a .in file.

"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

HAZY_FLOOR = 1e-35  # min distance which Cloudy accepts


@dataclass(frozen=True, slots=True)
class DlawTable:
    """
    Accepts:

    - log_densities
    - log_path_lengths

    And controls the `dlaw table` Cloudy .in file command, automatically generating this. Not
    intended to be human readable since the output could be very long. Uses the `depth` command,
    not `radius`.
    """

    log_densities: np.ndarray
    log_path_lengths: np.ndarray

    def __post_init__(self) -> None:

        if self.log_densities.shape != self.log_path_lengths.shape:
            raise ValueError(
                f"Density shape {self.log_densities.shape} is misaligned with path length shape {self.log_path_lengths.shape}"
            )

    def __str__(self) -> str:
        """
        Turns the str(DlawTable) into its representation ready for a .in file.
        """
        rows = "\n".join(
            f"continue {depth:.6f} {density:.6f}"  # 6 sig figs should be ok
            for depth, density in zip(
                self.log_path_lengths, self.log_densities, strict=True
            )  # also ensures same length
        )
        cmd = f"dlaw table depth\n{rows}\nend of dlaw"

        return cmd


@dataclass(frozen=True, slots=True)
class TlawTable:
    """
    Accepts:

    - log_temperatures
    - log_path_lengths

    And controls the `tlaw table` Cloudy .in file command, automatically generating this. Not
    intended to be human readable since the output could be very long. Uses the `depth` command,
    not `radius`.
    """

    log_temperatures: np.ndarray
    log_path_lengths: np.ndarray

    def __post_init__(self) -> None:

        if self.log_temperatures.shape != self.log_path_lengths.shape:
            raise ValueError(
                f"Temperature shape {self.log_temperatures.shape} is misaligned with path length shape {self.log_path_lengths.shape}"
            )

    def __str__(self) -> str:
        """
        Turns the str(TlawTable) into its representation ready for a .in file.
        """
        rows = "\n".join(
            f"continue {depth:.6f} {temp:.6f}"  # 6 sig figs should be ok
            for depth, temp in zip(
                self.log_path_lengths, self.log_temperatures, strict=True
            )  # also ensures same length
        )
        cmd = f"tlaw table depth\n{rows}\nend of tlaw"

        return cmd


@dataclass(frozen=True, slots=True)
class Background:
    """
    Controls the two basic background commands. Accepts:

    - redshift: for CMB
    - use_cosmic_rays: enables the cosmic rays flag (Cloudy tells me I'm making a big mistake without this...)
    """

    redshift: float
    use_cosmic_rays: bool = True

    def __str__(self) -> str:
        """
        Turns the str(Background) into its representation ready for a .in file.
        """
        lines = [f"CMB redshift {self.redshift}"]

        if self.use_cosmic_rays:
            lines.append("cosmic rays background")

        cmd = "\n".join(lines)

        return cmd


@dataclass(frozen=True, slots=True)
class Blackbody:
    """
    Controls the `blackbody` SED command. Accepts:

    - log_temperature
    """

    log_temperature: float

    def __str__(self) -> str:
        """
        Turns the str(Blackbody) into its representation ready for a .in file.
        """
        return f"blackbody log {self.log_temperature:.6f}"


@dataclass(frozen=True, slots=True)
class TableSED:
    """
    Controls the specified `table SED` command. Accepts:

    - sed_path
    """

    sed_path: Path

    def __str__(self) -> str:
        """
        Turns the str(TableSED) into its representation ready for a .in file.
        """
        return f'table SED "{self.sed_path}"'


@dataclass(frozen=True, slots=True)
class RadSource:
    """
    Controls the input SED, accepting either a blackbody or table.
    """

    shape: Blackbody | TableSED
    log_q_hydrogen: float

    def __str__(self) -> str:
        """
        Turns the str(RadSource) into its representation ready for a .in file.
        """
        return "\n".join([str(self.shape), f"q(h) {self.log_q_hydrogen:.6f}"])


@dataclass(frozen=True, slots=True)
class StopElectronFrac:
    """
    Controls the electron fraction stopping criterion `efrac`.
    """

    electron_frac: float

    def __post_init__(self) -> None:

        if self.electron_frac <= 0.0:
            raise ValueError("electron_frac must be a fraction above 0.0")

    def __str__(self) -> str:
        """
        Turns str(StopElectronFrac) into its representation ready for a .in file.
        """
        return f"stop efrac {self.electron_frac:.6f}"


@dataclass(frozen=True, slots=True)
class SphereGeometry:
    """
    Spherical (closed) geometry assuming a point source, like our PopIII stars are.
    """

    log_inner_radius: float  # let's say first cell from the star

    def __str__(self) -> str:
        """
        Turns str(SphereGeometry) into its representation ready for a .in file.
        """
        return "\n".join(
            [f"radius {self.log_inner_radius:.6f}", "sphere expanding"]
        )  # TODO: investigate spherical geometry
