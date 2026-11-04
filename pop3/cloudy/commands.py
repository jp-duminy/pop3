"""

Dataclasses which represent Cloudy commands and can write directly to a .in file.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HAZY_FLOOR = 1e-35  # min distance which Cloudy accepts


class CloudyCommand(ABC):
    """
    Abstract base class for Cloudy commands: stipulates overwriting the str representation
    of the dataclass and should always be inherited.
    """

    @abstractmethod
    def __str__(self) -> str: ...


def create_input_file(commands: list[CloudyCommand], input_filepath: Path) -> str:
    """
    Creates and writes a .in file for Cloudy using a specified `CloudyCommand` sequence.
    """
    cmd_text = "\n".join(str(cmd) for cmd in commands)

    # NOTE: this is a rather important check to ensure Cloudy is actually going to read the file
    if any(not line.strip() for line in cmd_text.splitlines()):
        raise ValueError("Blank lines in commands will prevent Cloudy from reading full .in file.")

    input_filepath.write_text(cmd_text + "\n", encoding="ascii")  # add a blank line so cloudy breaks

    return cmd_text


@dataclass(frozen=True, slots=True)
class DlawTable(CloudyCommand):
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

        rows = "\n".join(
            f"continue {depth:.6f} {density:.6f}"  # 6 sig figs should be ok
            for depth, density in zip(
                self.log_path_lengths, self.log_densities, strict=True
            )  # also ensures same length
        )
        cmd = f"dlaw table depth\n{rows}\nend of dlaw"

        return cmd


@dataclass(frozen=True, slots=True)
class TlawTable(CloudyCommand):
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

        rows = "\n".join(
            f"continue {depth:.6f} {temp:.6f}"  # 6 sig figs should be ok
            for depth, temp in zip(
                self.log_path_lengths, self.log_temperatures, strict=True
            )  # also ensures same length
        )
        cmd = f"tlaw table depth\n{rows}\nend of tlaw"

        return cmd


@dataclass(frozen=True, slots=True)
class Background(CloudyCommand):
    """
    Controls the two basic background commands. Accepts:

    - redshift: for CMB
    - use_cosmic_rays: enables the cosmic rays flag (Cloudy tells me I'm making a big mistake without this...)
    """

    redshift: float
    use_cosmic_rays: bool = True

    def __str__(self) -> str:

        lines = [f"CMB redshift {self.redshift}"]

        if self.use_cosmic_rays:
            lines.append("cosmic rays background")

        cmd = "\n".join(lines)

        return cmd


@dataclass(frozen=True, slots=True)
class Blackbody(CloudyCommand):
    """
    Controls the `blackbody` SED command. Accepts:

    - log_temperature
    """

    log_temperature: float

    def __str__(self) -> str:

        return f"blackbody log {self.log_temperature:.6f}"


@dataclass(frozen=True, slots=True)
class TableSED(CloudyCommand):
    """
    Controls the specified `table SED` command. Accepts:

    - sed_path
    """

    sed_path: Path

    def __str__(self) -> str:

        return f'table SED "{self.sed_path}"'


@dataclass(frozen=True, slots=True)
class RadSource(CloudyCommand):
    """
    Controls the input SED, accepting either a blackbody or table.
    """

    shape: Blackbody | TableSED
    log_q_hydrogen: float

    def __str__(self) -> str:

        return "\n".join([str(self.shape), f"q(h) {self.log_q_hydrogen:.6f}"])


@dataclass(frozen=True, slots=True)
class StopCondition(CloudyCommand):
    """
    Controls the electron fraction stopping criterion `efrac` and the fallback `depth` command.
    """

    electron_frac: float
    log_max_depth: float | None = None

    def __post_init__(self) -> None:

        if self.electron_frac <= 0.0:
            raise ValueError("electron_frac must be a fraction above 0.0")

    def __str__(self) -> str:

        lines = [f"stop efrac {self.electron_frac:.6f}"]
        if self.log_max_depth is not None:
            lines.append(f"stop depth {self.log_max_depth:.6f}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SphereGeometry(CloudyCommand):
    """
    Spherical (closed) geometry assuming a point source, like our PopIII stars are.
    """

    log_inner_radius: float  # let's say first cell from the star

    def __str__(self) -> str:
        return "\n".join(
            [f"radius {self.log_inner_radius:.6f}", "sphere expanding"]
        )  # TODO: investigate spherical geometry


@dataclass(frozen=True, slots=True)
class ElementTable(CloudyCommand):
    """
    Controls the element table command.

    `element <name> table depth`
    """

    element_name: str  # use .to_uppercase() per Cloudy's own .abn tables
    log_path_lengths: np.ndarray
    log_abundances: np.ndarray

    def __post_init__(self) -> None:

        if self.log_abundances.shape != self.log_path_lengths.shape:
            raise ValueError(
                f"Density shape {self.log_abundances.shape} is misaligned with path length shape {self.log_path_lengths.shape}"
            )

    def __str__(self) -> str:

        rows = "\n".join(
            f"{depth:.6f} {abundance:.6f}"
            for depth, abundance in zip(self.log_path_lengths, self.log_abundances, strict=True)
        )
        cmd = f"element {self.element_name} table depth\n{rows}\nend of table"

        return cmd


@dataclass(frozen=True, slots=True)
class Abundances(CloudyCommand):
    """
    Controls all `abundances` related work in Cloudy, including the primordial helium fraction (others relative to hydrogen).
    """

    elements: list[ElementTable]
    primordial_helium: float = 0.08232  # pilfered from Cloudy's primordial.abn

    def __str__(self) -> str:

        hydrogen_cmd = "element hydrogen abundance 0 log"  # hydrogen is just for normalisation
        helium_cmd = f"element helium abundance {np.log10(self.primordial_helium):.6f} log"

        cmd = "\n".join([hydrogen_cmd, helium_cmd, *(str(element) for element in self.elements)])

        return cmd


@dataclass(frozen=True, slots=True)
class SaveCmd(CloudyCommand):
    """
    Controls which data to save. For linelist, enabling this means passing the data file `Path` object.
    """

    run_id: int
    overview: bool = True
    continuum: bool = True
    linelist: Path | None = None

    def __str__(self) -> str:

        commands: list[str] = []

        if self.overview:
            commands.append(f'save overview last "{self.run_id}.ovr"')
        if self.continuum:
            commands.append(f'save continuum last units Angstroms "{self.run_id}.con"')
        if self.linelist is not None:
            commands.append("print line vacuum")
            commands.append(f'save line list last "{self.run_id}.lin" "{self.linelist.name}"')

        return "\n".join(commands)
