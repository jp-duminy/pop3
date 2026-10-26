"""

Dataclasses which represent Cloudy commands and can write directly to a .in file.

"""

from dataclasses import dataclass

import numpy as np

HAZY_FLOOR = 1e-30  # min distance which Cloudy accepts


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
