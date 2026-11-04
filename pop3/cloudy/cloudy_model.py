"""

The CloudyModel class. Each class corresponds to one independent run of Cloudy.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .commands import CloudyCommand, SaveCmd

import subprocess
from dataclasses import dataclass

import numpy as np

from ..utils import CloudyConfig
from .commands import create_input_file
from .parsers import (
    read_continuum,
    read_line_list,
    read_out_log,
    read_overview,
)


@dataclass(frozen=True, slots=True)
class CloudyResult:
    """
    Container for the results of a single CloudyModel run.

    - run_id: the CloudyModel's unique run_ID
    - status: per the .out file, whether the simulation completed
    - continuum: the parsed .con file
    - lines: the parsed .lin file (dict keyed by element name with line intensity)
    - overview: the parsed .ovr file
    """

    run_id: int
    status: str
    continuum: dict[str, np.ndarray] | None = None
    lines: dict[str, float] | None = None
    overview: dict[str, np.ndarray] | None = None

    @classmethod
    def err(cls, run_id: int) -> CloudyResult:
        """
        Returns an empty CloudyResult with the error status.
        """
        result = cls(run_id=run_id, status="error")

        return result


class CloudyModel:
    """
    The `CloudyModel` class handles input generation, Cloudy execution, and output parsing.
    """

    def __init__(
        self,
        config: CloudyConfig,
        in_commands: list[CloudyCommand],
        save_command: SaveCmd,
    ) -> None:

        # directories and files
        self.cloudy_dir = config.cloudy_dir
        self.cloudy_exec = config.cloudy_dir / "source" / "cloudy.exe"
        self.datafile_dir = config.datafile_dir
        self.output_dir = config.output_dir

        # commands and model identifier
        self.in_cmds = in_commands
        self.save_cmd = save_command
        self.run_id = save_command.run_id

        # runtime parameters
        self.timeout_s = config.timeout_s

    def write_input(self) -> None:
        """
        Writes a .in file for the model.
        """
        run_dir = self.output_dir / str(self.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        create_input_file(commands=[*self.in_cmds, self.save_cmd], input_filepath=run_dir / f"{self.run_id}.in")

    def execute(self) -> subprocess.CompletedProcess[str]:
        """
        Executes a run of Cloudy via subprocess by reading the run ID data from the output directory.
        """
        return subprocess.run(
            [str(self.cloudy_exec), "-r", str(self.run_id)],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )

    def package_data(self) -> CloudyResult:
        """
        Parses Cloudy's output files into a neat CloudyResult object.
        """
        run_succeeded = read_out_log(out_path=self.run_dir / f"{self.run_id}.out")

        if not run_succeeded:
            return CloudyResult.err(run_id=self.run_id)

        cont = read_continuum(continuum_path=self.run_dir / f"{self.run_id}.con") if self.save_cmd.continuum else None
        lin = read_line_list(line_path=self.run_dir / f"{self.run_id}.lin") if self.save_cmd.linelist else None
        ovr = read_overview(overview_path=self.run_dir / f"{self.run_id}.ovr") if self.save_cmd.overview else None

        result = CloudyResult(
            run_id=self.run_id,
            status="success",
            continuum=cont,
            lines=lin,
            overview=ovr,
        )

        return result
