"""

The CloudyModel class. Each class corresponds to one independent run of Cloudy.

"""

import subprocess

from ..utils import CloudyConfig


class CloudyModel:
    """
    The `CloudyModel` class handles input generation, Cloudy execution, and output parsing.
    """

    def __init__(self, config: CloudyConfig) -> None:

        self.cloudy_dir = config.cloudy_dir
        self.cloudy_exec = config.cloudy_dir / "source" / "cloudy.exe"
        self.datafile_dir = config.datafile_dir
        self.output_dir = config.output_dir

    def execute(self, run_ID: str, timeout_s: float = 3600.0) -> subprocess.CompletedProcess[str]:
        """
        Executes a run of Cloudy via subprocess by reading the run ID data from the output directory.
        """
        run_dir = self.output_dir / run_ID
        input_path = run_dir / f"{run_ID}.in"

        if not input_path.is_file():
            raise FileNotFoundError(f"Typo in {input_path}?")

        return subprocess.run(
            [str(self.cloudy_exec), "-r", run_ID],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
