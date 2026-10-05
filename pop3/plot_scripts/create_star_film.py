"""

Stitches together projection plots into a film for stars.

"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.text import Text
    from rich.console import Console

from pathlib import Path
import subprocess
import argparse

import yt
import numpy as np
import polars as pl
from scipy.special import expit
from rich.progress import Progress
from mpi4py.futures import MPIPoolExecutor
from mpi4py.MPI import COMM_WORLD

import matplotlib
matplotlib.use("Agg")  # this has to be here (matplotlib is really user-friendly)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import yt.visualization.color_maps  # noqa: F401

from ..utils import instantiate_logger, common_parser
from ..sim.pop3_ledger import star_lifetime_summary, select_stars

MPL_STYLE = Path.home() / "mnras.mplstyle"

FIELD_STYLES: dict[str, tuple[str, str]] = {
    "density": ("algae", r"$\rho_{b}$ [g cm$^{-3}$]"),
    "temperature": ("gist_heat", "T [K]"),
    "metallicity3": ("kamae", r"Z [Z$_{\odot}$]"),
}

METALLICITY_FLOOR = 1e-8

def parse_args() -> argparse.Namespace:
    """
    Parses the command-line arguments; returns the corresponding Namespace object.
    """
    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        prog="star-film",
        description="Routine for converting star projection plots into film.",
        suggest_on_error=True,
    )
    parser.add_argument(
        "-p",
        "--projdir",
        type=Path,
        required=True,
        help="Path to projection plots."
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=str,
        default="x",
        help="Axis used for projections."
    )
    parser.add_argument(
        "-f",
        "--framerate",
        type=int,
        default=60,
        help="Framerate for the films."
    )

    return parser.parse_args()


def make_star_film(
    console: Console,
    executor: MPIPoolExecutor,
    star: dict[str, Any],  # comes from dataframe iters
    projections_root: Path,
    plots_root: Path,
    axis: str,
    framerate: int,
) -> None:
    label: str = star["label"]

    paths, snapshot_times_myr, field_ranges, width_kpc = collect_projections(
        console=console, executor=executor, image_dir=projections_root / label, label=label, axis=axis,
    )
    field_ranges["metallicity3"] = (METALLICITY_FLOOR, field_ranges["metallicity3"][1])

    transition_time_myr = star["first_dead_myr"]

    frame_times_myr = build_frame_times(
        snapshot_times_myr=snapshot_times_myr,
        transition_time=transition_time_myr,
        cutoff_myr=snapshot_times_myr[-1],
        dt_min=0.01,  # NOTE: slightly increased from Britton's (0.15 is a bit patchy)
        dt_max=0.25, 
        transition_rate=1.0,  # increased to make faster
    )

    frame_dir = plots_root / "frames" / label
    frame_dir.mkdir(parents=True, exist_ok=True)
    [f.unlink() for f in frame_dir.glob("*") if f.is_file()]

    render_frames(
        executor=executor,
        projection_paths=paths,
        snapshot_times_myr=snapshot_times_myr,
        frame_times_myr=frame_times_myr,
        field_ranges=field_ranges,
        creation_time_myr=star["creation_times_myr"],
        width_kpc=width_kpc,
        frame_dir=frame_dir,
    )

    assemble_film(
        frame_dir=frame_dir,
        output_path=plots_root / f"{label}_{axis}.mp4",
        framerate=framerate,
    )

def build_frame_times(
    snapshot_times_myr: np.ndarray,
    transition_time: float, 
    dt_min: float, 
    dt_max: float, 
    transition_rate: float,
    cutoff_myr: float,
) -> np.ndarray:
    """
    Generates a time for each frame according to a sigmoid.
    """
    frame_times_myr = []

    current_time_myr = snapshot_times_myr[0]
    while current_time_myr <= cutoff_myr:
        frame_times_myr.append(current_time_myr)
        # NOTE: Britton's code increases the timestep after transition_time but I flip the sign so we get more granular around the supernova
        current_time_myr += dt_min + (dt_max - dt_min) * expit(transition_rate * (current_time_myr - transition_time))

    return np.array(frame_times_myr)


def collect_projections(
    console: Console,
    executor: MPIPoolExecutor,
    image_dir: Path, 
    label: str, 
    axis: str,
) -> tuple[list[Path], np.ndarray, dict[str, tuple[float, float]], float]:
    """
    Collects the projection plots from an image directory. Returns:

    - files: a list of projection paths
    - times_myr: their corresponding times in Myr
    - field_ranges: dict keyed by field with min/max of that field across plots
    - width_kpc: the kpc width of the frame
    """
    field_names = ("density", "temperature", "metallicity3")
    projection_paths = sorted(image_dir.glob(f"{label}_*_{axis}.h5"))
    assert projection_paths, f"No projections found in {image_dir}"

    with Progress(console=console) as progress:
        task_id = progress.add_task("Collecting projection plots", total=len(projection_paths))
        results = []
        for result in executor.starmap(_scan_projection, ((path, field_names) for path in projection_paths)):
            progress.advance(task_id) 
            if result is not None:
                results.append(result)

    snapshot_times_myr = np.array([result[0] for result in results])
    field_ranges = {  # logic from previous function put into this (somewhat ugly) form
        field_name: (
            min(result[1][field_name] for result in results),  # reduce over joblib list now
            max(result[2][field_name] for result in results),
        )
        for field_name in field_names
    }

    # grab the width in kpc of the plot (set as an extra attr in other function)
    quick_ds = yt.load(projection_paths[0])
    width_kpc = float(quick_ds.parameters["width_kpc"])  # should always be the same

    return projection_paths, snapshot_times_myr, field_ranges, width_kpc


def _scan_projection(projection_path: Path, field_names: tuple[str]) -> tuple[float, dict[str, float], dict[str, float]]:
    """
    For projection_path, reads:
     
    - time_myr
    - field_mins: a dict for each field containing the min (NaN masked)
    """
    yt.set_log_level("error")  # prevent lots of verbose output
    projection_ds = yt.load(projection_path)

    time_myr = float(projection_ds.current_time.to("Myr"))
    field_mins: dict[str, float] = dict.fromkeys(field_names, np.inf)
    field_maxs: dict[str, float] = dict.fromkeys(field_names, -np.inf)

    for field_name in field_names:
        field_values = projection_ds.data["data", field_name].d
        field_mins[field_name] = float(np.min(field_values, where=field_values > 0, initial=np.inf))
        field_maxs[field_name] = float(np.nanmax(field_values))

    return time_myr, field_mins, field_maxs


def _load_log_fields(projection_path: Path, field_ranges: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """
    Loads fields in logspace with field ranges applied. Returns:

    - log_fields: dict keyed by field name with the log values (nan masked and floored)
    """
    yt.set_log_level("error")
    ds = yt.load(projection_path)
    log_fields: dict[str, np.ndarray] = {}

    for field in field_ranges:

        values = np.nan_to_num(ds.data["data", field].d, nan=field_ranges[field][0])  # clip NaNs to min
        np.clip(values, a_min=field_ranges[field][0], a_max=None, out=values)  # clip vals < min to min
        log_fields[field] = np.log10(values)
        
    return log_fields


def _build_figure(
    first_logs: dict[str, np.ndarray],
    field_ranges: dict[str, tuple[float, float]],
    width_kpc: float,
    scale_bar_kpc: float,
) -> tuple[Figure, dict[str, AxesImage], Text]:
    """
    Builds a triangular figure where metallicity (the golden goose) stays at the top and the temperature/density
    fields are plotted underneath.
    """
    fig = plt.figure(figsize=(8, 7), facecolor="black", layout="constrained")
    grid = fig.add_gridspec(nrows=2, ncols=4)
    axes_by_field = {
        "metallicity3": fig.add_subplot(grid[0, 1:3]),  # top mid
        "density": fig.add_subplot(grid[1, 0:2]),  # bottom left
        "temperature": fig.add_subplot(grid[1, 2:4]),  # bottom right
    }

    image_handles: dict[str, AxesImage] = {}
    for field_name, field_axes in axes_by_field.items():
        cmap_name, cbar_label = FIELD_STYLES[field_name]
        image_handles[field_name] = field_axes.imshow(
            10 ** first_logs[field_name],
            norm=LogNorm(*field_ranges[field_name]),  # tuple unpack into lognorm, handles colour log interp automatically 
            cmap=cmap_name,
            origin="lower",
            interpolation="nearest",  # REVIEW: nearest produces nice output but could also consider other methods?
        )
        field_axes.set_axis_off()

        # the classic frac/pad colourbar settings from that one stackoverflow
        colourbar = fig.colorbar(image_handles[field_name], ax=field_axes, fraction=0.046, pad=0.04)
        colourbar.set_label(cbar_label, color="white")
        colourbar.ax.tick_params(colors="white", which="both")
        colourbar.outline.set_edgecolor("white")  # nice aesthetic

    n_pixels = first_logs["density"].shape[1]

    # scale bar: NOTE: set to parsecs for now but if we need a bigger frame we can expand 
    scale_bar = AnchoredSizeBar(
        axes_by_field["density"].transData,  # put on density only, centred
        scale_bar_kpc / width_kpc * n_pixels,
        f"{scale_bar_kpc * 1000:.0f} pc",
        "upper center",  
        color="white",
        frameon=False,
        sep=3.0,
        size_vertical=n_pixels / 200,
    )
    axes_by_field["density"].add_artist(scale_bar)

    age_text = fig.text(0.05, 0.05, "", color="white", fontsize=14, family="monospace")  # following Britton's aesthetic choices

    return fig, image_handles, age_text


def _render_chunk(  # ugly function signature but necessary for parallelisation
    frame_indices: np.ndarray,
    lower_indices: np.ndarray,
    projection_paths: list[Path],
    snapshot_times_myr: np.ndarray,
    frame_times_myr: np.ndarray,
    field_ranges: dict[str, tuple[float, float]],
    creation_time_myr: float,
    width_kpc: float,
    scale_bar_kpc: float,
    frame_dir: Path,
) -> None:
    """
    Renders a chunk of frames with its own figure (the figures aren't pooled across processes). 
    Saves to the frame_dir.
    """
    if frame_indices.size == 0:
        return

    # override some mnras params, especially usetex which is noticeably slower
    with plt.style.context([MPL_STYLE, {"text.usetex": False, "mathtext.fontset": "cm", "savefig.bbox": None}]):
        loaded_index = int(lower_indices[frame_indices[0]])
        lower_logs = _load_log_fields(projection_paths[loaded_index], field_ranges)
        upper_logs = _load_log_fields(projection_paths[loaded_index + 1], field_ranges)

        fig, image_handles, age_text = _build_figure(lower_logs, field_ranges, width_kpc, scale_bar_kpc)

        for frame_index in frame_indices:
            frame_time_myr = frame_times_myr[frame_index]
            lower_index = int(lower_indices[frame_index])

            if lower_index != loaded_index:
                lower_logs = _load_log_fields(projection_paths[lower_index], field_ranges)
                upper_logs = _load_log_fields(projection_paths[lower_index + 1], field_ranges)
                loaded_index = lower_index

            # for readability
            lower_time_myr = snapshot_times_myr[lower_index]
            upper_time_myr = snapshot_times_myr[lower_index + 1]

            interp_weight = (frame_time_myr - lower_time_myr) / (upper_time_myr - lower_time_myr)

            for field_name, image_handle in image_handles.items():
                image_handle.set_data(
                    10 ** ((1 - interp_weight) * lower_logs[field_name] + interp_weight * upper_logs[field_name])
                )

            age_text.set_text(f"{frame_time_myr - creation_time_myr:.2f} Myr")
            fig.savefig(frame_dir / f"frame_{frame_index:04d}.png", facecolor="black", dpi=200)

        plt.close(fig)


def render_frames(
    executor: MPIPoolExecutor,
    projection_paths: list[Path],
    snapshot_times_myr: np.ndarray,
    frame_times_myr: np.ndarray,
    field_ranges: dict[str, tuple[float, float]],
    creation_time_myr: float,
    width_kpc: float,
    frame_dir: Path,
    scale_bar_kpc: float = 0.1,
) -> None:
    """
    Parallelises over all frames.
    """
    assert frame_times_myr[0] >= creation_time_myr, "Creation time before first snapshot: check units."

    lower_indices = np.searchsorted(snapshot_times_myr, frame_times_myr, side="right") - 1  # finds where sigmoid time intersects snapshot time
    np.clip(lower_indices, a_min=0, a_max=len(snapshot_times_myr)-2, out=lower_indices)  # need -2 to avoid overshooting snapshot time

    n_workers = COMM_WORLD.Get_size() - 1  # one worker is the orchestrator
    frame_chunks = np.array_split(np.arange(len(frame_times_myr)), n_workers)  # give each worker a contiguous chunk

    # this call is hideous but performance is quite slow unless parallelised
    list(executor.starmap(_render_chunk, ((
        chunk, lower_indices, projection_paths, snapshot_times_myr, frame_times_myr,
        field_ranges, creation_time_myr, width_kpc, scale_bar_kpc, frame_dir,
    ) for chunk in frame_chunks)))  # list turns it into an iterator so results arrive before assembling the film


def assemble_film(frame_dir: Path, output_path: Path, framerate: int) -> None:
    """
    Assemble the frames together into a film as per Britton's recommended command.
    """
    subprocess.run(
        ["ffmpeg", 
         "-y", 
         "-framerate", str(framerate),  # needs every arg as a str or error
        "-i", str(frame_dir / "frame_%04d.png"),
        "-vcodec", "libx264", 
        "-vf", "scale=1280:-2,format=yuv420p",
        str(output_path)],
        check=True,
    )


if __name__ == "__main__":

    args = parse_args()
    console = instantiate_logger()
    ledger: pl.DataFrame = pl.read_parquet(args.ledger)
    summaries = star_lifetime_summary(ledger=ledger) 

    projection_dir: Path = args.projdir
    labels = args.stars or [path.name for path in projection_dir.iterdir() if path.is_dir()]

    with MPIPoolExecutor() as executor:
        for star in summaries.pipe(select_stars, labels).iter_rows(named=True):
            make_star_film(
                console=console,
                executor=executor,
                star=star, 
                projections_root=args.projdir, 
                plots_root=args.outdir, 
                axis=args.axis, 
                framerate=args.framerate
            )
            console.log(f"Finished film for {star['label']}.")

