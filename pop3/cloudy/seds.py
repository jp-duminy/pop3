"""

Functions for PopIII spectral energy distributions (SEDs). Main candidates:

- Enzo Schaerer+2002
- Larkin+2023
- Tabulated Schaerer
- Yggdrasil?

Hazy expects an SED file with:

LHS: photon energy (default in Rydberg, change with 'units' e.g. 'units eV')
RHS: intensity (default in J_nu, change with 'nuFnu')

# NOTE: the Larkin 40 SED is actually 42.376 in table 5, there is a model at 37.694 but I thought if I had to
pick one value I'd rather go for the harder SED. Likewise with the Larkin 260, which is 275.853 for the same reason.

"""

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from ..utils import CloudyConfig

import numpy as np
import polars as pl
from astropy.constants import codata2022 as co
from astropy.constants import iau2015 as iau

from ..utils import get_logger, instantiate_console, load_config, parse_args
from .commands import Blackbody, LuminositySolar, QHydrogen, RadSource, TableSED

# name, what to normalise by, log value
NORMALISATIONS: dict[str, tuple[str, float]] = {
    "BLACKBODY": QHydrogen(np.log10(2.469e49)),
    "SCHAERER-ENZO": QHydrogen(np.log10(2.469e49)),
    "SCHAERER-40": QHydrogen(np.log10(2.469e49)),
    "LARKIN-40": LuminositySolar(5.480),
    "LARKIN-260": LuminositySolar(6.764),
}


class AgeTrack(NamedTuple):
    """
    Container for a Schaerer age track.
    """

    log_L_sun: float
    T_eff: float
    log_g: float


# keyed by age_myr then log(Lsun), T_eff, log(g)
# from mode000_500.dat 40 Msun block
SCHAERER_40: dict[float, AgeTrack] = {
    5.2517e-05: AgeTrack(5.420, 80000.0, 5.0),
    7.5340e-01: AgeTrack(5.454, 80000.0, 5.0),
    1.4016e00: AgeTrack(5.494, 75000.0, 5.0),
    1.9639e00: AgeTrack(5.532, 75000.0, 5.0),
    2.4512e00: AgeTrack(5.567, 70000.0, 5.0),
    2.8809e00: AgeTrack(5.601, 70000.0, 4.5),
    3.2570e00: AgeTrack(5.633, 65000.0, 4.5),
    3.5047e00: AgeTrack(5.656, 60000.0, 4.5),
    3.7227e00: AgeTrack(5.677, 55000.0, 4.5),
    3.8639e00: AgeTrack(5.691, 55000.0, 4.0),
}

logger = get_logger()


def get_sed_command(config: CloudyConfig) -> RadSource:
    """
    Determines the SED filepath, normalisation, and returns the `RadSource` command.
    """
    selected_sed = config.physics.sed_type
    norm = NORMALISATIONS[selected_sed]

    sed_path = get_sed_path(config)

    if sed_path is None:
        shape = Blackbody(log_temperature=4.9)
    else:
        shape = TableSED(sed_path=sed_path)

    sed = RadSource(shape=shape, normalisation=norm)

    return sed


def get_sed_path(config: CloudyConfig) -> Path | None:
    """
    Get the SED path from the config. This is just so other files can find the SEDs too.
    """
    if config.physics.sed_type == "BLACKBODY":
        return None
    return config.misc.sed_dir / f"{config.physics.sed_type}.sed"


def tabulate_sed(
    wavelengths: np.ndarray,
    fluxes: np.ndarray,
    cloudy_unit: str,  # going to make this a hard requirement so units don't haunt me
    out_path: Path,
) -> None:
    """
    Turns arrays of energies and fluxes into a Cloudy-readable file. Writes it to `sed_path`.
    """
    if wavelengths.shape != fluxes.shape:
        raise ValueError("Shapes of input arrays are not equal.")

    out_path.unlink(missing_ok=True)

    first_row = f"{wavelengths[0]} {fluxes[0]} {cloudy_unit}"
    remaining_rows = "\n".join(
        f"{energy} {intensity}" for energy, intensity in zip(wavelengths[1:], fluxes[1:], strict=True)
    )

    sed = "\n".join((first_row, remaining_rows))

    out_path.write_text(sed, encoding="ascii")


def binned_schaerer_sed() -> tuple[np.ndarray, np.ndarray, str]:
    """
    Table 4 of Schaerer+2002 Q values (four-bin spectra, cumulative). Turned into monochromatic four-band SED.

    Q(H) = 2.469E+49: 13.6eV
    Q(He0) = 1.316E+49: 24.6eV
    Q(He+) =  8.798E+46: 54.4eV
    Q(LW) = 2.903E+49: 11.2eV (LW = capable of dissociating H2)

    Returns:

    - energies, in eV s^-1
    - fluxes, in s^-1
    """
    q_lw = 2.903e49
    # cumulative ionising photons/s^-1 (treat LW differently since it belongs to H)
    q_ionising = np.array([2.469e49, 1.316e49, 8.798e46])
    q_ionising_band = -np.diff(
        q_ionising, append=0.0
    )  # append lets final difference be calculated whilst preserving shape

    q_band = np.insert(q_ionising_band, 0, q_lw)
    # NOTE: enzo mean of 58 means the He+ bin extends to (2 * 58) - 54.4 = 61.6
    e_vals = np.array([11.2, 13.598, 24.587, 54.418, 61.6])  # eV

    delta = 1e-3
    interior_points = (
        e_vals[1:-1, np.newaxis] + [-delta, delta]
    ).ravel()  # LW is lower floor, get two rows with +/- delta, ravel to flat order
    table_energies = np.concatenate(([e_vals[0]], interior_points, [e_vals[-1]]))

    band_levels = q_band / np.diff(e_vals)
    table_fluxes = np.repeat(band_levels, 2)

    return table_energies, table_fluxes, "units eV"


def full_schaerer_sed(
    schaerer_dir: Path, track: dict[float, AgeTrack]
) -> dict[float, tuple[np.ndarray, np.ndarray, str]]:
    """
    Tabulates a full Schaerer SED for one mass track.
    """
    # I'm not changing these names so am hardcoding
    tlusty = read_schaerer_tlusty_model(schaerer_dir / "f_tlusty_extendedgrid.fluxes")
    kurucz_wavelengths = read_schaerer_kurucz(schaerer_dir / "fm50k2.pck")

    seds: dict[float, tuple[np.ndarray, np.ndarray, str]] = {}

    for age_myr, point in track.items():
        model = tlusty.filter((pl.col("T_eff") == point.T_eff) & (pl.col("log(g)") == point.log_g))

        # if this gets hit it's over
        if model.is_empty():
            raise ValueError(f"No TLUSTY model for T_eff = {point.T_eff}, log(g) = {point.log_g}")

        flux_lambda = resample_to_kurucz(
            wavelengths=model["wavelength_angstrom"].to_numpy(),
            flux_nu=model["flux_nu"].to_numpy(),
            kurucz_wavelengths=kurucz_wavelengths,
        )

        seds[age_myr] = (kurucz_wavelengths, flux_lambda, "units angstrom Flambda")

    return seds


def read_schaerer_tlusty_model(tlusty_path: Path) -> pl.DataFrame:
    """
    Turns the TLUSTY flux file into a dataframe with (T_eff, log(g), wavelength) on each row.
    """
    lines = pl.DataFrame({"line": tlusty_path.read_text().splitlines()})

    tlusty_grid = (
        lines.with_columns(
            # first value is Teff, second is log(g) (surface gravity), with awkward spacing
            # those two values are only in the header but the forward_fill alias puts them on every row
            pl.col("line").str.extract(r"Teff,logg=\s+(\S+)", 1).cast(pl.Float64).forward_fill().alias("T_eff"),
            pl.col("line").str.extract(r"Teff,logg=\s+\S+\s+(\S+)", 1).cast(pl.Float64).forward_fill().alias("log(g)"),
            # two paired numbers in capital E exponent notation
            pl.col("line")
            .str.extract_groups(r"^\s+(?P<wavelength_angstrom>\S+E[+-]\d+)\s+(?P<flux_nu>\S+E[+-]\d+)\s*$")
            .alias("vals"),
        )
        .filter(
            pl.col("vals").struct.field("wavelength_angstrom").is_not_null()
        )  # drop row if either parsed to null (1 covers both)
        .unnest("vals")  # decompose the val column into its two groups
        .with_columns(pl.col("wavelength_angstrom", "flux_nu").cast(pl.Float64))  # cast decomposed cols to float
        .drop(pl.col("line"))
    )

    return tlusty_grid


def read_schaerer_tracks(track_path: Path) -> pl.DataFrame:
    """
    Turns the evolutionary model tracks into a dataframe where each row is a track point.
    """
    lines = pl.DataFrame({"line": track_path.read_text().splitlines()})

    track_grid = (
        lines.with_columns(
            pl.col("line")
            # mass value (float) like 40.00, with whitespace at the end to ignore WR
            .str.extract(r"^\s+(\d+\.\d+)(?:\s|$)", 1)
            .cast(pl.Float64)
            .forward_fill()
            .alias("initial_mass"),
            # in order: index, age (capital E exp notation), mass, log(L), log(T_eff)
            pl.col("line").str.extract_groups(r"^\s*(\d+)\s+(\S+E[+-]\d+)\s+(\S+)\s+(\S+)\s+(\S+)").alias("row"),
        )
        .filter(pl.col("row").struct.field("1").is_not_null())  # only reason to grab index, check for weird rows
        # (this is horrific) grab the final 4 columns
        .select(
            "initial_mass",
            *[
                pl.col("row").struct.field(str(group)).cast(pl.Float64).alias(name)
                for group, name in zip(range(2, 6), ["age_yr", "mass", "log_L", "log_T_eff"], strict=True)
            ],
        )
        .with_columns(pl.col("age_yr").truediv(1.0e6).alias("age_myr"))
        .drop("age_yr")
    )

    return track_grid


def read_schaerer_kurucz(kurucz_path: Path) -> np.ndarray:
    """
    Turns the esoteric Kurucz file into a wavelengths array.
    """
    values: list[float] = []
    for line in kurucz_path.read_text().splitlines()[22:]:  # the fortran routine does this
        if "TEFF" in line:
            break
        values.extend(float(val) for val in line.split())

    wavelengths = np.asarray(values) * 10.0  # nm to angstrom

    return wavelengths


def match_schaerer_models(tracks: pl.DataFrame, tlusty: pl.DataFrame, initial_mass: float) -> pl.DataFrame:
    """
    What the (not compiling) fortran routine was supposed to do. Find a model first by T_eff then by log(g).
    Returns a dataframe of the corresponding SED.
    """
    grid_points = tlusty.select("T_eff", "log(g)").unique()  # Cloudy won't accept dupes so dedupe

    log_g_sun = np.log10((iau.GM_sun / iau.R_sun**2).to("cm * s**-2").value)
    log_T_eff_sun = np.log10(((iau.L_sun / (4 * np.pi * iau.R_sun**2 * co.sigma_sb)).to("K**4").value) ** 0.25)

    matched = (
        tracks.filter(pl.col("initial_mass") == initial_mass)  # reduce to right mass track
        # get luminosity and surface gravity (cheeky galaxies and cosmology simplified formula here)
        .with_columns(
            (
                pl.lit(log_g_sun)
                .add(pl.col("mass").log10())
                .add((pl.col("log_T_eff").sub(log_T_eff_sun).mul(4.0).sub(pl.col("log_L"))))
            ).alias("log(g)_tracks"),
            (pl.lit(10.0).pow(pl.col("log_T_eff"))).alias("T_eff_star"),
        )
        # find last value in the TLUSTY grid which is closest to the effective temperature of the star
        .sort("T_eff_star")
        .join_asof(
            grid_points.select(pl.col("T_eff").unique().sort()),
            left_on="T_eff_star",
            right_on="T_eff",
            strategy="nearest",
        )
        # within that Teff, find the last value which is closest to the surface gravity of the star
        .sort("log(g)_tracks")
        .join_asof(
            grid_points.sort("log(g)"), left_on="log(g)_tracks", right_on="log(g)", by="T_eff", strategy="nearest"
        )
        .sort("age_myr")
    )

    return matched


def resample_to_kurucz(wavelengths: np.ndarray, flux_nu: np.ndarray, kurucz_wavelengths: np.ndarray) -> np.ndarray:
    """
    Log-linear interp onto the Kurucz wavelengths in wavelength space, what the fortran routine was doing.
    """
    flux_lambda = (
        np.pi * flux_nu * co.c.to("AA/s").value / wavelengths**2
    )  # NOTE: tlusty uses astrophysical flux which has pi in denom
    keep = flux_lambda > 0  # there are some negative points in the file which crash Cloudy
    # do the interpolation in logspace but we want linear flux
    flux = 10 ** np.interp(kurucz_wavelengths, wavelengths[keep], np.log10(flux_lambda[keep]))

    return flux


def schaerer_sed(dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Schaerer+2002 SEDs.
    """


def larkin_sed(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Larkin+2023 SEDs.
    """
    wavelengths_angstrom, energies_cgs = np.loadtxt(path, comments="#", delimiter=",", unpack=True, skiprows=11)
    # have to deduplicate entries otherwise Cloudy crashes
    unique_wavelengths, unique_idx = np.unique(wavelengths_angstrom, return_index=True)
    unique_energies_cgs = energies_cgs[unique_idx]

    units = "units angstrom FLambda"

    return unique_wavelengths, unique_energies_cgs, units


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args=args)
    console = instantiate_console()
    misc = config.cloudy.misc

    logger.info("Regenerating SEDs.")

    # TODO: add Schaerer here when he emails
    sed_dict = {
        "LARKIN-40": misc.larkin_40_path,
        "LARKIN-260": misc.larkin_260_path,
    }
    sed_funcs = {
        "LARKIN-40": larkin_sed,
        "LARKIN-260": larkin_sed,
    }

    # Larkin+2023 SEDs
    for key, path in sed_dict.items():
        if not path.exists():
            raise ValueError("config path does not exist.")

        output_path = misc.sed_dir / f"{key}.sed"
        if output_path.exists() and not args.force:
            logger.info(f"{output_path.name} exists, skipping.")
            continue

        logger.info(f"Now making {key} SED.")
        wavelengths, fluxes, unit = sed_funcs[key](path)
        logger.info(f"  Points: {wavelengths.shape[0]:,}")

        tabulate_sed(wavelengths=wavelengths, fluxes=fluxes, cloudy_unit=unit, out_path=output_path)
        logger.info(f"  Wrote SED to {output_path}")
        del output_path, wavelengths, fluxes, unit  # so I don't pass these

    # Schaerer+2002 SEDs (age dependence), TODO: hardcode for now and update if need be, YAGNI
    schaerer_out_dir = misc.sed_dir / "SCHAERER-40"
    if schaerer_out_dir.exists() and not args.force:
        logger.info(f"{schaerer_out_dir.name} exists, skipping.")
    else:
        logger.info("Now making SCHAERER-40 SEDs.")
        schaerer_out_dir.mkdir(parents=True, exist_ok=True)
        for age_myr, (s_wavelengths, s_fluxes, s_unit) in full_schaerer_sed(misc.schaerer_dir, SCHAERER_40).items():
            s_path = schaerer_out_dir / f"tlusty_{age_myr:.4e}.sed"
            tabulate_sed(wavelengths=s_wavelengths, fluxes=s_fluxes, cloudy_unit=s_unit, out_path=s_path)
            logger.info(f"  Wrote {s_path.name}")
        del s_wavelengths, s_fluxes, s_unit  # same again, stop myself from using these

    # very basic Enzo SED
    enzo_sed_path = misc.sed_dir / "SCHAERER-ENZO.sed"
    if enzo_sed_path.exists() and not args.force:
        logger.info(f"{enzo_sed_path.name} exists, skipping.")
    else:
        logger.info("Now making SCHAERER-ENZO SED.")
        e_wavelengths, e_fluxes, e_unit = binned_schaerer_sed()
        e_intensities = (
            e_wavelengths * e_fluxes
        )  # proportional to L_nu which is proportional to J_nu, since cloudy normalises to Q(h) this should accurately control the shape
        tabulate_sed(wavelengths=e_wavelengths, fluxes=e_intensities, cloudy_unit=e_unit, out_path=enzo_sed_path)
        logger.info(f"  Wrote SED to {enzo_sed_path}")
