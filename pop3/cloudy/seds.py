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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils import CloudyConfig

import numpy as np

from ..utils import instantiate_console, load_config, parse_args
from .commands import Blackbody, LuminositySolar, QHydrogen, RadSource, TableSED

# name, what to normalise by, log value
NORMALISATIONS: dict[str, tuple[str, float]] = {
    "BLACKBODY": QHydrogen(np.log10(2.469e49)),
    "SCHAERER-ENZO": QHydrogen(np.log10(2.469e49)),
    "SCHAERER-40": QHydrogen(np.log10(2.469e49)),
    "LARKIN-40": LuminositySolar(5.480),
    "LARKIN-260": LuminositySolar(6.764),
}


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


def full_schaerer_sed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Tabulates the full Schaerer SED.
    """
    raise ValueError("Coming soon!")


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

    console.log("Regenerating SEDs.")

    # TODO: add Schaerer here when he emails
    sed_dict = {
        "LARKIN-40": misc.larkin_40_path,
        "LARKIN-260": misc.larkin_260_path,
    }
    sed_funcs = {
        "LARKIN-40": larkin_sed,
        "LARKIN-260": larkin_sed,
    }

    for key, path in sed_dict.items():
        if not path.exists():
            raise ValueError("config path does not exist.")

        output_path = misc.sed_dir / f"{key}.sed"
        if output_path.exists() and not args.force:
            console.log(f"{output_path.name} exists, skipping.")
            continue

        console.log(f"Now making {key} SED.")
        wavelengths, fluxes, unit = sed_funcs[key](path)
        console.log(f"  Points: {wavelengths.shape}")

        tabulate_sed(wavelengths=wavelengths, fluxes=fluxes, cloudy_unit=unit, out_path=output_path)
        console.log(f"  Wrote SED to {output_path}")
        del output_path, wavelengths, fluxes, unit  # so I don't pass these

    # this one doesn't come from a datafile so gets handled explicitly (should be the only of its kind)
    enzo_sed_path = misc.sed_dir / "SCHAERER-ENZO.sed"
    if enzo_sed_path.exists() and not args.force:
        console.log(f"{enzo_sed_path.name} exists, skipping.")
    else:
        e_wavelengths, e_fluxes, e_unit = binned_schaerer_sed()
        e_intensities = (
            e_wavelengths * e_fluxes
        )  # proportional to L_nu which is proportional to J_nu, since cloudy normalises to Q(h) this should accurately control the shape
        tabulate_sed(wavelengths=e_wavelengths, fluxes=e_intensities, cloudy_unit=e_unit, out_path=enzo_sed_path)
        console.log(f"  Wrote SED to {enzo_sed_path}")
