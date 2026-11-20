"""

Functions for PopIII spectral energy distributions (SEDs). Main candidates:

- Schaerer+2002
- Yggdrasil

Hazy expects an SED file with:

LHS: photon energy (default in Rydberg, change with 'units' e.g. 'units eV')
RHS: intensity (default in J_nu, change with 'nuFnu')

"""

from pathlib import Path

import numpy as np

from ..utils import instantiate_console, load_config, parse_args


def tabulate_sed(
    energies_eV: np.ndarray,
    fluxes: np.ndarray,
    sed_path: Path,
    force: bool = False,
) -> None:
    """
    Turns arrays of energies and fluxes into a Cloudy-readable file. Writes it to `sed_path`.
    """

    if energies_eV.shape != fluxes.shape:
        raise ValueError("Shapes of input arrays are not equal.")

    if sed_path.exists() and not force:
        raise ValueError("SED already exists and -force flag is not enabled.")

    sed_path.unlink(missing_ok=True)

    intensities = (
        energies_eV * fluxes
    )  # proportional to L_nu which is proportional to J_nu, since cloudy normalises to Q(h) this should accurately control the shape

    first_row = f"{energies_eV[0]:.6f} {intensities[0]:.6f} units eV"
    remaining_rows = "\n".join(
        f"{energy:.6f} {intensity:.6f}" for energy, intensity in zip(energies_eV[1:], intensities[1:], strict=True)
    )

    sed = "\n".join((first_row, remaining_rows))

    sed_path.write_text(sed, encoding="ascii")


def binned_schaerer_sed() -> tuple[np.ndarray, np.ndarray]:
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

    return table_energies, table_fluxes


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args=args)
    console = instantiate_console()
    physics = config.cloudy.physics

    match physics.sed_type:
        case "SCHAERER-BANDS":
            console.log("Generating Schaerer table 4 band SED.")
            energies, fluxes = binned_schaerer_sed()

    tabulate_sed(energies_eV=energies, fluxes=fluxes, sed_path=physics.sed_path, force=args.force)
    console.log(f"SED written to {physics.sed_path}")
