"""

Functionality for keeping track of all the PopIII lines and how they go into/come out of Cloudy.

"""

import argparse
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..utils import instantiate_console


class Line(NamedTuple):
    """
    Contains information for a single line.

    - cloudy_label: the key of this line in the output df
    - wavelength: the wavelength in angstrom of this line following Cloudy convention
    - plot_label: nice label for matplotlib
    """

    cloudy_label: str
    vacuum_wavelength: float
    plot_label: str

    def observed_wavelength(self, redshift: float) -> float:
        """
        Returns observed wavelength which is the redshifted vacuum wavelength.
        """
        return self.vacuum_wavelength * (1.0 + redshift)


@dataclass(frozen=True, slots=True)
class LineList:
    """
    Container for all of the Cloudy lines we're using, stored as `Line` objects.

    This was horrific to type out.
    """

    H_alpha: Line = Line("H  1 6562.80A", 6564.61, r"H$\alpha$")
    H_beta: Line = Line("H  1 4861.32A", 4862.68, r"H$\beta$")
    H_gamma: Line = Line("H  1 4340.46A", 4341.68, r"H$\gamma$")
    H_delta: Line = Line("H  1 4101.73A", 4102.89, r"H$\delta$")
    H_epsilon: Line = Line("H  1 3970.07A", 3971.19, r"H$\epsilon$")
    H_eight: Line = Line("H  1 3889.05A", 3890.15, r"H8")
    H_nine: Line = Line("H  1 3835.38A", 3836.47, r"H9")
    H_ten: Line = Line("H  1 3797.90A", 3798.98, r"H10")
    HeI_4471: Line = Line("Blnd 4471.50A", 4472.75, r"He I $\lambda$4471")
    HeI_5876: Line = Line("Blnd 5875.66A", 5877.29, r"He I $\lambda$5876")
    HeII_1640: Line = Line("He 2 1640.41A", 1640.41, r"He II $\lambda$1640")
    HeII_4686: Line = Line("He 2 4685.68A", 4686.99, r"He II $\lambda$4686")
    CIII_1909: Line = Line("Blnd 1909.00A", 1909.00, r"C III] $\lambda$1909")
    CIV_1549: Line = Line("Blnd 1549.00A", 1549.00, r"C IV $\lambda$1549")
    NV_1240: Line = Line("Blnd 1240.00A", 1240.00, r"N V $\lambda$1240")
    OI_6300: Line = Line("O  1 6300.30A", 6302.04, r"[O I] $\lambda$6300")
    OII_3727: Line = Line("Blnd 3727.00A", 3728.06, r"[O II] $\lambda$3727")
    OIII_1666: Line = Line("Blnd 1666.00A", 1666.00, r"O III] $\lambda$1666")
    OIII_4363: Line = Line("O  3 4363.21A", 4364.44, r"[O III] $\lambda$4363")
    OIII_4959: Line = Line("O  3 4958.91A", 4960.29, r"[O III] $\lambda$4959")
    OIII_5007: Line = Line("O  3 5006.84A", 5008.24, r"[O III] $\lambda$5007")
    NeIII_3869: Line = Line("Ne 3 3868.76A", 3869.86, r"[Ne III] $\lambda$3869")
    NII_6584: Line = Line("N  2 6583.45A", 6585.27, r"[N II] $\lambda$6584")


def air_to_vacuum(wavelengths_air: np.ndarray) -> np.ndarray:
    """
    Converts air wavelengths to vacuum wavelengths using the Peck & Reeder 1972
    formula (the five-parameter one). Ignores wavelengths < 2000.
    """
    wavelengths_vacuum = wavelengths_air.copy()
    is_air = wavelengths_air > 2000.0

    for _ in range(3):  # 3 iterations gives good enough convergence
        sigma_sq = (1.0e4 / wavelengths_vacuum[is_air]) ** 2
        refractive_index = 1.0 + (8060.51 + 2480990.0 / (132.274 - sigma_sq) + 17455.7 / (39.32957 - sigma_sq)) * 1.0e-8

        wavelengths_vacuum[is_air] = wavelengths_air[is_air] * refractive_index

    return wavelengths_vacuum


def parse_line_args() -> argparse.Namespace:
    """
    Parses the wavelengths
    """
    parser = argparse.ArgumentParser(description="Convert air wavelengths to vacuum.")
    parser.add_argument("wavelengths", nargs="+", help="List of air wavelengths.")

    return parser.parse_args()


if __name__ == "__main__":
    console = instantiate_console()

    # quick unit test
    test_wavelength = np.asarray([6562.80], dtype=np.float64)
    np.testing.assert_allclose(air_to_vacuum(test_wavelength), np.asarray([6564.61], dtype=np.float64), atol=1e-2)

    air_wavelengths = np.asarray(parse_line_args().wavelengths, dtype=np.float64)
    console.log(f"Input air wavelengths: {air_wavelengths}")
    vacuum_wavelengths = air_to_vacuum(air_wavelengths)
    console.log(f"Output vacuum wavelengths: {vacuum_wavelengths}")
