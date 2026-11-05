"""

Scripts for extracting information from spheres, including rays, profiles, sampling, etc.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unyt import unyt_array

import healpy as hp
import numpy as np


def cast_rays(centre: unyt_array, radius: unyt_array, n_side: int = 8) -> unyt_array:
    """
    Uniformly samples the sphere in (theta, phi) via healpix (healpy). n_side must be a power of 2. Returns:

    - ray_coords: cartesian coords of points on a sphere of `radius` to sample from (shape is ((x, y, z), n_rays))
    """
    n_pix = hp.pixelfunc.nside2npix(n_side)
    theta, phi = hp.pix2ang(n_side, np.arange(n_pix))

    # convert theta[rad], phi[rad] into cartesian directions then scale to ray length and star position
    direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    ray_coords = radius * direction  # inherits the radius units
    ray_coords += centre[:, np.newaxis]  # broadcast centre to match dimensions

    return ray_coords
