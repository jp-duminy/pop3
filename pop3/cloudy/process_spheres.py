"""

Scripts for extracting information from spheres, including rays, profiles, sampling, etc.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator
    from yt.data_objects.static_output import Dataset
    from unyt import unyt_array

import numpy as np
import healpy as hp

SEED = 2317434

def random_sample_sphere(
    rng: Generator,
    n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly samples a sphere in (theta, phi), per https://mathworld.wolfram.com/SpherePointPicking.html. To evenly
    sample the sphere many samples may be necessary.
    Returns:

    - theta: randomly sampled theta (polar) values
    - phi: randomly sampled phi (azimuth) values
    """

    rand_u = rng.uniform(0, 1, size=n_samples)
    rand_v = rng.uniform(0, 1, size=n_samples)

    phi = 2.0 * np.pi * rand_u  # NOTE: wolfram uses the mathematician's convention
    theta = np.arccos(2.0 * rand_v - 1.0)

    return theta, phi


def uniform_sample_sphere(
    n_side: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniformly samples the sphere in (theta, phi) via healpix (healpy). n_side must be a power of 2. Returns:

    - theta: theta (polar) values, sampled at evenly-spaced solid angles
    - phi: phi (azimuth) values, sampled at evenly-spaced solid angles
    """
    n_pix = hp.pixelfunc.nside2npix(n_side)
    theta, phi = hp.pix2ang(n_side, np.arange(n_pix))

    return theta, phi


def spherical_to_cartesian(
    radius: unyt_array,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts arrays in spherical polar coordinates to cartesian coordinates.
    """
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return x, y, z


def extract_sightline(
    sp_ds: Dataset,
    coords: unyt_array,
    fields: list[tuple[str, str]],
) -> tuple[tuple[unyt_array, ...], unyt_array]:
    """
    Extracts spatially-ordered arrays from the centre of a sphere dataset to the point specified by `coords` (a point offset from
    the origin of the sphere). This is done by casting a `yt` ray. `coords` should be in Cartesian and CGS. All fields specified by `fields` will be extracted.
    Returns:

    - arrays: a tuple of arrays, one for each field.
    - path_length: the path length of the ray
    """
    centre = sp_ds.parameters["center"].to("cm")
    ray = sp_ds.ray(centre, coords + centre)
    ray_sort = np.argsort(ray["t"], stable=True)

    arrays = tuple(ray[field][ray_sort] for field in fields)
    path_length = ray["t"][ray_sort] * np.linalg.norm(coords)

    return arrays, path_length
