"""

Functionality for extracting rays from the full Pop2Prime snapshots and converting these into tidy objects which
can then be fed to Cloudy via the CloudyModel and command dataclasses.

"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unyt import unyt_array
    from yt.data_objects.selection_objects.ray import YTRay
    from yt.data_objects.static_output import Dataset

from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
import yt

from ..sim.pop3_ledger import add_metallicity3

FIELDS: dict[str, tuple[str, str]] = {
    "temperature": ("gas", "temperature"),
    "metallicity": ("gas", "metallicity3"),
    "h_density": ("gas", "H_nuclei_density"),
}

UNITS: dict[str, str] = {
    "temperature": "K",
    "metallicity": "dimensionless",
    "h_density": "cm**-3",
    "hii_density": "cm**-3",
    "r0": "cm",
    "depth": "cm",
}


@dataclass(slots=True)
class RayHomogeneous:
    """
    Container for quantities of interest coming out of the rays, made homogenous, in CGS units.

    - temperature
    - metallicity
    - h_density
    - r0 (face of first illuminated cell)
    - depth
    """

    temperature: float
    metallicity: float
    h_density: float
    r0: float
    depth: float


@dataclass(slots=True)
class RayArrays:
    """
    Container for the four primary quantities of interest coming out of the rays. All quantities
    are in CGS units.

    - temperature
    - metallicity
    - h_density
    - r0 (face of first illuminated cell)
    - depth (midpoints of ray cells)
    """

    temperature: np.ndarray
    metallicity: np.ndarray
    h_density: np.ndarray
    r0: np.ndarray
    depth: np.ndarray

    def to_homogeneous(self, ray_dts: unyt_array, hii_density: unyt_array) -> RayHomogeneous:
        """
        Converts the object to a `RayHomogeneous` one by taking the mean of each quantity, weighted
        by cell path length `dts` * HII density.
        """
        max_depth = np.max(self.depth)
        quantities = {
            "temperature": self.temperature,
            "metallicity": self.metallicity,
            "h_density": self.h_density,
        }
        homogeneous_quantities: dict[str, float] = {}

        for name, quantity in quantities.items():
            homogeneous_quantities[name] = np.average(quantity, weights=(ray_dts * hii_density))

        homogeneous_ray = RayHomogeneous(
            **homogeneous_quantities,
            r0=self.r0,
            depth=max_depth,
        )

        return homogeneous_ray


def create_cloudy_tasks(
    snap_path: Path,
    star_pos_unitary: np.ndarray,
    ray_length_kpc: float,
    homogeneous: bool = False,  # all or nothing for now
    n_side: int = 8,
) -> Iterator[RayArrays | RayHomogeneous]:
    """
    Yields Cloudy tasks as an iterable so `mpi4py.futures` can have workers collect the tasks as they come in.
    """
    ds = yt.load(snap_path)
    add_metallicity3(ds=ds)
    centre = ds.arr(star_pos_unitary, "unitary")
    radius = ds.quan(ray_length_kpc, "kpc").to("unitary")

    coords = generate_ray_coordinates(centre=centre, radius=radius, n_side=n_side)

    for endpoint in coords.T:
        ray_result = extract_ray(ds=ds, centre=centre, radius=radius, endpoint=endpoint, homogeneous=homogeneous)
        yield ray_result


def extract_ray(
    ds: Dataset, centre: unyt_array, radius: unyt_array, endpoint: unyt_array, homogeneous: bool = False
) -> RayArrays | RayHomogeneous:
    """
    Extracts a single ray from a `yt ds` using `centre` to the point specified by the `endpoint`, where
    each ray takes length `radius`. Returns:

    - `RayArrays` or the homogeneous version `RayHomogeneous`, depending on the bool flag.
    """
    ray: YTRay = ds.ray(centre, endpoint)

    # preload fields onto the container so the following methods easily access them
    preload_fields = [*[field for field in FIELDS.values()]]
    if homogeneous:
        preload_fields.append(("gas", "H_p1_number_density"))
    ray.get_data(preload_fields)

    ray_t = ray["t"]  # where the ray enters each cell
    ray_dts = ray["dts"]  # length of ray segment inside each cell
    ray_sort = np.argsort(ray_t, stable=True)  # depth-order of ray

    ray_t = ray_t[ray_sort]
    ray_dts = ray_dts[ray_sort]

    # I'm defining r0 as the position of the first cell hit by the star cell's ionising radiation
    r0: unyt_array = radius * (ray_t[0] + ray_dts[0])
    midpoints: unyt_array = (
        (ray_t + ray_dts / 2)[1:] * radius
    ) - r0  # the midpoint of each cell shifted so r0 is the origin

    arrays = RayArrays(
        **{
            name: ray[field][ray_sort][1:].to(UNITS[name]).d
            for name, field in FIELDS.items()  # strip units
        },  # an elegant unpack indeed
        # these should come out in CGS but let's assure ourselves
        r0=r0.to(UNITS["r0"]).d,
        depth=midpoints.to(UNITS["depth"]).d,
    )

    if homogeneous:
        hii_density = ray[("gas", "H_p1_number_density")][ray_sort][1:].to(UNITS["hii_density"]).d
        return arrays.to_homogeneous(ray_dts=ray_dts[1:].d, hii_density=hii_density)

    return arrays


def generate_ray_coordinates(centre: unyt_array, radius: unyt_array, n_side: int = 8) -> unyt_array:
    """
    Casts rays which uniformly sample the sphere on (theta, phi) via healpix (healpy). n_side must be a power of 2. Returns:

    - ray_coords: cartesian coords of points on a sphere of `radius` to sample from (shape is ((x, y, z), n_rays))
    """
    n_pix = hp.pixelfunc.nside2npix(n_side)
    theta, phi = hp.pix2ang(n_side, np.arange(n_pix))

    # convert theta[rad], phi[rad] into cartesian directions then scale to ray length and star position
    direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    ray_coords = radius * direction  # inherits the radius units
    ray_coords += centre[:, np.newaxis]  # broadcast centre to match dimensions

    return ray_coords
