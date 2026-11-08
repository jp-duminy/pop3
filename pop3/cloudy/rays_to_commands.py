"""

Functionality for converting the yt rays into Cloudy commands for the CloudyModel.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .commands import CloudyCommand

import numpy as np

from ..utils import CloudyConfig
from .commands import (
    HAZY_FLOOR,
    Abundances,
    Background,
    Blackbody,
    ConstantDensity,
    ConstantTemperature,
    DlawTable,
    ElementAbundance,
    ElementTable,
    RadSource,
    SetDensityTolerance,
    SphereGeometry,
    StopCondition,
    TlawTable,
)
from .ray_extraction import RayArrays, RayHomogeneous


def build_cloudy_cmds(
    ray: RayArrays | RayHomogeneous,
    yields: dict[str, float],
    redshift: float,
    config: CloudyConfig,
) -> list[CloudyCommand]:
    """
    Converts a ray object into the associated `CloudyCommand` objects for the `CloudyModel`
    """
    expected_type = RayHomogeneous if config.homogeneous else RayArrays

    if not isinstance(ray, expected_type):
        raise ValueError("The config and passed type of ray disagree.")

    if config.homogeneous:
        cmds = homogeneous_to_cmds(ray_hom=ray, yields=yields)
        log_max_depth = np.log10(ray.depth)
    else:
        cmds = arrays_to_cmds(ray_arrs=ray, yields=yields)
        log_max_depth = np.log10(ray.depth[-1]) - 0.01  # go 0.01cm back so Cloudy doesn't extrapolate-abort

    cmds.extend(
        common_cmds(
            r0=ray.r0,
            log_max_depth=log_max_depth,
            redshift=redshift,
            stop_efrac=config.efrac_threshold,
            cosmic_rays=config.use_cosmic_rays,
            density_tol=config.density_tolerance,
        )
    )

    return cmds


def arrays_to_cmds(
    ray_arrs: RayArrays,
    yields: dict[str, float],
) -> list[CloudyCommand]:
    """
    Converts a RayArrays object to a list of necessary commands for the `CloudyModel`.
    """
    cmds: list[CloudyCommand] = []
    # prepend the tiny float to the depth arrays (hazy stipulates 1e-30 so let's go with 1e-35)
    # concat instead of 0-index then [1:] assignment is safer
    log_depths = np.log10(np.concatenate(([HAZY_FLOOR], ray_arrs.depth)))
    # concat needs the same dimensionality so slice with [:1] instead of extract with [0]
    log_h_density = np.log10(np.concatenate((ray_arrs.h_density[:1], ray_arrs.h_density)))
    log_temperature = np.log10(np.concatenate((ray_arrs.temperature[:1], ray_arrs.temperature)))
    # metallicity could actually be zero so floor it
    metallicity = np.maximum(ray_arrs.metallicity, 1e-34)  # NOTE: check whether this might accidentally turn metals off
    log_metallicity = np.log10(np.concatenate((metallicity[:1], metallicity)))

    cmds.extend(
        [
            DlawTable(log_densities=log_h_density, log_path_lengths=log_depths),
            TlawTable(log_temperatures=log_temperature, log_path_lengths=log_depths),
        ]
    )

    element_tables: list[ElementTable] = []
    for element, log_abundance in yields.items():
        element_tables.append(
            ElementTable(
                element_name=element,
                log_path_lengths=log_depths,
                log_abundances=log_metallicity + log_abundance,
            )
        )
    cmds.append(Abundances(elements=element_tables))

    return cmds


def homogeneous_to_cmds(
    ray_hom: RayHomogeneous,
    yields: dict[str, float],
) -> list[CloudyCommand]:
    """
    Converts a `RayHomogeneous` object to a list of necessary commands for the `CloudyModel`.
    """
    cmds: list[CloudyCommand] = []
    log_metallicity = np.log10(ray_hom.metallicity)

    cmds.append(ConstantDensity(log_h_density=np.log10(ray_hom.h_density)))
    cmds.append(ConstantTemperature(log_temperature=np.log10(ray_hom.temperature)))
    cmds.append(
        Abundances(
            elements=[
                ElementAbundance(element_name=name, log_abundance=log_metallicity + log_yield)
                for name, log_yield in yields.items()
            ]
        )
    )

    return cmds


def common_cmds(
    r0: float,
    log_max_depth: float,
    redshift: float,
    stop_efrac: float,
    cosmic_rays: bool,
    density_tol: float,
) -> list[CloudyCommand]:
    """
    Builds commands which are agnostic to the arrays vs homogeneity choice.
    """
    cmds: list[CloudyCommand] = []
    cmds.append(SphereGeometry(log_inner_radius=np.log10(r0)))
    cmds.append(StopCondition(electron_frac=stop_efrac, log_max_depth=log_max_depth))
    cmds.append(Background(redshift=redshift, use_cosmic_rays=cosmic_rays))
    cmds.append(RadSource(shape=Blackbody(log_temperature=4.9), log_q_hydrogen=49.4))  # schaerer+2003 vals
    cmds.append(SetDensityTolerance(relative_tolerance=density_tol))

    return cmds
