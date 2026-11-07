"""

Functionality for converting the yt rays into Cloudy commands for the CloudyModel.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .commands import CloudyCommand
    from .ray_extraction import RayArrays, RayHomogeneous

import numpy as np

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
    SphereGeometry,
    StopCondition,
    TlawTable,
)


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
    log_depths = np.log10(np.concatenate(([HAZY_FLOOR], ray_arrs.depth.to("cm").d)))

    h_density = ray_arrs.h_density.to("cm**-3").d
    log_h_density = np.log10(np.concatenate((h_density[:1], h_density)))

    temperature = ray_arrs.temperature.to("K").d
    log_temperature = np.log10(np.concatenate((temperature[:1], temperature)))

    metallicity = np.maximum(ray_arrs.metallicity.to("dimensionless").d, 1e-34)
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


def common_commands(r0: float, log_max_depth: float, redshift: float) -> list[CloudyCommand]:
    """
    Commands which are agnostic to the arrays vs homogeneity choice.
    """
    cmds: list[CloudyCommand] = []
    cmds.append(SphereGeometry(log_inner_radius=np.log10(r0)))
    cmds.append(
        StopCondition(electron_frac=0.01, log_max_depth=log_max_depth)
    )  # go 0.01cm back so Cloudy doesn't extrapolate-abort
    cmds.append(Background(redshift=redshift, use_cosmic_rays=True))
    cmds.append(RadSource(shape=Blackbody(log_temperature=4.9), log_q_hydrogen=49.4))  # schaerer+2003 vals

    return cmds
