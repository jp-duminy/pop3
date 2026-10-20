"""

Scripts for extracting information from spheres, including rays, profiles, sampling, etc.

"""

import numpy as np

SEED = 2317434

def sample_sphere(
    n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples a sphere in (theta, phi) evenly, per https://mathworld.wolfram.com/SpherePointPicking.html. Returns:

    - theta: randomly sampled theta (polar) values
    - phi: randomly sampled phi (azimuth) values
    """
    rng = np.random.default_rng(seed=SEED)

    rand_u = rng.uniform(0, 1, size=n_samples)
    rand_v = rng.uniform(0, 1, size=n_samples)

    phi = 2.0 * np.pi * rand_u  # NOTE: wolfram uses the mathematician's convention
    theta = np.arccos(2.0 * rand_v - 1.0)

    return theta, phi

    