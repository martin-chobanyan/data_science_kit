import numpy as np


def resample_curve(curve: np.ndarray, num_points: int) -> np.ndarray:
    dists = np.cumsum(np.r_[0, np.sqrt((np.diff(curve, axis=0) ** 2).sum(axis=1))])
    dists_even = np.linspace(0, dists.max(), num_points)
    return np.c_[
        np.interp(dists_even, dists, curve[:, 0]),
        np.interp(dists_even, dists, curve[:, 1])
    ]


def close_polygon(poly: np.ndarray) -> np.ndarray:
    return np.insert(poly, 0, poly[-1], axis=0)
