from typing import List

from numpy import ndarray


def same_type(func):
    def wrapper(box1, box2, *args, **kwargs):
        assert isinstance(box1, type(box2)), "Boxes must be the same type!"
        return func(box1, box2, *args, **kwargs)
    return wrapper


def get_extents(points: ndarray) -> List[float]:
    """Get the extents of a pointcloud

    Parameters
    ----------
    points: An array of shape (num_points, num_dims)

    Returns
    -------
    A list of extents where the first half contains minimums and last half contains maximums.

    For example, given an array of shape (N, 2) then the extents will be [x_min, y_min, x_max, y_max]
    where x refers to the first component `points[:, 0]` and y refers to the second component `points[:, 1]`
    """
    mins = []
    maxs = []
    for values in points.T:
        mins.append(values.min())
        maxs.append(values.max())
    return mins + maxs
