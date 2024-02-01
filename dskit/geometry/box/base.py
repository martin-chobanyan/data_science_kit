from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from itertools import product
# TODO: use `cached_property` instead of `property` where appropriate
# from functools import cached_property
from math import floor
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from bvs.geometry.box.utils import get_extents, same_type


class AbstractBox(ABC):
    def __init__(self, extents: List[float], name: str = ""):
        """Abstract box class inherited by both 2D and 3D boxes

        Parameters
        ----------
        extents: List of extents with first half containing minimums and second half containing maximums
        name: Optional string name / ID
        """
        self.extents = extents
        self.ndim = int(len(extents) / 2)
        self.name = name

    @classmethod
    def from_centroid(cls, centroid, box_dims, name: str = ""):
        ndim = len(centroid)
        if isinstance(box_dims, (int, float)):
            box_dims = ndim * [box_dims]
        mins = []
        maxs = []
        for box_dim, component in zip(box_dims, centroid):
            mins.append(component - box_dim / 2)
            maxs.append(component + box_dim / 2)
        return cls(extents=mins+maxs, name=name)

    @abstractmethod
    def new_box(self, extents: List[float], name: str = "") -> AbstractBox:
        """Create a new box with the same name but different extents"""
        ...

    @property
    def mins(self) -> List[float]:
        return self.extents[:self.ndim]

    @property
    def maxs(self) -> List[float]:
        return self.extents[self.ndim:]

    @property
    def x_min(self) -> float:
        return self.mins[0]

    @property
    def y_min(self) -> float:
        return self.mins[1]

    @property
    def x_max(self) -> float:
        return self.maxs[0]

    @property
    def y_max(self) -> float:
        return self.maxs[1]

    @property
    def dim_ranges(self) -> List[Tuple[float, float]]:
        return list(zip(self.mins, self.maxs))

    @property
    def dim_sizes(self) -> List[float]:
        return [v_max - v_min for v_min, v_max in zip(self.mins, self.maxs)]

    @property
    def dim_prod(self) -> float:
        return np.prod(self.dim_sizes).item()

    @property
    def x_range(self) -> Tuple[float, float]:
        return self.dim_ranges[0]

    @property
    def y_range(self) -> Tuple[float, float]:
        return self.dim_ranges[1]

    @property
    def width(self) -> float:
        return self.dim_sizes[0]

    @property
    def height(self) -> float:
        return self.dim_sizes[1]

    @property
    def centroid(self) -> Tuple:
        return tuple(sum(r)/2 for r in self.dim_ranges)

    @property
    def corners(self) -> List[Tuple]:
        corners = []
        options_per_dim = [(False, True)] * self.ndim
        for flags in product(*options_per_dim):
            corner = tuple(self.extents[i + (self.ndim if flag else 0)] for i, flag in enumerate(flags))
            corners.append(corner)
        return corners

    def contains(self, point: Tuple) -> bool:
        for value, (v_min, v_max) in zip(point, self.dim_ranges):
            if not (v_min <= value <= v_max):
                return False
        return True

    @same_type
    def contains_box(self, other: AbstractBox) -> bool:
        return all(map(self.contains, other.corners))

    def sample_point(self) -> List[float]:
        point = []
        for v_min, v_max in self.dim_ranges:
            v = (v_max - v_min) * np.random.rand() + v_min
            point.append(v)
        return point

    @same_type
    def __eq__(self, other: AbstractBox) -> bool:
        return list(self.extents) == list(other.extents)

    def intersects(self, other: AbstractBox) -> bool:
        return intersects(self, other)

    def pad(self, padding) -> AbstractBox:
        return pad(self, padding)

    def clip_with_extents(self, clip_extents: List[float]) -> AbstractBox:
        return clip(self, clip_extents)

    def clip_with_box(self, box: AbstractBox) -> AbstractBox:
        return self.clip_with_extents(box.extents)

    def translate(self, translation: List[float]) -> AbstractBox:
        return translate(self, translation)

    def to_int(self) -> AbstractBox:
        return self.new_box(extents=list(map(int, self.extents)), name=self.name)

    def floor(self) -> AbstractBox:
        return self.new_box(extents=list(map(floor, self.extents)), name=self.name)

    def union(self, other: AbstractBox) -> AbstractBox:
        return union(self, other)

    def intersection(self, other: AbstractBox) -> Optional[AbstractBox]:
        return intersection(self, other)

    def iou(self, other: AbstractBox) -> float:
        intersection_box = intersection(self, other)
        if intersection_box is not None:
            return intersection_box.dim_prod / union(self, other).dim_prod
        return 0

    def relative_to(self, parent: AbstractBox) -> AbstractBox:
        if not parent.contains_box(self):
            print("WARNING: Parent bounding box does not contain the 'relative' box!")
        return self.translate(translation=list(map(lambda x: -1 * x, parent.mins)))


def pad(box: AbstractBox, padding: Union[int, float, Sequence]) -> AbstractBox:
    if isinstance(padding, (int, float)):
        extent_paddings = [sign * padding for sign in [-1, 1] for _ in range(box.ndim)]
    elif len(padding) == box.ndim:
        extent_paddings = list(map(lambda p: -1 * p, padding)) + list(padding)
    elif len(padding) == len(box.extents):
        extent_paddings = padding
    else:
        raise ValueError(f"Padding must be a single number or a tuple of size {box.ndim} or {2 * box.ndim}")
    new_extents = [v + p for v, p in zip(box.extents, extent_paddings)]
    return box.new_box(extents=new_extents, name=box.name)


def clip(box: AbstractBox, clip_extents: List[float]) -> AbstractBox:
    new_extents = []
    for v, c in zip(box.mins, clip_extents[:box.ndim]):
        new_extents.append(v if v - c > 0 else c)
    for v, c in zip(box.maxs, clip_extents[box.ndim:]):
        new_extents.append(v if v - c < 0 else c)
    return box.new_box(extents=new_extents, name=box.name)


def translate(box: AbstractBox, translation: List[float]) -> AbstractBox:
    assert len(translation) == box.ndim, f"Translation vector must be of length {box.ndim}"
    extent_shifts = list(translation) + list(translation)
    new_extents = [v + t for v, t in zip(box.extents, extent_shifts)]
    return box.new_box(extents=new_extents, name=box.name)


@same_type
def union_extents(box1: AbstractBox, box2: AbstractBox) -> List[float]:
    mins = [min(a, b) for a, b in zip(box1.mins, box2.mins)]
    maxs = [max(a, b) for a, b in zip(box1.maxs, box2.maxs)]
    return mins + maxs


@same_type
def intersects(box1: AbstractBox, box2: AbstractBox) -> bool:
    for (min1, max1), (min2, max2) in zip(box1.dim_ranges, box2.dim_ranges):
        if (min2 - max1) * (max2 - min1) >= 0:
            return False
    return True


def intersect_extents(box1: AbstractBox, box2: AbstractBox) -> Optional[List[float]]:
    if intersects(box1, box2):
        new_mins = []
        new_maxs = []
        for (min1, max1), (min2, max2) in zip(box1.dim_ranges, box2.dim_ranges):
            _, v_min, v_max, _ = sorted([min1, max1, min2, max2])
            new_mins.append(v_min)
            new_maxs.append(v_max)
        return new_mins + new_maxs
    return None


def union(box1: AbstractBox, box2: AbstractBox) -> AbstractBox:
    return box1.new_box(union_extents(box1, box2))


def intersection(box1: AbstractBox, box2: AbstractBox) -> Optional[AbstractBox]:
    new_extents = intersect_extents(box1, box2)
    if new_extents:
        return box1.new_box(new_extents)
    return None


def union_all(bboxes: List[AbstractBox]) -> AbstractBox:
    return reduce(union, bboxes)


def intersection_all(bboxes: List[AbstractBox]) -> AbstractBox:
    return reduce(intersection, bboxes)
