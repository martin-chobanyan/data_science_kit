from __future__ import annotations
from typing import List, Optional, Tuple

from numpy import ndarray

from bvs.geometry.box.base import AbstractBox
from bvs.geometry.box.utils import get_extents
from bvs.misc.iter_utils import circular_pairs


class Box3D(AbstractBox):
    @classmethod
    def from_points(cls, points: ndarray, name: str = "") -> Box3D:
        return cls(get_extents(points), name=name)

    def new_box(self, extents: List[float], name: Optional[str] = None) -> Box3D:
        return Box3D(extents, name=self.name if name is None else name)

    @property
    def z_min(self) -> float:
        return self.mins[2]

    @property
    def z_max(self) -> float:
        return self.maxs[2]

    @property
    def z_range(self) -> Tuple[float, float]:
        return self.dim_ranges[2]

    @property
    def depth(self) -> float:
        return self.dim_sizes[2]

    @property
    def volume(self) -> float:
        return self.dim_prod

    @property
    def faces(self) -> List[List[Tuple]]:
        corners = self.corners
        face1 = [corners[0], corners[1], corners[3], corners[2]]
        face2 = [corners[4], corners[5], corners[7], corners[6]]
        other_faces = []
        for (a, b), (c, d) in circular_pairs(list(zip(face1, face2))):
            other_faces.append([a, b, d, c])
        faces = [face1, face2] + other_faces
        return faces

    def clip(
        self,
        x_min: Optional[float] = None,
        y_min: Optional[float] = None,
        z_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_max: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> Box3D:
        clip_extents = [
            x_min if x_min is not None else self.x_min,
            y_min if y_min is not None else self.y_min,
            z_min if z_min is not None else self.z_min,
            x_max if x_max is not None else self.x_max,
            y_max if y_max is not None else self.y_max,
            z_max if z_max is not None else self.z_max,
        ]
        return self.clip_with_extents(clip_extents)

    def __repr__(self) -> str:
        return f"Box3D({self.x_min}, {self.y_min}, {self.z_min}, {self.x_max}, {self.y_max}, {self.z_max})"
