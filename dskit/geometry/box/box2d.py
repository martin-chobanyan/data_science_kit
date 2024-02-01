from __future__ import annotations
from typing import List, Optional, Union

from numpy import ndarray

from bvs.geometry.box.base import AbstractBox
from bvs.geometry.box.utils import get_extents
from bvs.geometry.poly import to_points


class Box2D(AbstractBox):
    @classmethod
    def from_points(cls, points: ndarray, name: str = "") -> Box2D:
        return cls(get_extents(points), name=name)

    @classmethod
    def from_polys(cls, polys: Union[ndarray, List[ndarray]], name: str = "") -> Box2D:
        return cls.from_points(to_points(polys), name=name)

    def new_box(self, extents: List[float], name: Optional[str] = None) -> Box2D:
        return Box2D(extents, name=self.name if name is None else name)

    @property
    def area(self) -> float:
        return self.dim_prod

    def clip(
        self,
        x_min: Optional[float] = None,
        y_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> Box2D:
        clip_extents = [
            x_min if x_min is not None else self.x_min,
            y_min if y_min is not None else self.y_min,
            x_max if x_max is not None else self.x_max,
            y_max if y_max is not None else self.y_max,
        ]
        return self.clip_with_extents(clip_extents)

    def __repr__(self) -> str:
        return f"Box2D({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"
