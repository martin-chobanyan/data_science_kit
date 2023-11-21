from __future__ import annotations

from functools import reduce
from typing import List, Optional, Tuple, Union

Point = Tuple[float, float]


class BoundingBox:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, name: str = ""):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.name = name

    @classmethod
    def from_centroid(cls, centroid: Point, box_dim: Union[float, Tuple[float, float]]):
        if isinstance(box_dim, (int, float)):
            w = h = box_dim
        else:
            w, h = box_dim
        x, y = centroid
        return cls(x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    @property
    def x_lim(self) -> Tuple[float, float]:
        return self.x_min, self.x_max

    @property
    def y_lim(self) -> Tuple[float, float]:
        return self.y_min, self.y_max

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    @property
    def corners(self) -> Tuple[Point, Point, Point, Point]:
        return (self.x_min, self.y_min), (self.x_min, self.y_max), (self.x_max, self.y_max), (self.x_max, self.y_min)

    def contains(self, pnt: Point) -> bool:
        x, y = pnt
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    def contains_bbox(self, other: BoundingBox) -> bool:
        return all(map(self.contains, other.corners))

    def pad(self, padding) -> BoundingBox:
        return pad(self, padding)

    def clip(
            self,
            x_min: Optional[float] = None,
            y_min: Optional[float] = None,
            x_max: Optional[float] = None,
            y_max: Optional[float] = None,
    ) -> BoundingBox:
        return clip(self, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def translate(self, translation: Tuple[float, float]) -> BoundingBox:
        return translate(self, translation)

    def union(self, other: BoundingBox) -> BoundingBox:
        return union(self, other)

    def intersection(self, other: BoundingBox) -> BoundingBox:
        return intersection(self, other)

    def iou(self, other: BoundingBox) -> float:
        if (intersection_bbox := intersection(self, other)) is not None:
            return intersection_bbox.area / union(self, other).area
        return 0

    def relative_to(self, parent: BoundingBox) -> BoundingBox:
        if not parent.contains_bbox(self):
            print("WARNING: Parent bounding box does not contain the 'relative' box!")
        return self.translate((-1 * parent.x_min, -1 * parent.y_min))

    def __eq__(self, other: BoundingBox) -> bool:
        return (self.x_lim == other.x_lim) and (self.y_lim == other.y_lim)

    def __repr__(self) -> str:
        return f"BoundingBox({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"


def pad(bbox: BoundingBox, padding) -> BoundingBox:
    if isinstance(padding, int) or isinstance(padding, float):
        pad_xmin = pad_ymin = pad_xmax = pad_ymax = padding
    else:
        if len(padding) == 2:
            pad_x, pad_y = padding
            pad_xmin = pad_xmax = pad_x
            pad_ymin = pad_ymax = pad_y
        elif len(padding) == 4:
            pad_xmin, pad_ymin, pad_xmax, pad_ymax = padding
        else:
            raise ValueError("Padding must be a single number, a (x, y) tuple, or a 4-tuple in each direction")
    x_min, x_max = bbox.x_lim
    y_min, y_max = bbox.y_lim
    return BoundingBox(
        x_min=x_min - pad_xmin,
        y_min=y_min - pad_ymin,
        x_max=x_max + pad_xmax,
        y_max=y_max + pad_ymax,
    )


def clip(
        bbox: BoundingBox,
        x_min: Optional[float] = None,
        y_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_max: Optional[float] = None,
) -> BoundingBox:
    return BoundingBox(
        x_min=max(bbox.x_min, x_min) if x_min is not None else bbox.x_min,
        y_min=max(bbox.y_min, y_min) if y_min is not None else bbox.y_min,
        x_max=min(bbox.x_max, x_max) if x_max is not None else bbox.x_max,
        y_max=min(bbox.y_max, y_max) if y_max is not None else bbox.y_max,
    )


def translate(bbox: BoundingBox, translation: Tuple[float, float]) -> BoundingBox:
    x_shift, y_shift = translation
    return BoundingBox(
        x_min=bbox.x_min + x_shift, y_min=bbox.y_min + y_shift, x_max=bbox.x_max + x_shift, y_max=bbox.y_max + y_shift
    )


def union(bbox1: BoundingBox, bbox2: BoundingBox) -> BoundingBox:
    return BoundingBox(
        x_min=min(bbox1.x_min, bbox2.x_min),
        y_min=min(bbox1.y_min, bbox2.y_min),
        x_max=max(bbox1.x_max, bbox2.x_max),
        y_max=max(bbox1.y_max, bbox2.y_max)
    )


def intersection(bbox1: BoundingBox, bbox2: BoundingBox) -> Optional[BoundingBox]:
    x_prod = (bbox2.x_min - bbox1.x_max) * (bbox2.x_max - bbox1.x_min)
    y_prod = (bbox2.y_min - bbox1.y_max) * (bbox2.y_max - bbox1.y_min)
    if x_prod < 0 and y_prod < 0:
        _, x_min, x_max, _ = sorted(bbox1.x_lim + bbox2.x_lim)
        _, y_min, y_max, _ = sorted(bbox1.y_lim + bbox2.y_lim)
        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    return None


def union_all(bboxes: List[BoundingBox]) -> BoundingBox:
    return reduce(union, bboxes)


def intersection_all(bboxes: List[BoundingBox]) -> BoundingBox:
    return reduce(intersection, bboxes)
