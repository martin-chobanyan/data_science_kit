# TODO: break these tests up and adapt them to work with pytest

import numpy as np

from bvs.geometry.box import Box2D, Box3D


def test_box2d():
    box_a = Box2D([0, 0, 100, 200])
    box_b = Box2D([-40, 5, 20, 90])
    box_c = Box2D([20, 0, 40, 50])

    assert box_a.centroid == (50, 100), "Failed centroid for box A"
    assert box_b.centroid == (-10, 47.5), "Failed centroid for box B"
    assert box_c.centroid == (30, 25), "Failed centroid for box C"

    assert box_a.width == 100, "Failed width for box A"
    assert box_a.height == 200, "Failed height for box A"

    assert box_a.area == 20_000, "Failed to compute area for box A"
    assert box_b.area == 5100, "Failed to compute area for box B"
    assert box_c.area == 1000, "Failed to compute area for box C"

    assert box_c.corners == [(20, 0), (20, 50), (40, 0), (40, 50)], "Failed to get corners from box C"

    assert box_a.contains((10, 20)), "Point (10, 20) should be inside box A"
    assert not box_a.contains((300, 20)), "Point (300, 20) should not be inside box A"
    assert box_b.contains((-40, 5)), "Point (-40, 5) should be inside box B"
    assert box_b.contains((0, 20)), "Point (0, 20) should be inside box B"
    assert not box_c.contains((60, 70)), "Point (60, 70) should not be inside box C"

    assert box_a.contains_box(box_c), "Box A should contain Box C"
    assert not box_c.contains_box(box_a), "Box C should not contain Box A"
    assert not box_b.contains_box(box_c), "Box B should not contain Box C"

    assert box_a.pad(10) == Box2D([-10, -10, 110, 210]), "Padding box A with single padding amount failed"
    assert box_a.pad([10, 20]) == Box2D([-10, -20, 110, 220]), "Padding box A with equal padding per dim failed"
    assert box_a.pad([10, 0, 30, 40]) == Box2D([10, 0, 130, 240]), "Padding box A with full extent padding failed"

    assert box_a.clip(-100, -100, 1000, 1000) == box_a, "Box A clip failed"
    assert box_b.clip(x_min=0, y_min=0) == Box2D([0, 5, 20, 90]), "Box B clip failed"
    assert box_c.clip(30, 35, 35, 50) == Box2D([30, 35, 35, 50]), "Box C clip failed"

    assert box_a.translate([-10, 10]) == Box2D([-10, 10, 90, 210]), "Box A translation failed"

    assert box_a.union(box_b) == Box2D([-40, 0, 100, 200]), "Union of boxes A and B failed"
    assert box_b.union(box_c) == Box2D([-40, 0, 40, 90]), "Union of boxes B and C failed"

    assert box_a.intersects(box_b), "Box A must intersect Box B"
    assert box_a.intersection(box_b) == Box2D([0, 5, 20, 90]), "Box A intersection with Box B is incorrect"
    assert not box_a.intersects(Box2D([1000, 1000, 2000, 2000])), "Box A should not intersect (I)"
    assert not box_a.intersects(Box2D([20, 1000, 50, 2000])), "Box A should not intersect (II)"

    assert round(box_a.iou(box_b), 3) == round(17/280, 3), "IoU of A and B failed"
    assert box_a.iou(Box2D([1000, 1000, 2000, 2000])) == 0, "IoU of A and dummy box failed"

    assert box_c.relative_to(box_a) == box_c, "relative_to() failed for C in A"
    assert box_c.relative_to(Box2D([10, -10, 300, 300])) == Box2D([10, 10, 30, 60]), "relative_to() failed for C"

    polys = [
        np.array([
            [0, 0],
            [0, 40],
            [30, 30],
        ]),
        np.array([
            [60, 80],
            [100, 150],
            [90, 120],
        ]),
        np.array([
            [0, 200],
            [50, 50],
            [80, 120],
        ]),
    ]
    assert Box2D.from_polys(polys) == box_a, "Box2D.from_polys() failed"


def test_box3d():
    box_a = Box3D([100, 0, 200, 200, 100, 300])
    box_b = Box3D([150, 50, 250, 250, 150, 350])
    box_c = Box3D([100, 0, 200, 250, 150, 350])

    assert box_a.centroid == (150, 50, 250), "Failed to compute centroid for 3D Box A"
    assert box_b.centroid == (200, 100, 300), "Failed to compute centroid for 3D Box B"

    assert (box_a.width, box_a.height, box_a.depth) == (100, 100, 100), "Failed (W, H, D) check for 3D Box A"

    assert box_a.contains((125, 25, 225)), "Failed contains for 3D Box A"
    assert not box_a.contains((-10, 25, 225)), "Failed contains (negative) for 3D Box A"
    assert box_c.contains_box(box_a), "Failed contains_box. Box C should contain Box A"
    assert box_c.contains_box(box_b), "Failed contains_box. Box C should contain Box B"
    assert not box_b.contains_box(box_a), "Failed contains_box. Box B should not contain Box A"

    assert box_a.pad(10) == Box3D([90, -10, 190, 210, 110, 310]), "Failed to uniformly pad 3D Box A"
    assert box_a.pad((10, 20, 30)) == Box3D([90, -20, 170, 210, 120, 330]), "Failed per-dim pad for 3D Box A"
    assert box_c.pad(-25).translate([-25, -25, -25]) == box_a, "Failed to pad and translate 3D C to A"

    assert box_a.union(box_b) == box_c, "Union of A and B should equal C"
    assert box_a.union(box_a.translate([0, -50, 100])) != box_c, "Union of A and its translation should not equal C"

    assert box_a.intersection(box_b) == Box3D([150, 50, 250, 200, 100, 300]), "Failed A and B intersection"
    assert box_c.intersection(box_a) == box_a, "Failed A and C intersection"
    assert round(box_a.iou(box_b), 5) == round(1 / 27, 5), "Failed to compute IoU of A and B"
    
    assert box_b.relative_to(box_c) == Box3D([50, 50, 50, 150, 150, 150]), "Failed relative_to for B to C"


def test_all():
    test_box2d()
    test_box3d()


test_all()
