# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for lavlab.geojson.geometry -- pure-Python, no OMERO required."""

from lavlab.geojson.geometry import (
    bridge_hole,
    close_ring,
    omero_color_to_rgb,
    open_ring,
    points_to_ring,
    rgb_to_omero_color,
    ring_to_points,
    unbridge_ring,
)


def test_rgb_omero_color_round_trip():
    for r, g, b, a in [
        (255, 0, 0, 255),
        (0, 128, 64, 90),
        (0, 0, 0, 0),
        (255, 255, 255, 255),
    ]:
        packed = rgb_to_omero_color(r, g, b, a)
        assert omero_color_to_rgb(packed) == (r, g, b)


def test_omero_color_to_rgb_none():
    assert omero_color_to_rgb(None) is None


def test_points_ring_round_trip():
    ring = [[0.0, 0.0], [10.5, 0.0], [10.5, 10.5], [0.0, 10.5]]
    text = ring_to_points(ring)
    assert points_to_ring(text) == ring


def test_points_to_ring_empty():
    assert points_to_ring(None) == []
    assert points_to_ring("") == []


def test_open_close_ring():
    ring = [[0, 0], [1, 0], [1, 1]]
    closed = close_ring(ring)
    assert closed[0] == closed[-1]
    assert open_ring(closed) == ring


def test_bridge_and_unbridge_hole_round_trip():
    outer = [[0, 0], [10, 0], [10, 10], [0, 10]]
    hole = [[3, 3], [7, 3], [7, 7], [3, 7]]

    bridged = bridge_hole(outer, hole)
    recovered_outer, holes = unbridge_ring(bridged)

    assert len(holes) == 1
    assert set(map(tuple, recovered_outer)) == set(map(tuple, outer))


def test_unbridge_ring_with_no_hole_is_unchanged():
    ring = [[0, 0], [10, 0], [10, 10], [0, 10]]
    outer, holes = unbridge_ring(ring)
    assert outer == ring
    assert holes == []


def test_unbridge_ring_ignores_consecutive_duplicate_seam_points():
    # A real production ROI (image 362, ROI 68078) hit this: its closing
    # point was redundantly repeated at both the start and end of the
    # ring (a digitization artifact, not a hole). find_bridge() used to
    # match on that redundant seam and treat the entire rest of the ring
    # as a bridged-out "hole", leaving the real outer ring with under 3
    # points -- silently dropping the whole shape on export.
    seam = [0, 0]
    ring = [seam, seam, [10, 0], [10, 10], [0, 10], [5, 10], seam, seam, seam]
    outer, holes = unbridge_ring(ring)

    assert holes == []
    assert len(outer) >= 3
    assert set(map(tuple, outer)) == {(0, 0), (10, 0), (10, 10), (0, 10), (5, 10)}
