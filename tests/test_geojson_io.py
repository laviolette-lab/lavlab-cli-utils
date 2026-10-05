# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for lavlab.geojson.geojson_io -- pure-Python, no OMERO required."""

import json

import pytest

from lavlab.geojson.geojson_io import (
    Annotation,
    ConversionError,
    ShapeSpec,
    annotation_to_feature,
    feature_to_annotation,
    load_features,
    shapes_to_geometry,
    summarise_features,
)

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
HOLE = [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]


def _polygon_feature(coordinates, feature_id="abc", classification=None):
    properties = {}
    if classification:
        properties["classification"] = classification
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": {"type": "Polygon", "coordinates": coordinates},
        "properties": properties,
    }


def test_feature_to_annotation_simple_polygon():
    feature = _polygon_feature(
        [SQUARE], classification={"name": "tumor", "color": [255, 0, 0]}
    )
    warnings = []
    annotation = feature_to_annotation(feature, warnings)

    assert annotation is not None
    assert annotation.label == "tumor"
    assert annotation.color == (255, 0, 0)
    assert len(annotation.shapes) == 1
    assert not warnings


def test_feature_to_annotation_bridges_hole_and_warns():
    feature = _polygon_feature([SQUARE, HOLE])
    warnings = []
    annotation = feature_to_annotation(feature, warnings)

    assert annotation is not None
    assert any("bridged" in w for w in warnings)


def test_feature_to_annotation_unsupported_geometry_warns_and_skips():
    feature = {
        "type": "Feature",
        "id": "x",
        "geometry": {"type": "Bogus", "coordinates": [[0, 0], [1, 1]]},
        "properties": {},
    }
    warnings = []
    annotation = feature_to_annotation(feature, warnings)

    assert annotation is None
    assert any("unsupported geometry" in w for w in warnings)


def test_feature_to_annotation_point():
    feature = {
        "type": "Feature",
        "id": "p1",
        "geometry": {"type": "Point", "coordinates": [5, 7]},
        "properties": {},
    }
    annotation = feature_to_annotation(feature, [])

    assert annotation is not None
    assert len(annotation.shapes) == 1
    assert annotation.shapes[0].kind == "point"
    assert annotation.shapes[0].points == [[5, 7]]


def test_feature_to_annotation_multipoint():
    feature = {
        "type": "Feature",
        "id": "mp1",
        "geometry": {"type": "MultiPoint", "coordinates": [[1, 1], [2, 2], [3, 3]]},
        "properties": {},
    }
    annotation = feature_to_annotation(feature, [])

    assert annotation is not None
    assert len(annotation.shapes) == 3
    assert all(s.kind == "point" for s in annotation.shapes)


def test_feature_to_annotation_linestring():
    feature = {
        "type": "Feature",
        "id": "l1",
        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1], [2, 0]]},
        "properties": {},
    }
    annotation = feature_to_annotation(feature, [])

    assert annotation is not None
    assert len(annotation.shapes) == 1
    assert annotation.shapes[0].kind == "polyline"
    assert annotation.shapes[0].points == [[0, 0], [1, 1], [2, 0]]


def test_feature_to_annotation_multilinestring():
    feature = {
        "type": "Feature",
        "id": "ml1",
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [[[0, 0], [1, 1]], [[5, 5], [6, 6], [7, 5]]],
        },
        "properties": {},
    }
    annotation = feature_to_annotation(feature, [])

    assert annotation is not None
    assert len(annotation.shapes) == 2
    assert all(s.kind == "polyline" for s in annotation.shapes)


def test_feature_to_annotation_geometry_collection_mixed_kinds():
    feature = {
        "type": "Feature",
        "id": "gc1",
        "geometry": {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Polygon", "coordinates": [SQUARE]},
                {"type": "Point", "coordinates": [1, 1]},
            ],
        },
        "properties": {},
    }
    annotation = feature_to_annotation(feature, [])

    assert annotation is not None
    kinds = sorted(s.kind for s in annotation.shapes)
    assert kinds == ["point", "polygon"]


def test_round_trip_feature_annotation_feature():
    original = _polygon_feature(
        [SQUARE, HOLE], classification={"name": "tumor", "color": [1, 2, 3]}
    )
    warnings = []
    annotation = feature_to_annotation(original, warnings)

    rebuilt = annotation_to_feature(annotation, fallback_id="fallback")
    assert rebuilt["geometry"]["type"] == "Polygon"
    # bridged-then-unbridged should recover an outer ring and one hole
    assert len(rebuilt["geometry"]["coordinates"]) == 2
    assert rebuilt["properties"]["classification"]["name"] == "tumor"


def test_shapes_to_geometry_mixed_kinds_returns_geometry_collection():
    shapes = [
        ShapeSpec("polygon", SQUARE),
        ShapeSpec("point", [[1, 1]]),
    ]
    geometry = shapes_to_geometry(shapes)

    assert geometry["type"] == "GeometryCollection"
    types = sorted(g["type"] for g in geometry["geometries"])
    assert types == ["Point", "Polygon"]


def test_round_trip_mixed_kind_roi_via_geometry_collection():
    # A ROI with a Polygon and a Point together has no single-type GeoJSON
    # equivalent -- this is the actual OMERO round-trip scenario (a ROI
    # that would otherwise be silently dropped on export).
    annotation = Annotation(
        shapes=[
            ShapeSpec("polygon", SQUARE, label="tumor", color=(1, 2, 3)),
            ShapeSpec("point", [[1, 1]], label="tumor", color=(1, 2, 3)),
        ],
    )
    feature = annotation_to_feature(annotation, fallback_id="fallback")
    assert feature is not None
    assert feature["geometry"]["type"] == "GeometryCollection"

    rebuilt = feature_to_annotation(feature, [])
    assert rebuilt is not None
    kinds = sorted(s.kind for s in rebuilt.shapes)
    assert kinds == ["point", "polygon"]
    assert rebuilt.label == "tumor"


def test_round_trip_point_only_roi():
    annotation = Annotation(shapes=[ShapeSpec("point", [[4, 5]], label="marker")])
    feature = annotation_to_feature(annotation, fallback_id="fallback")
    assert feature["geometry"]["type"] == "Point"

    rebuilt = feature_to_annotation(feature, [])
    assert rebuilt is not None
    assert rebuilt.shapes[0].kind == "point"
    assert rebuilt.shapes[0].points == [[4, 5]]


def test_round_trip_polyline_only_roi():
    line_points = [[0, 0], [1, 1], [2, 0]]
    annotation = Annotation(shapes=[ShapeSpec("polyline", line_points, label="line")])
    feature = annotation_to_feature(annotation, fallback_id="fallback")
    assert feature["geometry"]["type"] == "LineString"

    rebuilt = feature_to_annotation(feature, [])
    assert rebuilt is not None
    assert rebuilt.shapes[0].kind == "polyline"
    assert rebuilt.shapes[0].points == line_points


def test_summarise_features_handles_geometry_collection_without_crashing():
    features = [
        _polygon_feature([SQUARE]),
        {
            "type": "Feature",
            "id": "gc1",
            "geometry": {
                "type": "GeometryCollection",
                "geometries": [
                    {"type": "Polygon", "coordinates": [SQUARE]},
                    {"type": "Point", "coordinates": [1, 1]},
                ],
            },
            "properties": {},
        },
    ]

    stats = summarise_features(features)

    assert stats["features"] == 2
    assert stats["geometries"] == {"Polygon": 1, "GeometryCollection": 1}
    assert stats["holed"] == 0


def test_load_features_malformed_json_raises_conversion_error(tmp_path):
    bad_file = tmp_path / "bad.geojson"
    bad_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ConversionError):
        load_features(bad_file)


def test_load_features_wrong_top_level_type_raises(tmp_path):
    path = tmp_path / "wrong.geojson"
    path.write_text(
        json.dumps({"type": "Point", "coordinates": [0, 0]}), encoding="utf-8"
    )

    with pytest.raises(ConversionError):
        load_features(path)


def test_load_features_accepts_feature_collection(tmp_path):
    path = tmp_path / "fc.geojson"
    path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": [_polygon_feature([SQUARE])]}
        ),
        encoding="utf-8",
    )
    features = load_features(path)
    assert len(features) == 1
