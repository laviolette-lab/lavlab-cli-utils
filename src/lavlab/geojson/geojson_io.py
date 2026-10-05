# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
QuPath GeoJSON reading, writing, and conversion to and from OMERO shapes.

This module holds the whole conversion in a form that needs no server: a
GeoJSON feature becomes a list of :class:`ShapeSpec`, and a list of
:class:`ShapeSpec` becomes a GeoJSON geometry again.`lavlab.geojson.omero_io`
is the only part that talks to OMERO.

Warnings are collected into the returned result rather than printed, so the
library stays usable from a notebook or another package. The CLI prints them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lavlab.geojson.geometry import (
    MIN_RING_POINTS,
    bridge_hole,
    close_ring,
    ellipse_to_ring,
    open_ring,
    points_to_ring,
    qupath_colorrgb_to_rgb,
    rectangle_to_ring,
    ring_to_points,
    unbridge_ring,
)

#: Colour used for a feature that carries no classification.
DEFAULT_COLOR = (255, 255, 0)

#: Marker written into ``Roi.description`` so we only read back our own JSON.
PROVENANCE_TAG = "qupath"

#: Shape kinds
POLYGON = "polygon"
POLYLINE = "polyline"
POINT = "point"


class ConversionError(Exception):
    """Raised when a file cannot be read or is not GeoJSON."""


@dataclass
class ShapeSpec:
    """One OMERO shape, described without depending on ``omero``.

    :param kind: one of ``polygon``, ``polyline`` or ``point``
    :param points: coordinate pairs, closing point omitted
    :param label: classification name, or ``None``
    :param color: red/green/blue triple, or ``None``
    """

    kind: str
    points: list[list[float]]
    label: str | None = None
    color: tuple | None = None


@dataclass
class Annotation:
    """One QuPath object, ready to become a single OMERO ROI.

    A MultiPolygon yields several shapes here; they belong to one ROI so the
    parts stay grouped as the single object they were in QuPath.

    :param shapes: the shapes making up this annotation
    :param source_id: the QuPath UUID, if the source recorded one
    :param object_type: QuPath's ``objectType``, usually ``annotation``
    """

    shapes: list[ShapeSpec]
    source_id: str | None = None
    object_type: str = "annotation"

    @property
    def label(self) -> str | None:
        """Return the first classification name among this object's shapes.

        :return: the class name, or ``None`` if unclassified
        :rtype: str | None
        """
        return next((s.label for s in self.shapes if s.label), None)

    @property
    def color(self) -> tuple | None:
        """Return the first colour among this object's shapes.

        :return: red/green/blue triple, or ``None``
        :rtype: tuple | None
        """
        return next((s.color for s in self.shapes if s.color), None)


@dataclass
class ConversionResult:
    """What a conversion produced, plus anything worth telling the user.

    :param annotations: successfully converted objects
    :param warnings: notes about skipped or altered content
    """

    annotations: list[Annotation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_features(path: str | Path) -> list[dict]:
    """Read a GeoJSON file and return its features.

    Accepts a ``FeatureCollection``, a lone ``Feature``, or a bare list of
    features, since all three turn up in exports.

    :param path: file to read
    :type path: str | Path
    :raises ConversionError: if the file is unreadable or not GeoJSON
    :return: the feature dictionaries
    :rtype: list[dict]
    """
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"not valid JSON ({exc})") from exc
    except OSError as exc:
        raise ConversionError(f"could not read ({exc})") from exc

    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ConversionError("expected a JSON object or array at the top level")
    if data.get("type") == "FeatureCollection":
        return data.get("features") or []
    if data.get("type") == "Feature":
        return [data]
    raise ConversionError(
        f"expected a FeatureCollection, got type={data.get('type')!r}"
    )


def parse_classification(
    feature: dict,
) -> tuple[str | None, tuple | None]:
    """Extract the class name and colour from a feature's properties.

    :param feature: a GeoJSON feature
    :type feature: dict
    :return: ``(name, (r, g, b))``, either of which may be ``None``
    :rtype: tuple[str | None, tuple | None]
    """
    classification = (feature.get("properties") or {}).get("classification")
    if not classification:
        return None, None
    if isinstance(classification, str):
        return classification, None

    name = classification.get("name")
    color = None
    raw = classification.get("color")
    if isinstance(raw, (list, tuple)) and len(raw) >= MIN_RING_POINTS:
        color = tuple(int(channel) for channel in raw[:3])
    elif "colorRGB" in classification:
        color = qupath_colorrgb_to_rgb(int(classification["colorRGB"]))
    return name, color


def _polygon_shapes_from_parts(
    parts: list,
    name: str | None,
    color: tuple | None,
    warnings: list[str],
    feature_id,
) -> list[ShapeSpec]:
    shapes: list[ShapeSpec] = []
    bridged = 0
    for part in parts:
        if not part or len(part[0]) < MIN_RING_POINTS:
            warnings.append(
                f"feature {feature_id}: skipped a part with under "
                f"{MIN_RING_POINTS} points"
            )
            continue
        ring = part[0]
        for hole in part[1:]:
            if len(hole) < MIN_RING_POINTS:
                warnings.append(f"feature {feature_id}: ignored a degenerate hole")
                continue
            ring = bridge_hole(ring, hole)
            bridged += 1
        shapes.append(ShapeSpec(POLYGON, open_ring(ring), name, color or DEFAULT_COLOR))
    if bridged:
        warnings.append(
            f"feature {feature_id}: {bridged} hole(s) bridged into the outline"
        )
    return shapes


def _geometry_to_shapes(
    geometry: dict,
    name: str | None,
    color: tuple | None,
    warnings: list[str],
    feature_id,
) -> list[ShapeSpec]:
    """Convert one non-collection GeoJSON geometry into ShapeSpecs.

    :param geometry: a single (non-``GeometryCollection``) GeoJSON geometry
    :type geometry: dict
    :param name: classification name to attach to every shape produced
    :type name: str | None
    :param color: colour to attach to every shape produced
    :type color: tuple | None
    :param warnings: list appended to when content is skipped or altered
    :type warnings: list[str]
    :param feature_id: the owning feature's id, for warning messages
    :return: the converted shapes (possibly empty)
    :rtype: list[ShapeSpec]
    """
    geometry_type = geometry.get("type")

    if geometry_type in ("Polygon", "MultiPolygon"):
        parts = (
            [geometry["coordinates"]]
            if geometry_type == "Polygon"
            else geometry["coordinates"]
        )
        return _polygon_shapes_from_parts(parts, name, color, warnings, feature_id)

    if geometry_type in ("Point", "MultiPoint"):
        points = (
            [geometry["coordinates"]]
            if geometry_type == "Point"
            else geometry["coordinates"]
        )
        shapes = []
        for point in points:
            if not point or len(point) < 2:
                warnings.append(f"feature {feature_id}: skipped a degenerate point")
                continue
            shapes.append(
                ShapeSpec(POINT, [list(point[:2])], name, color or DEFAULT_COLOR)
            )
        return shapes

    if geometry_type in ("LineString", "MultiLineString"):
        lines = (
            [geometry["coordinates"]]
            if geometry_type == "LineString"
            else geometry["coordinates"]
        )
        shapes = []
        for line in lines:
            if not line or len(line) < 2:
                warnings.append(f"feature {feature_id}: skipped a degenerate line")
                continue
            shapes.append(ShapeSpec(POLYLINE, list(line), name, color or DEFAULT_COLOR))
        return shapes

    warnings.append(
        f"feature {feature_id}: skipped, unsupported geometry {geometry_type!r}"
    )
    return []


def feature_to_annotation(feature: dict, warnings: list[str]) -> Annotation | None:
    """Convert one GeoJSON feature into an :class:`Annotation`.

    Holes are folded into their outer ring rather than dropped. Degenerate
    rings are skipped, because an under-specified polygon would be stored as
    invalid geometry rather than failing loudly. A ``GeometryCollection`` --
    how a ROI mixing shape kinds (e.g. a Polygon and a Point together) round
    trips, since no single-type geometry can hold both -- becomes one
    Annotation with one shape per sub-geometry, restoring the original ROI.

    :param feature: a GeoJSON feature
    :type feature: dict
    :param warnings: list appended to when content is skipped or altered
    :type warnings: list[str]
    :return: the converted object, or ``None`` if nothing usable remained
    :rtype: Annotation | None
    """
    geometry = feature.get("geometry") or {}
    geometry_type = geometry.get("type")
    feature_id = feature.get("id", "<no id>")
    name, color = parse_classification(feature)

    if geometry_type == "GeometryCollection":
        shapes: list[ShapeSpec] = []
        for sub_geometry in geometry.get("geometries") or []:
            shapes.extend(
                _geometry_to_shapes(sub_geometry, name, color, warnings, feature_id)
            )
    else:
        shapes = _geometry_to_shapes(geometry, name, color, warnings, feature_id)

    if not shapes:
        return None

    properties = feature.get("properties") or {}
    return Annotation(
        shapes=shapes,
        source_id=feature.get("id"),
        object_type=(
            properties.get("objectType")
            or properties.get("object_type")
            or "annotation"
        ),
    )


def convert_file(path: str | Path) -> ConversionResult:
    """Read a GeoJSON file into annotations ready for OMERO.

    :param path: file to read
    :type path: str | Path
    :raises ConversionError: if the file is unreadable or not GeoJSON
    :return: converted annotations and any warnings
    :rtype: ConversionResult
    """
    result = ConversionResult()
    for feature in load_features(path):
        annotation = feature_to_annotation(feature, result.warnings)
        if annotation is not None:
            result.annotations.append(annotation)
    return result


def provenance_json(annotation: Annotation) -> str | None:
    """Build the JSON blob stored in ``Roi.description``.

    OMERO's ROI table has no field for a source identifier, so without this
    the QuPath UUID is lost on import and an archive cannot be traced back to
    the file it came from. ``Roi.description`` is free text that
    ``findByImage`` already returns, so it costs no extra query in either
    direction. A ``MapAnnotation`` would be the more idiomatic home and would
    be searchable in the web UI, at the price of an extra save per ROI and a
    join on read.

    :param annotation: the object being stored
    :type annotation: Annotation
    :return: compact JSON, or ``None`` when there is nothing worth recording
    :rtype: str | None
    """
    payload: dict[str, Any] = {"tag": PROVENANCE_TAG}
    if annotation.source_id:
        payload["id"] = str(annotation.source_id)
    if annotation.object_type and annotation.object_type != "annotation":
        payload["objectType"] = annotation.object_type
    if len(payload) == 1:
        return None
    return json.dumps(payload, separators=(",", ":"))


def read_provenance(description: str | None) -> dict:
    """Recover what :func:`provenance_json` stored, if anything.

    Text that is not our JSON -- a note somebody typed, another tool's
    output, an empty field -- is ignored rather than guessed at.

    :param description: the ``Roi.description`` value
    :type description: str | None
    :return: the decoded payload, empty when absent or foreign
    :rtype: dict
    """
    if not description:
        return {}
    try:
        payload = json.loads(description)
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict) or payload.get("tag") != PROVENANCE_TAG:
        return {}
    return payload


def _geometry_for_kind(
    kind: str, shapes: list[ShapeSpec], unbridge: bool
) -> dict | None:
    """Combine same-kind shapes into a single GeoJSON geometry (no collection)."""
    if kind == POINT:
        usable = [s for s in shapes if s.points]
        if not usable:
            return None
        if len(usable) == 1:
            return {"type": "Point", "coordinates": list(usable[0].points[0])}
        return {
            "type": "MultiPoint",
            "coordinates": [list(s.points[0]) for s in usable],
        }

    if kind == POLYLINE:
        usable = [s for s in shapes if len(s.points) >= 2]
        if not usable:
            return None
        if len(usable) == 1:
            return {"type": "LineString", "coordinates": usable[0].points}
        return {
            "type": "MultiLineString",
            "coordinates": [s.points for s in usable],
        }

    parts = []
    for shape in shapes:
        if len(shape.points) < MIN_RING_POINTS:
            continue
        if unbridge:
            outer, holes = unbridge_ring(shape.points)
        else:
            outer, holes = shape.points, []
        if len(outer) < MIN_RING_POINTS:
            continue
        parts.append([close_ring(outer), *holes])

    if not parts:
        return None
    if len(parts) == 1:
        return {"type": "Polygon", "coordinates": parts[0]}
    return {"type": "MultiPolygon", "coordinates": parts}


def shapes_to_geometry(shapes: list[ShapeSpec], unbridge: bool = True) -> dict | None:
    """Combine one ROI's shapes into a single GeoJSON geometry.

    Several polygons in one ROI become a ``MultiPolygon``, which is how a
    QuPath MultiPolygon survives the trip in. Mixed shape kinds -- e.g. a
    Polygon and a Point in the same ROI -- have no single-type GeoJSON
    equivalent, so they become a ``GeometryCollection`` instead: the only
    way to keep every shape in the file rather than silently dropping the
    whole ROI. ``build_roi`` on the OMERO import side already accepts a
    flat, mixed-kind shape list per ROI, so this round trips cleanly even
    though it isn't how QuPath itself would have written the file.

    :param shapes: the shapes belonging to one ROI
    :type shapes: list[ShapeSpec]
    :param unbridge: restore keyhole slits as real interior rings
    :type unbridge: bool
    :return: a GeoJSON geometry, or ``None`` if nothing usable remained
    :rtype: dict | None
    """
    kinds = {shape.kind for shape in shapes}
    if len(kinds) == 1:
        return _geometry_for_kind(kinds.pop(), shapes, unbridge)

    geometries = []
    for kind in sorted(kinds):
        geometry = _geometry_for_kind(
            kind, [s for s in shapes if s.kind == kind], unbridge
        )
        if geometry is not None:
            geometries.append(geometry)

    if not geometries:
        return None
    if len(geometries) == 1:
        return geometries[0]
    return {"type": "GeometryCollection", "geometries": geometries}


def annotation_to_feature(
    annotation: Annotation, fallback_id: str, unbridge: bool = True
) -> dict | None:
    """Convert an annotation back into a QuPath 0.4+ GeoJSON feature.

    :param annotation: the object to write out
    :type annotation: Annotation
    :param fallback_id: identifier to use when no QuPath UUID was recorded
    :type fallback_id: str
    :param unbridge: restore keyhole slits as real interior rings
    :type unbridge: bool
    :return: a GeoJSON feature, or ``None`` if the geometry was unusable
    :rtype: dict | None
    """
    geometry = shapes_to_geometry(annotation.shapes, unbridge=unbridge)
    if geometry is None:
        return None

    properties: dict[str, Any] = {"objectType": annotation.object_type}
    if annotation.label:
        classification: dict[str, Any] = {"name": annotation.label}
        if annotation.color:
            classification["color"] = list(annotation.color)
        properties["classification"] = classification

    return {
        "type": "Feature",
        "id": annotation.source_id or fallback_id,
        "geometry": geometry,
        "properties": properties,
    }


def dump_geojson(features: list[dict], indent: int | None = 1) -> str:
    """Serialise features as a GeoJSON ``FeatureCollection``.

    :param features: the features to write
    :type features: list[dict]
    :param indent: JSON indent, or ``None`` for compact output
    :type indent: int | None
    :return: GeoJSON text
    :rtype: str
    """
    return json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=indent
    )


def summarise_features(features: list[dict]) -> dict:
    """Describe a set of features for reporting.

    :param features: GeoJSON features
    :type features: list[dict]
    :return: counts of features, classes, geometry types and holed shapes
    :rtype: dict
    """
    classes = sorted(
        {
            feature["properties"]["classification"]["name"]
            for feature in features
            if "classification" in feature["properties"]
        }
    )
    geometries: dict[str, int] = {}
    holed = 0
    for feature in features:
        geometry_type = feature["geometry"]["type"]
        geometries[geometry_type] = geometries.get(geometry_type, 0) + 1
        # Holes are a Polygon/MultiPolygon concept; a GeometryCollection has
        # no top-level "coordinates" key at all, so it must be excluded here
        # rather than accessed unconditionally.
        if geometry_type not in ("Polygon", "MultiPolygon"):
            continue
        coordinates = feature["geometry"]["coordinates"]
        parts = [coordinates] if geometry_type == "Polygon" else coordinates
        if any(len(rings) > 1 for rings in parts):
            holed += 1
    return {
        "features": len(features),
        "classes": classes,
        "geometries": geometries,
        "holed": holed,
    }


def summarise_annotations(annotations: list[Annotation]) -> dict:
    """Describe converted annotations for reporting.

    :param annotations: converted objects
    :type annotations: list[Annotation]
    :return: counts of objects, shapes, labelled objects and class names
    :rtype: dict
    """
    labels = sorted({a.label for a in annotations if a.label})
    labelled = sum(1 for a in annotations if a.label)
    return {
        "annotations": len(annotations),
        "shapes": sum(len(a.shapes) for a in annotations),
        "labelled": labelled,
        "unlabelled": len(annotations) - labelled,
        "classes": labels,
    }


__all__ = [
    "DEFAULT_COLOR",
    "POINT",
    "POLYGON",
    "POLYLINE",
    "PROVENANCE_TAG",
    "Annotation",
    "ConversionError",
    "ConversionResult",
    "ShapeSpec",
    "annotation_to_feature",
    "convert_file",
    "dump_geojson",
    "ellipse_to_ring",
    "feature_to_annotation",
    "load_features",
    "parse_classification",
    "points_to_ring",
    "provenance_json",
    "read_provenance",
    "rectangle_to_ring",
    "ring_to_points",
    "shapes_to_geometry",
    "summarise_annotations",
    "summarise_features",
]
