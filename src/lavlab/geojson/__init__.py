# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Move GeoJSON annotations into and out of OMERO.

geojson files and the ROI's that omero holds are different modes. This
is to help convert the them.

* **Interior rings.** OMERO's polygon is one flat point list with no concept
  of a hole. Holes are folded in as a zero-width "keyhole" slit that renders
  correctly under the nonzero fill rule, and are restored on the way back out.
* **Coordinate precision.** Whole-slide coordinates have more significant
  digits than ``%g`` formatting preserves, so they are written explicitly.
* **Provenance.** OMERO has no field for a source identifier, so the QuPath
  UUID is stored in ``Roi.description`` and recovered on export.

Only :mod:`lavlab.geojson.omero_io` imports ``omero``; everything else works
without it, which is what lets ``--dry-run`` and the whole test suite run on
a machine with no server and no Ice.

Typical use::

    from lavlab.geojson import convert_file, import_annotations
    from lavlab.omero_client import connect

    result = convert_file("slide.geojson")
    conn = connect(creds)
    import_annotations(conn, image_id=101, annotations=result.annotations)
"""

from lavlab.geojson.geojson_io import (
    Annotation,
    ConversionError,
    ConversionResult,
    ShapeSpec,
    annotation_to_feature,
    convert_file,
    dump_geojson,
    feature_to_annotation,
    load_features,
    shapes_to_geometry,
    summarise_annotations,
    summarise_features,
)
from lavlab.geojson.geometry import (
    bridge_hole,
    omero_color_to_rgb,
    points_to_ring,
    rgb_to_omero_color,
    ring_to_points,
    unbridge_ring,
)

__all__ = [
    "Annotation",
    "ConversionError",
    "ConversionResult",
    "ShapeSpec",
    "annotation_to_feature",
    "bridge_hole",
    "convert_file",
    "dump_geojson",
    "feature_to_annotation",
    "load_features",
    "omero_color_to_rgb",
    "points_to_ring",
    "rgb_to_omero_color",
    "ring_to_points",
    "shapes_to_geometry",
    "summarise_annotations",
    "summarise_features",
    "unbridge_ring",
]
