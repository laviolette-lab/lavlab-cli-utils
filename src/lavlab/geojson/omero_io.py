# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Uses Omero not just Python.

``omero`` is imported lazily inside each function so the rest of the package
-- and its tests -- work on a machine without ``omero-py`` installed. That is
also what lets ``--dry-run`` do real work with no server and no Ice.

Nothing here deletes or edits existing data. Importing adds ROIs; exporting
only reads. ROIs are rows in OMERO's database, not something written into the
image file.

Connecting to OMERO is not this module's job -- callers get a connection
from `lavlab.omero_client.connect`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lavlab.geojson.geojson_io import (
    POINT,
    POLYGON,
    POLYLINE,
    Annotation,
    ShapeSpec,
    annotation_to_feature,
    points_to_ring,
    provenance_json,
    read_provenance,
    ring_to_points,
)
from lavlab.geojson.geometry import (
    ellipse_to_ring,
    omero_color_to_rgb,
    rectangle_to_ring,
    rgb_to_omero_color,
)

SAVE_BATCH = 50

UNSUPPORTED_SHAPES = ("MaskI", "LabelI", "LineI")

#: Namespace exported GeoJSON archives are attached under. Deliberately not
#: under ``LargeRecon.*`` like the lr/roi attachments: those are artifacts of
#: a specific downsample, whereas a GeoJSON export is vector data with no
#: resolution attached to it at all. OMERO matches namespaces by exact
#: equality, so this can never be confused with either of those.
GEOJSON_NAMESPACE = "lavlab.geojson"

#: RFC 7946's registered media type for GeoJSON.
GEOJSON_MIMETYPE = "application/geo+json"


@dataclass
class ImportResult:
    """Outcome of importing one file onto one image.

    :param created: number of ROIs saved
    :param warnings: notes about skipped or altered content
    """

    created: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExportResult:
    """Outcome of reading one image's ROIs.

    :param features: GeoJSON features recovered
    :param warnings: notes about skipped shapes
    """

    features: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def unwrap(value: Any) -> Any:
    """Unwrap an OMERO rtype, tolerating ``None``.

    :param value: a wrapped value or ``None``
    :type value: Any
    :return: the plain Python value
    :rtype: Any
    """
    return None if value is None else value.getValue()


def safe_filename(name: str) -> str:
    """Reduce an image name to something safe to use as a filename.

    :param name: the OMERO image name
    :type name: str
    :return: a filesystem-safe stem
    :rtype: str
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "").strip("._-")
    return cleaned or "image"


def geojson_annotation_name(image) -> str:
    """Return the canonical filename a GeoJSON export is stored under.

    Identical to the local export filename, so an attachment downloaded
    from OMERO and a file written by ``--out`` are recognisably the same
    thing.

    :param image: an OMERO ``ImageWrapper``
    :return: e.g. ``N101_S06_HE.ome.tiff__omero-362.geojson``
    :rtype: str
    """
    return f"{safe_filename(image.getName())}__omero-{image.getId()}.geojson"


def _find_geojson_annotation(image, remote_name: str):
    """Return the GeoJSON file annotation named exactly *remote_name*."""
    for ann in image.listAnnotations(ns=GEOJSON_NAMESPACE):
        if not hasattr(ann, "getFile"):
            continue
        f = ann.getFile()
        if f is not None and f.getName() == remote_name:
            return ann
    return None


def has_uploaded_geojson(image) -> bool:
    """Return True if this image already has its GeoJSON export attached.

    A cheap existence check -- it lists annotations, it never downloads --
    so a caller can skip images that are already archived.

    :param image: an OMERO ``ImageWrapper``
    :return: whether a matching GeoJSON annotation exists
    :rtype: bool
    """
    return _find_geojson_annotation(image, geojson_annotation_name(image)) is not None


def upload_geojson(conn, image, local_path: str) -> str:
    """Attach *local_path* to *image* as its GeoJSON export, replacing any
    previous upload of the same file.

    Only an annotation with this exact canonical filename is replaced;
    anything else sharing the namespace is left alone.

    :param conn: a connected gateway, already switched into the image's group
    :param image: an OMERO ``ImageWrapper``
    :param local_path: the ``.geojson`` file to upload
    :type local_path: str
    :return: the canonical filename it was uploaded as
    :rtype: str
    """
    remote_name = geojson_annotation_name(image)

    stale = _find_geojson_annotation(image, remote_name)
    if stale is not None:
        image.removeAnnotations([stale])
        conn.deleteObject(stale._obj)

    file_ann = conn.createFileAnnfromLocalFile(
        local_path,
        origFilePathAndName=remote_name,
        mimetype=GEOJSON_MIMETYPE,
        ns=GEOJSON_NAMESPACE,
    )
    image.linkAnnotation(file_ann)
    return remote_name


def build_roi(
    image_id: int,
    annotation: Annotation,
    fill_alpha: int = 90,
    stroke_width: float = 2.0,
):
    """Build one OMERO ROI holding an annotation's shapes.

    The class name goes to ``Shape.textValue`` and ``Roi.name``, the colour to
    ``strokeColor`` and a translucent ``fillColor``, and the QuPath UUID to
    ``Roi.description``.

    :param image_id: target image
    :type image_id: int
    :param annotation: the object to store
    :type annotation: Annotation
    :param fill_alpha: 0-255, or 0 for outline only
    :type fill_alpha: int
    :param stroke_width: outline width in pixels
    :type stroke_width: float
    :return: an unsaved ROI
    :rtype: omero.model.RoiI
    """
    import omero
    from omero.rtypes import rint, rstring

    roi = omero.model.RoiI()
    roi.setImage(omero.model.ImageI(image_id, False))

    if annotation.label:
        roi.setName(rstring(annotation.label))
    provenance = provenance_json(annotation)
    if provenance:
        roi.setDescription(rstring(provenance))

    for spec in annotation.shapes:
        if spec.kind == POLYGON:
            shape = omero.model.PolygonI()
            shape.points = rstring(ring_to_points(spec.points))
        elif spec.kind == POLYLINE:
            shape = omero.model.PolylineI()
            shape.points = rstring(ring_to_points(spec.points))
        elif spec.kind == POINT:
            shape = omero.model.PointI()
            shape.x = omero.rtypes.rdouble(float(spec.points[0][0]))
            shape.y = omero.rtypes.rdouble(float(spec.points[0][1]))
        else:
            continue

        red, green, blue = spec.color or (255, 255, 0)
        shape.strokeColor = rint(rgb_to_omero_color(red, green, blue))
        shape.strokeWidth = omero.model.LengthI(
            stroke_width, omero.model.enums.UnitsLength.PIXEL
        )
        if fill_alpha > 0 and spec.kind == POLYGON:
            shape.fillColor = rint(rgb_to_omero_color(red, green, blue, fill_alpha))
        if spec.label:
            shape.textValue = rstring(spec.label)
        roi.addShape(shape)

    return roi


def import_annotations(
    conn,
    image_id: int,
    annotations: list[Annotation],
    fill_alpha: int = 90,
    stroke_width: float = 2.0,
) -> ImportResult:
    """Create ROIs on an image from converted annotations.

    :param conn: a connected gateway
    :param image_id: target image
    :type image_id: int
    :param annotations: the objects to store
    :type annotations: list[Annotation]
    :param fill_alpha: 0-255, or 0 for outline only
    :type fill_alpha: int
    :param stroke_width: outline width in pixels
    :type stroke_width: float
    :raises LookupError: if the image is missing or unreadable
    :return: how many ROIs were created
    :rtype: ImportResult
    """
    image = conn.getObject("Image", image_id)
    if image is None:
        raise LookupError(f"image {image_id} not found, or you cannot read it")
    conn.SERVICE_OPTS.setOmeroGroup(image.getDetails().getGroup().getId())
    update = conn.getUpdateService()

    result = ImportResult()
    batch = []
    for annotation in annotations:
        batch.append(build_roi(image_id, annotation, fill_alpha, stroke_width))
        if len(batch) >= SAVE_BATCH:
            update.saveAndReturnArray(batch, conn.SERVICE_OPTS)
            result.created += len(batch)
            batch = []
    if batch:
        update.saveAndReturnArray(batch, conn.SERVICE_OPTS)
        result.created += len(batch)
    return result


def read_shape(shape, ellipse_segments: int, warnings: list[str]) -> ShapeSpec | None:
    """Convert one OMERO shape into a :class:`ShapeSpec`.

    Rectangles become four-point polygons and ellipses become polygons of
    ``ellipse_segments`` vertices, since GeoJSON has no curve primitive.
    Masks, labels and lines are reported and skipped.

    :param shape: an ``omero.model`` shape
    :param ellipse_segments: vertices used to approximate an ellipse
    :type ellipse_segments: int
    :param warnings: list appended to when a shape is skipped
    :type warnings: list[str]
    :return: the converted shape, or ``None``
    :rtype: ShapeSpec | None
    """
    import omero

    label = unwrap(shape.getTextValue())
    color = omero_color_to_rgb(unwrap(shape.getStrokeColor()))

    if isinstance(shape, omero.model.PolygonI):
        return ShapeSpec(
            POLYGON, points_to_ring(unwrap(shape.getPoints())), label, color
        )
    if isinstance(shape, omero.model.PolylineI):
        return ShapeSpec(
            POLYLINE, points_to_ring(unwrap(shape.getPoints())), label, color
        )
    if isinstance(shape, omero.model.PointI):
        return ShapeSpec(
            POINT, [[unwrap(shape.getX()), unwrap(shape.getY())]], label, color
        )
    if isinstance(shape, omero.model.RectangleI):
        ring = rectangle_to_ring(
            unwrap(shape.getX()),
            unwrap(shape.getY()),
            unwrap(shape.getWidth()),
            unwrap(shape.getHeight()),
        )
        return ShapeSpec(POLYGON, ring, label, color)
    if isinstance(shape, omero.model.EllipseI):
        ring = ellipse_to_ring(
            unwrap(shape.getX()),
            unwrap(shape.getY()),
            unwrap(shape.getRadiusX()),
            unwrap(shape.getRadiusY()),
            segments=ellipse_segments,
        )
        return ShapeSpec(POLYGON, ring, label, color)

    name = type(shape).__name__
    suffix = " (no GeoJSON equivalent)" if name in UNSUPPORTED_SHAPES else ""
    warnings.append(f"skipped an unsupported shape type: {name}{suffix}")
    return None


def export_image(
    conn, image_id: int, ellipse_segments: int = 64, unbridge: bool = True
) -> ExportResult:
    """Read every ROI on an image and convert them to GeoJSON features.

    :param conn: a connected gateway
    :param image_id: the image to read
    :type image_id: int
    :param ellipse_segments: vertices used to approximate an ellipse
    :type ellipse_segments: int
    :param unbridge: restore keyhole slits as real interior rings
    :type unbridge: bool
    :return: the recovered features and any warnings
    :rtype: ExportResult
    """
    result = ExportResult()
    found = conn.getRoiService().findByImage(image_id, None, conn.SERVICE_OPTS)

    for roi in found.rois:
        roi_id = unwrap(roi.getId())
        shapes = []
        for shape in roi.copyShapes():
            spec = read_shape(shape, ellipse_segments, result.warnings)
            if spec is not None:
                shapes.append(spec)
        if not shapes:
            result.warnings.append(
                f"ROI {roi_id}: every shape was an unsupported type (see above), skipped"
            )
            continue

        provenance = read_provenance(unwrap(roi.getDescription()))
        annotation = Annotation(
            shapes=shapes,
            source_id=provenance.get("id"),
            object_type=provenance.get("objectType", "annotation"),
        )
        feature = annotation_to_feature(
            annotation, fallback_id=f"omero-roi-{roi_id}", unbridge=unbridge
        )
        if feature is None:
            # Mixed shape kinds now export as a GeometryCollection (see
            # shapes_to_geometry), so reaching here means every shape in
            # this ROI was individually degenerate/unusable (e.g. a polygon
            # ring under 3 points, or a polyline under 2), not that the
            # kinds were mixed. Report exactly what was present so this is
            # diagnosable instead of a bare "skipped".
            kind_counts: dict[str, int] = {}
            for s in shapes:
                kind_counts[s.kind] = kind_counts.get(s.kind, 0) + 1
            breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(kind_counts.items()))
            point_counts = ", ".join(str(len(s.points)) for s in shapes)
            result.warnings.append(
                f"ROI {roi_id}: no usable shapes after conversion "
                f"({len(shapes)} shape(s) -- {breakdown}; point counts: {point_counts}), skipped"
            )
            continue
        result.features.append(feature)

    return result


def match_by_name(
    conn, dataset_id: int, paths: list[Path]
) -> tuple[list[tuple], list[str]]:
    """Match GeoJSON files to images in a dataset by filename stem.

    :param conn: a connected gateway
    :param dataset_id: the dataset to search
    :type dataset_id: int
    :param paths: the files to match
    :type paths: list[Path]
    :raises LookupError: if the dataset is missing or unreadable
    :return: ``([(path, image_id)], [problems])``
    :rtype: tuple[list[tuple], list[str]]
    """
    dataset = conn.getObject("Dataset", dataset_id)
    if dataset is None:
        raise LookupError(f"dataset {dataset_id} not found, or you cannot read it")

    images: dict[str, list[int]] = {}
    for image in dataset.listChildren():
        images.setdefault(Path(image.getName()).stem.lower(), []).append(image.getId())

    pairs: list[tuple] = []
    problems: list[str] = []
    for path in paths:
        stem = path.stem.lower()
        matches = images.get(stem) or [
            image_id
            for name, ids in images.items()
            if name.startswith(stem) or stem.startswith(name)
            for image_id in ids
        ]
        if not matches:
            problems.append(f"{path.name}: no image in dataset {dataset_id} matches")
        elif len(matches) > 1:
            problems.append(
                f"{path.name}: matches {len(matches)} images {matches}, ambiguous"
            )
        else:
            pairs.append((path, matches[0]))
    return pairs, problems


def iter_images(conn, image=None, dataset=None, project=None, group=None):
    """Yield the images selected by exactly one of the given identifiers.

    :param conn: a connected gateway
    :param image: an image id
    :type image: int | None
    :param dataset: a dataset id
    :type dataset: int | None
    :param project: a project id
    :type project: int | None
    :param group: a group id -- every image in it
    :type group: int | None
    :raises LookupError: if the requested object is missing or unreadable
    :raises ValueError: if not exactly one identifier was given
    """
    if image is not None:
        found = conn.getObject("Image", image)
        if found is None:
            raise LookupError(f"image {image} not found, or you cannot read it")
        yield found
    elif dataset is not None:
        found = conn.getObject("Dataset", dataset)
        if found is None:
            raise LookupError(f"dataset {dataset} not found, or you cannot read it")
        yield from found.listChildren()
    elif project is not None:
        found = conn.getObject("Project", project)
        if found is None:
            raise LookupError(f"project {project} not found, or you cannot read it")
        for child in found.listChildren():
            yield from child.listChildren()
    elif group is not None:
        # Scope the session to the group first, the same way
        # omero_client.iter_image_ids does -- otherwise the dummy group
        # (-1) would list images from every group the user belongs to.
        conn.SERVICE_OPTS.setOmeroGroup(str(group))
        yield from conn.getObjects("Image")
    else:
        raise ValueError("give exactly one of image, dataset, project or group")


__all__ = [
    "SAVE_BATCH",
    "ExportResult",
    "ImportResult",
    "build_roi",
    "export_image",
    "import_annotations",
    "iter_images",
    "match_by_name",
    "read_shape",
    "safe_filename",
    "unwrap",
]
