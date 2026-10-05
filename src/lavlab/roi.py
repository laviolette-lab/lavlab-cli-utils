# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""ROI shape gathering and rasterization into either an RGB color mask or a
single-channel palette (label) mask, plus uploading the result back to OMERO
as a file annotation.

Rasterization uses scikit-image instead of OpenCV per project conventions.
"""

from __future__ import annotations

import logging

import numpy as np
from omero_model_EllipseI import EllipseI
from omero_model_PolygonI import PolygonI
from omero_model_RectangleI import RectangleI
from skimage import draw

from lavlab.naming import build_filename

log = logging.getLogger(__name__)

#: Namespace uploaded ROI masks live under, matching what legacy
#: batch_roi.py/single_roi.py used ("LargeRecon.10.roi") so masks this tool
#: uploads land alongside -- and are found by -- anything already there.
_ROI_NAMESPACE_FMT = "LargeRecon.{}.roi"

#: Deliberately a small local copy rather than a shared import from
#: lavlab.large_recon: that module's upload path is live-tested against the
#: production server, and roi's namespace/filename rules differ from its
#: (masks match on full filename, since _annot and _exclude variants share
#: one namespace). Worth consolidating if a third caller ever appears.
_MIMETYPES = {
    "jp2": "image/jp2",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}


def roi_namespace(downsample: int) -> str:
    """Return the OMERO annotation namespace ROI masks are stored under.

    :param downsample: downsample factor the mask was rendered at
    :type downsample: int
    :return: e.g. ``LargeRecon.10.roi``
    :rtype: str
    """
    return _ROI_NAMESPACE_FMT.format(downsample)


def mask_annotation_name(
    image_name: str, downsample: int, suffix: str | None, ext: str = "jp2"
) -> str:
    """Return the canonical filename an uploaded mask is stored under.

    Same shape as the local filename (``LR10_<stem>_annot.jp2``), so a mask
    uploaded from one machine is recognisable as the same thing generated on
    another regardless of where either wrote it locally.

    :param image_name: the OMERO image's name
    :type image_name: str
    :param downsample: downsample factor
    :type downsample: int
    :param suffix: filename suffix, e.g. ``_annot``
    :type suffix: str | None
    :param ext: file extension
    :type ext: str
    :return: the canonical remote filename
    :rtype: str
    """
    return build_filename(downsample, image_name, suffix, ext)


def _find_mask_annotation(image, namespace: str, remote_name: str):
    """Return the file annotation in *namespace* named exactly *remote_name*.

    Matches the whole filename rather than just the extension, because one
    namespace legitimately holds several masks for the same image -- an
    ``_annot`` and an ``_exclude`` differ only by suffix.
    """
    for ann in image.listAnnotations(ns=namespace):
        if not hasattr(ann, "getFile"):
            continue
        f = ann.getFile()
        if f is not None and f.getName() == remote_name:
            return ann
    return None


def has_uploaded_mask(
    image, downsample: int, suffix: str | None, ext: str = "jp2"
) -> bool:
    """Return True if this exact mask is already attached to *image*.

    A cheap existence check -- it lists annotations, it never downloads --
    so a caller can skip images that are already done.

    :param image: an OMERO ``ImageWrapper``
    :param downsample: downsample factor
    :type downsample: int
    :param suffix: filename suffix, e.g. ``_annot``
    :type suffix: str | None
    :param ext: file extension
    :type ext: str
    :return: whether a matching mask annotation exists
    :rtype: bool
    """
    remote_name = mask_annotation_name(image.getName(), downsample, suffix, ext)
    return (
        _find_mask_annotation(image, roi_namespace(downsample), remote_name) is not None
    )


def upload_mask(
    conn,
    image,
    local_path: str,
    downsample: int,
    suffix: str | None,
    ext: str = "jp2",
) -> str:
    """Attach *local_path* to *image* as its ROI mask, replacing any previous
    upload of the same mask.

    Only an annotation with this exact canonical filename is replaced -- a
    different mask of the same image (a different ``suffix``) or an
    unrelated attachment sharing the namespace is left alone.

    :param conn: a connected gateway, already switched into the image's group
    :param image: an OMERO ``ImageWrapper``
    :param local_path: the rendered mask file to upload
    :type local_path: str
    :param downsample: downsample factor
    :type downsample: int
    :param suffix: filename suffix, e.g. ``_annot``
    :type suffix: str | None
    :param ext: file extension
    :type ext: str
    :return: the canonical filename it was uploaded as
    :rtype: str
    """
    namespace = roi_namespace(downsample)
    remote_name = mask_annotation_name(image.getName(), downsample, suffix, ext)

    stale = _find_mask_annotation(image, namespace, remote_name)
    if stale is not None:
        image.removeAnnotations([stale])
        conn.deleteObject(stale._obj)

    file_ann = conn.createFileAnnfromLocalFile(
        local_path,
        origFilePathAndName=remote_name,
        mimetype=_MIMETYPES.get(ext, "application/octet-stream"),
        ns=namespace,
    )
    image.linkAnnotation(file_ann)
    return remote_name


def _rectangle_perimeter(
    start: tuple[float, float],
    end: tuple[float, float],
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-free replacement for ``skimage.draw.rectangle_perimeter``.

    The installed skimage version gates that function behind a matplotlib
    requirement (an undeclared dependency this project deliberately avoids
    bundling -- see the JP2/libvips history in imaging.py), so it isn't
    available. Returns ``(rr, cc)`` walking the axis-aligned rectangle's
    boundary in a continuous clockwise order (needed by ``draw.polygon``
    downstream, which fills whatever path its input traces -- an unordered
    boundary point cloud would fill incorrectly), clipped to *shape*.
    """
    r0, r1 = sorted((int(round(start[0])), int(round(end[0]))))
    c0, c1 = sorted((int(round(start[1])), int(round(end[1]))))

    top_cols = np.arange(c0, c1 + 1)
    right_rows = np.arange(r0, r1 + 1)
    bottom_cols = np.arange(c1, c0 - 1, -1)
    left_rows = np.arange(r1, r0 - 1, -1)

    rr = np.concatenate(
        [
            np.full(top_cols.shape, r0),
            right_rows,
            np.full(bottom_cols.shape, r1),
            left_rows,
        ]
    )
    cc = np.concatenate(
        [
            top_cols,
            np.full(right_rows.shape, c1),
            bottom_cols,
            np.full(left_rows.shape, c0),
        ]
    )

    if shape is not None:
        valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        rr, cc = rr[valid], cc[valid]

    return rr, cc


def uint_to_rgba(uint: int) -> tuple[int, int, int, int]:
    """Convert OMERO's packed signed-32-bit RGBA color integer to (r, g, b, a)."""
    if uint < 0:
        uint = uint + 2**32

    red = (uint >> 24) & 0xFF
    green = (uint >> 16) & 0xFF
    blue = (uint >> 8) & 0xFF
    alpha = uint & 0xFF

    return red, green, blue, alpha


def get_rois(img, roi_service=None):
    """Gather OMERO RoiI objects for an image."""
    close_roi = roi_service is None
    if roi_service is None:
        roi_service = img._conn.getRoiService()

    rois = roi_service.findByImage(img.getId(), None, img._conn.SERVICE_OPTS).rois

    if close_roi:
        roi_service.close()

    return rois


def shape_roi_ids(img, roi_service=None) -> dict[int, int]:
    """Map every shape id on an image to the id of the ROI that holds it.

    ``get_shapes_as_points`` reports shape ids; a QuPath import usually has
    one shape per ROI, but OMERO allows several, and datasets are grouped by
    ROI.
    """
    mapping: dict[int, int] = {}
    for roi in get_rois(img, roi_service):
        roi_id = roi.getId()._val
        for shape in roi.copyShapes():
            mapping[shape.getId()._val] = roi_id
    return mapping


def get_shapes_as_points(
    img,
    point_downsample: int = 4,
    img_downsample: int = 1,
    roi_service=None,
    include_all: bool = False,
    text_filter: list[str] | None = None,
) -> (
    list[tuple[int, tuple[int, int, int], str | None, list[tuple[float, float]]]] | None
):
    """Gather Rectangles, Polygons, and Ellipses as shape id, RGB color, matched
    text label (lowercased, or None), and a list of (x, y) boundary points.

    Selection rules:
      - include_all=True: every shape is included.
      - otherwise a shape is only included if its textValue (lowercased)
        is present in text_filter.
    """
    text_filter = text_filter or []

    size_x = img.getSizeX() / img_downsample
    size_y = img.getSizeY() / img_downsample
    yx_shape = (size_y, size_x)

    shapes = []
    for roi in get_rois(img, roi_service):
        for shape in roi.copyShapes():
            text_value = shape.getTextValue()
            text_label = (
                text_value.getValue().lower() if text_value is not None else None
            )

            if not include_all:
                if text_label is None or text_label not in text_filter:
                    continue

            points = None

            if type(shape) is RectangleI:
                x = float(shape.getX().getValue()) / img_downsample
                y = float(shape.getY().getValue()) / img_downsample
                w = float(shape.getWidth().getValue()) / img_downsample
                h = float(shape.getHeight().getValue()) / img_downsample
                points = _rectangle_perimeter((y, x), (y + h, x + w), shape=yx_shape)
                points = [(points[1][i], points[0][i]) for i in range(len(points[0]))]
                # A dense pixel-by-pixel perimeter trace -- thinning it is harmless.
                points = points[::point_downsample]

            elif type(shape) is EllipseI:
                # draw.ellipse_perimeter's Cython implementation requires
                # ints, not floats -- passing floats raises TypeError.
                points = draw.ellipse_perimeter(
                    round(shape._y._val / img_downsample),
                    round(shape._x._val / img_downsample),
                    round(shape._radiusY._val / img_downsample),
                    round(shape._radiusX._val / img_downsample),
                    shape=yx_shape,
                )
                points = [(points[1][i], points[0][i]) for i in range(len(points[0]))]
                # Same as Rectangle: a dense perimeter trace, safe to thin.
                points = points[::point_downsample]

            elif type(shape) is PolygonI:
                point_str_arr = shape.getPoints()._val.split(" ")
                xy = []
                for coord_str in point_str_arr:
                    coord_list = coord_str.split(",")
                    xy.append(
                        (
                            float(coord_list[0]) / img_downsample,
                            float(coord_list[1]) / img_downsample,
                        )
                    )
                if xy:
                    points = xy
                # Unlike Rectangle/Ellipse, these are the actual vertices the
                # user drew (typically already sparse) -- applying
                # point_downsample here would visibly distort the shape, so
                # they're kept in full.

            else:
                log.warning(
                    "Shape %d: unsupported shape type %s, skipping.",
                    shape.getId()._val,
                    type(shape).__name__,
                )

            if points is not None:
                color_val = shape.getStrokeColor()._val
                rgb = uint_to_rgba(color_val)[:-1]  # ignore alpha
                shapes.append((shape.getId()._val, rgb, text_label, points))

    if not shapes:
        return None

    return sorted(shapes)


def _fill_polygon(
    mask: np.ndarray, points_xy: list[tuple[float, float]], value
) -> None:
    if len(points_xy) < 3:
        return
    xs = np.array([p[0] for p in points_xy])
    ys = np.array([p[1] for p in points_xy])
    rr, cc = draw.polygon(ys, xs, shape=mask.shape[:2])
    mask[rr, cc] = value


def get_roi_mask(
    image,
    downsample: int,
    include_all: bool,
    text_filter: list[str],
    palette: bool = False,
    point_downsample: int = 4,
) -> tuple[np.ndarray, int]:
    """Render ROI shapes into an image mask.

    Returns ``(mask, shape_count)``: an (H, W, C) RGB mask by default, or an
    (H, W) single-channel label mask (background 0, shapes numbered 1..N in
    text_filter order) when palette=True, plus how many shapes were actually
    rendered into it -- so callers can confirm nothing was silently dropped
    (see the "unsupported shape type" warning in get_shapes_as_points).
    """
    height = int(image.getSizeY() / downsample)
    width = int(image.getSizeX() / downsample)

    if palette:
        mask = np.zeros((height, width), dtype=np.uint8)
    else:
        mask = np.zeros((height, width, image.getSizeC()), dtype=np.uint8)
        mask[:] = 255

    shapes = get_shapes_as_points(
        image,
        point_downsample=point_downsample,
        img_downsample=downsample,
        include_all=include_all,
        text_filter=text_filter,
    )
    if shapes is None:
        return np.array([]), 0

    rendered = 0
    for _shape_id, rgb, text_label, points in shapes:
        if palette:
            if text_label not in text_filter:
                continue
            _fill_polygon(mask, points, text_filter.index(text_label) + 1)
        else:
            _fill_polygon(mask, points, rgb)
        rendered += 1

    return mask, rendered
