# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

try:
    from omero.rtypes import rdouble, rint, rlong, rstring
    from omero_model_EllipseI import EllipseI
    from omero_model_LineI import LineI
    from omero_model_PolygonI import PolygonI
    from omero_model_RectangleI import RectangleI
except Exception as exc:  # pragma: no cover
    pytest.skip(f"omero-py unavailable: {exc}", allow_module_level=True)

import lavlab.roi as roi_mod
from lavlab.roi import get_roi_mask, get_shapes_as_points, uint_to_rgba


def _pack_rgba(r, g, b, a):
    return (r << 24) | (g << 16) | (b << 8) | a


def _rectangle(id_, x, y, w, h, rgba=(255, 0, 0, 255), text=None):
    s = RectangleI()
    s.setId(rlong(id_))
    s.setX(rdouble(x))
    s.setY(rdouble(y))
    s.setWidth(rdouble(w))
    s.setHeight(rdouble(h))
    s.setStrokeColor(rint(_pack_rgba(*rgba)))
    if text is not None:
        s.setTextValue(rstring(text))
    return s


def _ellipse(id_, x, y, rx, ry, rgba=(0, 255, 0, 255), text=None):
    s = EllipseI()
    s.setId(rlong(id_))
    s.setX(rdouble(x))
    s.setY(rdouble(y))
    s.setRadiusX(rdouble(rx))
    s.setRadiusY(rdouble(ry))
    s.setStrokeColor(rint(_pack_rgba(*rgba)))
    if text is not None:
        s.setTextValue(rstring(text))
    return s


def _polygon(id_, points, rgba=(0, 0, 255, 255), text=None):
    s = PolygonI()
    s.setId(rlong(id_))
    s.setPoints(rstring(" ".join(f"{x},{y}" for x, y in points)))
    s.setStrokeColor(rint(_pack_rgba(*rgba)))
    if text is not None:
        s.setTextValue(rstring(text))
    return s


def _line(id_, x1=0, y1=0, x2=5, y2=5, rgba=(0, 0, 0, 255), text=None):
    s = LineI()
    s.setId(rlong(id_))
    s.setX1(rdouble(x1))
    s.setY1(rdouble(y1))
    s.setX2(rdouble(x2))
    s.setY2(rdouble(y2))
    s.setStrokeColor(rint(_pack_rgba(*rgba)))
    if text is not None:
        s.setTextValue(rstring(text))
    return s


class _FakeRoi:
    def __init__(self, shapes):
        self._shapes = shapes

    def copyShapes(self):
        return list(self._shapes)


class _FakeRoiServiceResult:
    def __init__(self, rois):
        self.rois = rois


class _FakeRoiService:
    def __init__(self, rois):
        self._rois = rois

    def findByImage(self, image_id, filter_, service_opts):
        return _FakeRoiServiceResult(self._rois)

    def close(self):
        pass


class _FakeConn:
    SERVICE_OPTS = None


class _FakeImage:
    def __init__(self, size_x, size_y, size_c=3, image_id=1):
        self._size_x = size_x
        self._size_y = size_y
        self._size_c = size_c
        self._id = image_id
        # get_rois() reads img._conn.SERVICE_OPTS even when an explicit
        # roi_service is passed in -- harmless in production (img is
        # always a real ImageWrapper there) but the fake needs it too.
        self._conn = _FakeConn()

    def getSizeX(self):
        return self._size_x

    def getSizeY(self):
        return self._size_y

    def getSizeC(self):
        return self._size_c

    def getId(self):
        return self._id


def test_uint_to_rgba_positive_value():
    packed = _pack_rgba(255, 128, 0, 255)
    assert uint_to_rgba(packed) == (255, 128, 0, 255)


def test_uint_to_rgba_negative_wraps_like_int32():
    packed = _pack_rgba(255, 0, 0, 255)
    signed = packed - 2**32  # how a real OMERO stroke color arrives over Ice
    assert uint_to_rgba(signed) == (255, 0, 0, 255)


def test_get_shapes_as_points_rectangle_bbox_and_color_and_no_text():
    rect = _rectangle(1, x=10, y=5, w=20, h=8, rgba=(200, 100, 50, 255))
    image = _FakeImage(size_x=100, size_y=100)
    rs = _FakeRoiService([_FakeRoi([rect])])

    shapes = get_shapes_as_points(
        image, point_downsample=1, img_downsample=1, roi_service=rs, include_all=True
    )

    assert shapes is not None
    assert len(shapes) == 1
    shape_id, rgb, text_label, points = shapes[0]
    assert shape_id == 1
    assert rgb == (200, 100, 50)
    assert text_label is None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert min(xs) >= 10 and max(xs) <= 30
    assert min(ys) >= 5 and max(ys) <= 13


def test_get_shapes_as_points_text_filter_include_exclude():
    wanted = _rectangle(1, 0, 0, 5, 5, text="tumor")
    unwanted = _rectangle(2, 0, 0, 5, 5, text="stroma")
    image = _FakeImage(size_x=50, size_y=50)
    rs = _FakeRoiService([_FakeRoi([wanted, unwanted])])

    shapes = get_shapes_as_points(
        image,
        img_downsample=1,
        roi_service=rs,
        include_all=False,
        text_filter=["tumor"],
    )

    assert shapes is not None
    assert len(shapes) == 1
    assert shapes[0][0] == 1
    assert shapes[0][2] == "tumor"


def test_get_shapes_as_points_ellipse_bbox_and_color():
    ellipse = _ellipse(1, x=50, y=40, rx=10, ry=6, rgba=(0, 200, 0, 255))
    image = _FakeImage(size_x=100, size_y=100)
    rs = _FakeRoiService([_FakeRoi([ellipse])])

    shapes = get_shapes_as_points(
        image, point_downsample=1, img_downsample=1, roi_service=rs, include_all=True
    )

    assert shapes is not None
    shape_id, rgb, text_label, points = shapes[0]
    assert shape_id == 1
    assert rgb == (0, 200, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert 39 <= min(xs) and max(xs) <= 61
    assert 33 <= min(ys) and max(ys) <= 47


def test_get_shapes_as_points_polygon_preserves_all_vertices():
    # Regression test: polygon vertices are the actual points the user
    # drew (already sparse), unlike Rectangle/Ellipse's dense perimeter
    # trace -- point_downsample must NOT thin them.
    verts = [(0, 0), (5, 0), (5, 5), (3, 3), (0, 5), (2, 2), (1, 4), (4, 1)]
    poly = _polygon(1, verts)
    image = _FakeImage(size_x=50, size_y=50)
    rs = _FakeRoiService([_FakeRoi([poly])])

    shapes = get_shapes_as_points(
        image, point_downsample=4, img_downsample=1, roi_service=rs, include_all=True
    )

    assert shapes is not None
    _, _, _, points = shapes[0]
    assert points == [(float(x), float(y)) for x, y in verts]


def test_get_shapes_as_points_rectangle_perimeter_is_downsampled():
    rect_full = _rectangle(1, x=0, y=0, w=40, h=40)
    image = _FakeImage(size_x=100, size_y=100)

    full = get_shapes_as_points(
        image,
        point_downsample=1,
        img_downsample=1,
        roi_service=_FakeRoiService([_FakeRoi([rect_full])]),
        include_all=True,
    )
    thinned = get_shapes_as_points(
        image,
        point_downsample=4,
        img_downsample=1,
        roi_service=_FakeRoiService([_FakeRoi([rect_full])]),
        include_all=True,
    )

    full_points = full[0][3]
    thinned_points = thinned[0][3]
    assert len(thinned_points) < len(full_points)
    assert thinned_points == full_points[::4]


def test_get_shapes_as_points_unsupported_type_warns_and_skips(caplog):
    rect = _rectangle(1, 0, 0, 5, 5)
    line = _line(2)
    image = _FakeImage(size_x=50, size_y=50)
    rs = _FakeRoiService([_FakeRoi([rect, line])])

    with caplog.at_level("WARNING"):
        shapes = get_shapes_as_points(
            image, img_downsample=1, roi_service=rs, include_all=True
        )

    assert shapes is not None
    assert [s[0] for s in shapes] == [1]
    assert any(
        "unsupported shape type" in r.message and "LineI" in r.message
        for r in caplog.records
    )


def test_get_shapes_as_points_no_shapes_returns_none():
    image = _FakeImage(size_x=50, size_y=50)
    rs = _FakeRoiService([])
    assert get_shapes_as_points(image, roi_service=rs, include_all=True) is None


def test_get_roi_mask_rgb_fills_rectangle_region(monkeypatch):
    rect = _rectangle(1, x=5, y=5, w=10, h=10, rgba=(10, 20, 30, 255))
    image = _FakeImage(size_x=20, size_y=20, size_c=3)
    monkeypatch.setattr(
        roi_mod, "get_rois", lambda img, roi_service=None: [_FakeRoi([rect])]
    )

    mask, shape_count = get_roi_mask(
        image, downsample=1, include_all=True, text_filter=[], point_downsample=1
    )

    assert mask.shape == (20, 20, 3)
    assert tuple(mask[10, 10]) == (10, 20, 30)
    assert tuple(mask[1, 1]) == (255, 255, 255)
    assert shape_count == 1


def test_get_roi_mask_reports_count_matching_number_of_shapes(monkeypatch):
    shapes = [_rectangle(i, x=i, y=i, w=2, h=2) for i in range(5)]
    image = _FakeImage(size_x=50, size_y=50)
    monkeypatch.setattr(
        roi_mod, "get_rois", lambda img, roi_service=None: [_FakeRoi(shapes)]
    )

    _, shape_count = get_roi_mask(image, downsample=1, include_all=True, text_filter=[])

    assert shape_count == 5


def test_get_roi_mask_count_excludes_unsupported_shapes(monkeypatch):
    rect = _rectangle(1, x=0, y=0, w=5, h=5)
    line = _line(2)
    image = _FakeImage(size_x=50, size_y=50)
    monkeypatch.setattr(
        roi_mod, "get_rois", lambda img, roi_service=None: [_FakeRoi([rect, line])]
    )

    _, shape_count = get_roi_mask(image, downsample=1, include_all=True, text_filter=[])

    assert shape_count == 1


def test_get_roi_mask_palette_numbers_by_text_filter_order(monkeypatch):
    a = _rectangle(1, x=0, y=0, w=5, h=5, text="a")
    b = _rectangle(2, x=10, y=10, w=5, h=5, text="b")
    image = _FakeImage(size_x=20, size_y=20)
    monkeypatch.setattr(
        roi_mod, "get_rois", lambda img, roi_service=None: [_FakeRoi([a, b])]
    )

    mask, shape_count = get_roi_mask(
        image,
        downsample=1,
        include_all=False,
        text_filter=["a", "b"],
        palette=True,
        point_downsample=1,
    )

    assert mask[2, 2] == 1
    assert mask[12, 12] == 2
    assert mask[19, 19] == 0
    assert shape_count == 2


def test_get_roi_mask_no_shapes_returns_empty_array_and_zero_count(monkeypatch):
    image = _FakeImage(size_x=20, size_y=20)
    monkeypatch.setattr(roi_mod, "get_rois", lambda img, roi_service=None: [])

    mask, shape_count = get_roi_mask(
        image, downsample=1, include_all=True, text_filter=[]
    )

    assert mask.size == 0
    assert shape_count == 0


# ---------- upload / existence check ----------


class _FakeOriginalFile:
    def __init__(self, name):
        self._name = name

    def getName(self):
        return self._name


class _FakeFileAnnotation:
    def __init__(self, name):
        self._file = _FakeOriginalFile(name)
        self._obj = object()

    def getFile(self):
        return self._file


class _FakeUploadImage:
    def __init__(self, name="N101_S06_HE.ome.tiff", annotations=None):
        self._name = name
        self._annotations = list(annotations or [])
        self.removed = []
        self.linked = []

    def getId(self):
        return 362

    def getName(self):
        return self._name

    def listAnnotations(self, ns=None):
        return list(self._annotations)

    def removeAnnotations(self, anns):
        self.removed.extend(anns)

    def linkAnnotation(self, ann):
        self.linked.append(ann)


class _FakeUploadConn:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

    def deleteObject(self, obj):
        self.deleted.append(obj)

    def createFileAnnfromLocalFile(
        self, path, origFilePathAndName=None, mimetype=None, ns=None
    ):
        self.uploaded.append((path, origFilePathAndName, mimetype, ns))
        return _FakeFileAnnotation(origFilePathAndName or path)


def test_roi_namespace_matches_legacy_convention():
    assert roi_mod.roi_namespace(10) == "LargeRecon.10.roi"
    assert roi_mod.roi_namespace(8) == "LargeRecon.8.roi"


def test_mask_annotation_name_includes_suffix():
    name = roi_mod.mask_annotation_name("N101_S06_HE.ome.tiff", 10, "_annot", "jp2")
    assert name == "LR10_N101_S06_HE__annot.jp2"


def test_has_uploaded_mask_true_when_present():
    image = _FakeUploadImage(
        annotations=[_FakeFileAnnotation("LR10_N101_S06_HE__annot.jp2")]
    )
    assert roi_mod.has_uploaded_mask(image, 10, "_annot", "jp2") is True


def test_has_uploaded_mask_false_when_absent():
    assert roi_mod.has_uploaded_mask(_FakeUploadImage(), 10, "_annot", "jp2") is False


def test_has_uploaded_mask_is_suffix_specific():
    # _annot and _exclude masks share one namespace -- an _exclude upload
    # must not count as an _annot already being done.
    image = _FakeUploadImage(
        annotations=[_FakeFileAnnotation("LR10_N101_S06_HE__exclude.jp2")]
    )
    assert roi_mod.has_uploaded_mask(image, 10, "_exclude", "jp2") is True
    assert roi_mod.has_uploaded_mask(image, 10, "_annot", "jp2") is False


def test_upload_mask_uses_canonical_name_and_namespace(tmp_path):
    local = tmp_path / "whatever_local_name.jp2"
    local.write_bytes(b"mask")
    image = _FakeUploadImage()
    conn = _FakeUploadConn()

    remote_name = roi_mod.upload_mask(conn, image, str(local), 10, "_annot", "jp2")

    assert remote_name == "LR10_N101_S06_HE__annot.jp2"
    assert conn.uploaded == [
        (str(local), "LR10_N101_S06_HE__annot.jp2", "image/jp2", "LargeRecon.10.roi")
    ]
    assert len(image.linked) == 1
    assert conn.deleted == []


def test_upload_mask_replaces_only_the_same_mask(tmp_path):
    local = tmp_path / "m.jp2"
    local.write_bytes(b"mask")
    stale = _FakeFileAnnotation("LR10_N101_S06_HE__annot.jp2")
    other = _FakeFileAnnotation("LR10_N101_S06_HE__exclude.jp2")
    image = _FakeUploadImage(annotations=[stale, other])
    conn = _FakeUploadConn()

    roi_mod.upload_mask(conn, image, str(local), 10, "_annot", "jp2")

    assert image.removed == [stale]
    assert conn.deleted == [stale._obj]


def test_roi_and_lr_annotations_do_not_collide():
    """roi masks and lr recons share a namespace *prefix* but not a
    namespace -- OMERO filters by exact ns equality, so neither command's
    lookup can pick up the other's attachment. lr matches on extension
    alone, so the namespace is its only protection; this pins that down."""
    from lavlab.large_recon import has_cached_recon

    class _F:
        def __init__(self, name):
            self._name = name

        def getName(self):
            return self._name

    class _Ann:
        def __init__(self, name, ns):
            self._f = _F(name)
            self.ns = ns
            self._obj = object()

        def getFile(self):
            return self._f

    class _Image:
        # mirrors omero.gateway's real filter: exact ns equality
        def __init__(self, anns):
            self._anns = anns

        def getName(self):
            return "N101_S06_HE.ome.tiff"

        def listAnnotations(self, ns=None):
            return [a for a in self._anns if ns is None or a.ns == ns]

    lr_ann = _Ann("LR10_N101_S06_HE.jp2", "LargeRecon.10")
    roi_ann = _Ann("LR10_N101_S06_HE__annot.jp2", "LargeRecon.10.roi")

    both = _Image([lr_ann, roi_ann])
    assert has_cached_recon(both, 10, "jp2") is True
    assert roi_mod.has_uploaded_mask(both, 10, "_annot", "jp2") is True

    # Neither may fall back to the other's attachment when its own is absent.
    assert has_cached_recon(_Image([roi_ann]), 10, "jp2") is False
    assert roi_mod.has_uploaded_mask(_Image([lr_ann]), 10, "_annot", "jp2") is False
