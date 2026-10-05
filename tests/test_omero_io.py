# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import omero.gateway
    import omero.model
    from omero.rtypes import rint, rlong, rstring
    from omero_model_LineI import LineI
    from omero_model_PolygonI import PolygonI
except Exception as exc:  # pragma: no cover
    pytest.skip(f"omero-py unavailable: {exc}", allow_module_level=True)

from lavlab.geojson.geojson_io import POINT, POLYGON, POLYLINE, Annotation, ShapeSpec
from lavlab.geojson.omero_io import (
    SAVE_BATCH,
    build_roi,
    export_image,
    import_annotations,
    iter_images,
    match_by_name,
    safe_filename,
    unwrap,
)


def _pack_rgba(r, g, b, a=255):
    return (r << 24) | (g << 16) | (b << 8) | a


def _polygon_shape(points_str, text=None):
    s = PolygonI()
    s.setPoints(rstring(points_str))
    if text is not None:
        s.setTextValue(rstring(text))
    s.setStrokeColor(rint(_pack_rgba(255, 0, 0)))
    return s


def _make_roi(roi_id, shapes, description=None):
    roi = omero.model.RoiI()
    roi.setId(rlong(roi_id))
    if description is not None:
        roi.setDescription(rstring(description))
    for s in shapes:
        roi.addShape(s)
    return roi


# ---------- unwrap / safe_filename ----------


def test_unwrap_none():
    assert unwrap(None) is None


def test_unwrap_rtype():
    assert unwrap(rstring("hello")) == "hello"
    assert unwrap(rlong(5)) == 5


def test_safe_filename_normal_name_passes_through():
    assert safe_filename("N101_S06_HE.ome.tiff") == "N101_S06_HE.ome.tiff"


def test_safe_filename_replaces_unsafe_characters():
    assert safe_filename("weird name/with:chars?") == "weird_name_with_chars"


def test_safe_filename_empty_or_none_falls_back():
    assert safe_filename("") == "image"
    assert safe_filename(None) == "image"


# ---------- build_roi ----------


def test_build_roi_polygon_shape():
    annotation = Annotation(
        shapes=[
            ShapeSpec(
                POLYGON,
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                label="tumor",
                color=(200, 50, 50),
            )
        ],
        source_id="abc-123",
    )
    roi = build_roi(image_id=362, annotation=annotation)

    assert roi.getImage().getId().getValue() == 362
    assert roi.getName().getValue() == "tumor"
    assert "abc-123" in roi.getDescription().getValue()
    shapes = roi.copyShapes()
    assert len(shapes) == 1
    shape = shapes[0]
    assert isinstance(shape, omero.model.PolygonI)
    assert shape.getTextValue().getValue() == "tumor"
    assert shape.getFillColor() is not None  # default fill_alpha=90


def test_build_roi_fill_alpha_zero_means_outline_only():
    annotation = Annotation(
        shapes=[ShapeSpec(POLYGON, [[0, 0], [10, 0], [10, 10], [0, 10]])]
    )
    roi = build_roi(image_id=1, annotation=annotation, fill_alpha=0)
    shape = roi.copyShapes()[0]
    assert shape.getFillColor() is None


def test_build_roi_mixed_kinds_produces_multiple_shape_types():
    # The core scenario from the round-trip fix: one ROI, several kinds.
    annotation = Annotation(
        shapes=[
            ShapeSpec(POLYGON, [[0, 0], [10, 0], [10, 10], [0, 10]], label="tumor"),
            ShapeSpec(POINT, [[5, 5]], label="marker"),
            ShapeSpec(POLYLINE, [[0, 0], [1, 1], [2, 0]], label="line"),
        ]
    )
    roi = build_roi(image_id=1, annotation=annotation)
    shapes = roi.copyShapes()

    kinds = sorted(type(s).__name__ for s in shapes)
    assert kinds == ["PointI", "PolygonI", "PolylineI"]
    # Only the polygon shape gets a fillColor, per build_roi's own rule.
    fills = [s.getFillColor() for s in shapes]
    assert sum(1 for f in fills if f is not None) == 1


def test_build_roi_no_provenance_no_description():
    annotation = Annotation(
        shapes=[ShapeSpec(POLYGON, [[0, 0], [10, 0], [10, 10], [0, 10]])]
    )
    roi = build_roi(image_id=1, annotation=annotation)
    assert roi.getDescription() is None


# ---------- import_annotations ----------


class _FakeGroupId:
    def __init__(self, val):
        self._val = val

    def getId(self):
        return self._val


class _FakeDetails:
    def __init__(self, group_id):
        self._group_id = group_id

    def getGroup(self):
        return SimpleNamespace(getId=lambda: self._group_id)


class _FakeImage:
    def __init__(self, image_id, name="img.ome.tiff", group_id=1):
        self._id = image_id
        self._name = name
        self._details = _FakeDetails(group_id)

    def getId(self):
        return self._id

    def getName(self):
        return self._name

    def getDetails(self):
        return self._details


class _FakeServiceOpts:
    def __init__(self):
        self.group_calls = []

    def setOmeroGroup(self, group_id):
        self.group_calls.append(group_id)


class _FakeUpdateService:
    def __init__(self):
        self.batches = []

    def saveAndReturnArray(self, batch, service_opts):
        self.batches.append(list(batch))
        return batch


class _FakeConnForImport:
    def __init__(self, image):
        self._image = image
        self.SERVICE_OPTS = _FakeServiceOpts()
        self._update = _FakeUpdateService()

    def getObject(self, type_, image_id):
        return self._image if self._image and self._image.getId() == image_id else None

    def getUpdateService(self):
        return self._update


def _polygon_annotation():
    return Annotation(shapes=[ShapeSpec(POLYGON, [[0, 0], [1, 0], [1, 1]])])


def test_import_annotations_image_not_found_raises():
    conn = _FakeConnForImport(image=None)
    with pytest.raises(LookupError):
        import_annotations(conn, 999, [])


def test_import_annotations_creates_and_reports_count():
    conn = _FakeConnForImport(_FakeImage(362))
    result = import_annotations(conn, 362, [_polygon_annotation() for _ in range(3)])

    assert result.created == 3
    assert len(conn._update.batches) == 1
    assert len(conn._update.batches[0]) == 3
    assert conn.SERVICE_OPTS.group_calls == [1]


def test_import_annotations_batches_at_save_batch_boundary():
    conn = _FakeConnForImport(_FakeImage(362))
    n = SAVE_BATCH + 5
    result = import_annotations(conn, 362, [_polygon_annotation() for _ in range(n)])

    assert result.created == n
    assert len(conn._update.batches) == 2
    assert len(conn._update.batches[0]) == SAVE_BATCH
    assert len(conn._update.batches[1]) == 5


# ---------- export_image ----------


class _FakeRoiServiceResult:
    def __init__(self, rois):
        self.rois = rois


class _FakeRoiService:
    def __init__(self, rois):
        self._rois = rois

    def findByImage(self, image_id, filter_, service_opts):
        return _FakeRoiServiceResult(self._rois)


class _FakeConnForExport:
    def __init__(self, rois):
        self._rois = rois
        self.SERVICE_OPTS = object()

    def getRoiService(self):
        return _FakeRoiService(self._rois)


def test_export_image_basic_polygon_roi():
    shape = _polygon_shape("0,0 10,0 10,10 0,10")
    conn = _FakeConnForExport([_make_roi(1, [shape])])

    result = export_image(conn, image_id=362)

    assert len(result.features) == 1
    assert result.features[0]["geometry"]["type"] == "Polygon"
    assert not result.warnings


def test_export_image_all_unsupported_shapes_warns_with_roi_id():
    # LineI is in UNSUPPORTED_SHAPES -- no GeoJSON equivalent.
    conn = _FakeConnForExport([_make_roi(42, [LineI()])])

    result = export_image(conn, image_id=362)

    assert result.features == []
    assert any("ROI 42" in w and "unsupported type" in w for w in result.warnings)


def test_export_image_degenerate_polygon_warns_no_usable_shapes():
    # Only 2 points -- below the 3-point minimum for a ring.
    shape = _polygon_shape("0,0 1,1")
    conn = _FakeConnForExport([_make_roi(99, [shape])])

    result = export_image(conn, image_id=362)

    assert result.features == []
    assert any(
        "ROI 99" in w and "no usable shapes after conversion" in w
        for w in result.warnings
    )


def test_export_image_seam_duplicate_polygon_exports_successfully():
    # Regression test for the real production bug (image 362, ROI 68078):
    # a ring whose closing point was redundantly duplicated at both ends
    # used to make find_bridge() treat the whole ring as a bridged hole
    # and drop the entire ROI. Confirmed end-to-end through export_image,
    # not just the lower-level unbridge_ring() unit test.
    points = "0,0 0,0 10,0 10,10 0,10 5,10 0,0 0,0 0,0"
    shape = _polygon_shape(points)
    conn = _FakeConnForExport([_make_roi(68078, [shape])])

    result = export_image(conn, image_id=362)

    assert len(result.features) == 1
    assert not result.warnings


def test_export_image_multiple_rois():
    shape1 = _polygon_shape("0,0 10,0 10,10 0,10")
    shape2 = _polygon_shape("20,20 30,20 30,30 20,30")
    conn = _FakeConnForExport([_make_roi(1, [shape1]), _make_roi(2, [shape2])])

    result = export_image(conn, image_id=362)

    assert len(result.features) == 2


# ---------- match_by_name ----------


class _FakeDataset:
    def __init__(self, images):
        self._images = images

    def listChildren(self):
        return self._images


class _FakeConnForMatch:
    def __init__(self, dataset=None):
        self._dataset = dataset

    def getObject(self, type_, dataset_id):
        return self._dataset


def test_match_by_name_dataset_not_found_raises():
    conn = _FakeConnForMatch(dataset=None)
    with pytest.raises(LookupError):
        match_by_name(conn, 1, [Path("foo.geojson")])


def test_match_by_name_exact_stem_match():
    conn = _FakeConnForMatch(_FakeDataset([_FakeImage(1, name="foo.ome.tiff")]))
    pairs, problems = match_by_name(conn, 1, [Path("foo.geojson")])
    assert pairs == [(Path("foo.geojson"), 1)]
    assert problems == []


def test_match_by_name_no_match_records_problem():
    conn = _FakeConnForMatch(_FakeDataset([_FakeImage(1, name="bar.ome.tiff")]))
    pairs, problems = match_by_name(conn, 1, [Path("foo.geojson")])
    assert pairs == []
    assert len(problems) == 1
    assert "no image" in problems[0]


def test_match_by_name_ambiguous_prefix_match_records_problem():
    images = [_FakeImage(1, name="sampleA.tiff"), _FakeImage(2, name="sampleAB.tiff")]
    conn = _FakeConnForMatch(_FakeDataset(images))
    pairs, problems = match_by_name(conn, 1, [Path("sample.geojson")])
    assert pairs == []
    assert len(problems) == 1
    assert "ambiguous" in problems[0]


# ---------- iter_images ----------


class _FakeConnForIter:
    def __init__(self, images=None, datasets=None, projects=None):
        self._images = images or {}
        self._datasets = datasets or {}
        self._projects = projects or {}

    def getObject(self, type_, id_):
        return {
            "Image": self._images,
            "Dataset": self._datasets,
            "Project": self._projects,
        }[type_].get(id_)


def test_iter_images_by_image_id():
    img = _FakeImage(1)
    conn = _FakeConnForIter(images={1: img})
    assert list(iter_images(conn, image=1)) == [img]


def test_iter_images_image_not_found_raises():
    conn = _FakeConnForIter()
    with pytest.raises(LookupError):
        list(iter_images(conn, image=999))


def test_iter_images_by_dataset():
    img1, img2 = _FakeImage(1), _FakeImage(2)
    dataset = SimpleNamespace(listChildren=lambda: [img1, img2])
    conn = _FakeConnForIter(datasets={5: dataset})
    assert list(iter_images(conn, dataset=5)) == [img1, img2]


def test_iter_images_by_project_flattens_datasets():
    img1, img2 = _FakeImage(1), _FakeImage(2)
    ds1 = SimpleNamespace(listChildren=lambda: [img1])
    ds2 = SimpleNamespace(listChildren=lambda: [img2])
    project = SimpleNamespace(listChildren=lambda: [ds1, ds2])
    conn = _FakeConnForIter(projects={9: project})
    assert list(iter_images(conn, project=9)) == [img1, img2]


def test_iter_images_no_identifier_raises_value_error():
    conn = _FakeConnForIter()
    with pytest.raises(ValueError):
        list(iter_images(conn))


# ---------- geojson upload / existence check ----------


class _FakeGeojsonAnn:
    def __init__(self, name, ns):
        self._name = name
        self.ns = ns
        self._obj = object()

    def getFile(self):
        return SimpleNamespace(getName=lambda: self._name)


class _FakeGeojsonImage:
    def __init__(self, name="N101_S06_HE.ome.tiff", image_id=362, anns=None):
        self._name = name
        self._id = image_id
        self._anns = list(anns or [])
        self.removed = []
        self.linked = []

    def getId(self):
        return self._id

    def getName(self):
        return self._name

    def listAnnotations(self, ns=None):
        # mirrors omero.gateway's real filter: exact ns equality
        return [a for a in self._anns if ns is None or a.ns == ns]

    def removeAnnotations(self, anns):
        self.removed.extend(anns)

    def linkAnnotation(self, ann):
        self.linked.append(ann)


class _FakeGeojsonConn:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

    def deleteObject(self, obj):
        self.deleted.append(obj)

    def createFileAnnfromLocalFile(
        self, path, origFilePathAndName=None, mimetype=None, ns=None
    ):
        self.uploaded.append((path, origFilePathAndName, mimetype, ns))
        return _FakeGeojsonAnn(origFilePathAndName, ns)


def test_geojson_annotation_name_matches_local_export_filename():
    from lavlab.geojson.omero_io import geojson_annotation_name

    name = geojson_annotation_name(_FakeGeojsonImage())
    assert name == "N101_S06_HE.ome.tiff__omero-362.geojson"


def test_has_uploaded_geojson_true_and_false():
    from lavlab.geojson.omero_io import GEOJSON_NAMESPACE, has_uploaded_geojson

    present = _FakeGeojsonImage(
        anns=[
            _FakeGeojsonAnn(
                "N101_S06_HE.ome.tiff__omero-362.geojson", GEOJSON_NAMESPACE
            )
        ]
    )
    assert has_uploaded_geojson(present) is True
    assert has_uploaded_geojson(_FakeGeojsonImage()) is False


def test_upload_geojson_uses_namespace_and_mimetype(tmp_path):
    from lavlab.geojson.omero_io import (
        GEOJSON_MIMETYPE,
        GEOJSON_NAMESPACE,
        upload_geojson,
    )

    local = tmp_path / "whatever.geojson"
    local.write_text("{}", encoding="utf-8")
    image = _FakeGeojsonImage()
    conn = _FakeGeojsonConn()

    remote = upload_geojson(conn, image, str(local))

    assert remote == "N101_S06_HE.ome.tiff__omero-362.geojson"
    assert conn.uploaded == [(str(local), remote, GEOJSON_MIMETYPE, GEOJSON_NAMESPACE)]
    assert len(image.linked) == 1


def test_upload_geojson_replaces_previous_upload(tmp_path):
    from lavlab.geojson.omero_io import GEOJSON_NAMESPACE, upload_geojson

    local = tmp_path / "x.geojson"
    local.write_text("{}", encoding="utf-8")
    stale = _FakeGeojsonAnn(
        "N101_S06_HE.ome.tiff__omero-362.geojson", GEOJSON_NAMESPACE
    )
    image = _FakeGeojsonImage(anns=[stale])
    conn = _FakeGeojsonConn()

    upload_geojson(conn, image, str(local))

    assert image.removed == [stale]
    assert conn.deleted == [stale._obj]


def test_geojson_namespace_does_not_collide_with_lr_or_roi():
    """All three attachment kinds coexist on one image without any of them
    picking up another's file -- OMERO matches namespaces exactly."""
    from lavlab.geojson.omero_io import GEOJSON_NAMESPACE, has_uploaded_geojson
    from lavlab.large_recon import has_cached_recon
    from lavlab.roi import has_uploaded_mask

    image = _FakeGeojsonImage(
        anns=[
            _FakeGeojsonAnn("LR10_N101_S06_HE.jp2", "LargeRecon.10"),
            _FakeGeojsonAnn("LR10_N101_S06_HE__annot.jp2", "LargeRecon.10.roi"),
            _FakeGeojsonAnn(
                "N101_S06_HE.ome.tiff__omero-362.geojson", GEOJSON_NAMESPACE
            ),
        ]
    )

    assert has_cached_recon(image, 10, "jp2") is True
    assert has_uploaded_mask(image, 10, "_annot", "jp2") is True
    assert has_uploaded_geojson(image) is True

    # and none of them sees the others when its own is missing
    lr_only = _FakeGeojsonImage(
        anns=[_FakeGeojsonAnn("LR10_N101_S06_HE.jp2", "LargeRecon.10")]
    )
    assert has_uploaded_geojson(lr_only) is False
    assert has_uploaded_mask(lr_only, 10, "_annot", "jp2") is False
