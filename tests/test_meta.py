# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

try:
    from omero.rtypes import rint, rstring
    from omero_model_RectangleI import RectangleI
except Exception as exc:  # pragma: no cover
    pytest.skip(f"omero-py unavailable: {exc}", allow_module_level=True)

from lavlab.commands.meta import _process_image, _shape_color, _shape_has_comment


def _pack_rgba(r, g, b, a=255):
    return (r << 24) | (g << 16) | (b << 8) | a


def _rectangle(id_, rgba=None, text=None):
    s = RectangleI()
    if rgba is not None:
        s.setStrokeColor(rint(_pack_rgba(*rgba)))
    if text is not None:
        s.setTextValue(rstring(text))
    return s


def test_shape_has_comment_none():
    assert _shape_has_comment(_rectangle(1)) is False


def test_shape_has_comment_whitespace_only_is_false():
    assert _shape_has_comment(_rectangle(1, text="   ")) is False


def test_shape_has_comment_real_text_is_true():
    assert _shape_has_comment(_rectangle(1, text="Seminal Vesicles")) is True


def test_shape_color_none_when_no_stroke_set():
    assert _shape_color(_rectangle(1)) is None


def test_shape_color_decodes_positive_and_signed_packed_values():
    s_pos = _rectangle(1, rgba=(25, 20, 255))
    assert _shape_color(s_pos) == (25, 20, 255)

    # A channel >= 128 in the red byte pushes the packed value past
    # INT32_MAX, so real OMERO data arrives here as a negative int.
    s_neg = RectangleI()
    packed = _pack_rgba(255, 122, 0)
    s_neg.setStrokeColor(rint(packed - 2**32))
    assert _shape_color(s_neg) == (255, 122, 0)


class _FakeRLong:
    def __init__(self, val):
        self.val = val


class _FakeGroup:
    def __init__(self, gid):
        self.id = _FakeRLong(gid)


class _FakeDetails:
    def __init__(self, gid):
        self.group = _FakeGroup(gid)


class _FakeImage:
    def __init__(self, image_id, group_id=1):
        self._id = image_id
        self.details = _FakeDetails(group_id)

    def getId(self):
        return self._id


class _FakeServiceOpts:
    def __init__(self):
        self.calls = []

    def setOmeroGroup(self, group_id):
        self.calls.append(group_id)


class _FakeUpdateService:
    def __init__(self):
        self.saved = []

    def saveObject(self, obj):
        self.saved.append(obj)


class _FakeConn:
    def __init__(self, image=None):
        self._image = image
        self.SERVICE_OPTS = _FakeServiceOpts()
        self._update_service = _FakeUpdateService()

    def getObject(self, type_, image_id):
        if self._image is not None and self._image.getId() == image_id:
            return self._image
        return None

    def getUpdateService(self):
        return self._update_service


class _FakeRoi:
    def __init__(self, shapes):
        self._shapes = shapes

    def copyShapes(self):
        return list(self._shapes)


def test_process_image_not_found_returns_zeroes(monkeypatch, caplog):
    conn = _FakeConn(image=None)

    with caplog.at_level("WARNING"):
        result = _process_image(conn, 999, {}, tolerance=10)

    assert result == (0, 0, 0)
    assert any("not found" in r.message for r in caplog.records)


def test_process_image_updates_matches_skips_commented_and_unmatched(monkeypatch):
    already_commented = _rectangle(1, rgba=(25, 20, 255), text="already set")
    matches_palette = _rectangle(2, rgba=(25, 20, 255))
    no_color = _rectangle(3)
    unmatched_color = _rectangle(4, rgba=(1, 2, 3))

    image = _FakeImage(image_id=362)
    conn = _FakeConn(image=image)
    # _process_image imports get_rois from lavlab.roi at call time (keeping
    # numpy/skimage/omero out of parser construction), so patch it at source.
    monkeypatch.setattr(
        "lavlab.roi.get_rois",
        lambda img: [
            _FakeRoi([already_commented, matches_palette, no_color, unmatched_color])
        ],
    )

    mapping = {(25, 20, 255): "Seminal Vesicles"}
    updated, skipped_has_comment, skipped_no_match = _process_image(
        conn, 362, mapping, tolerance=10
    )

    assert updated == 1
    assert skipped_has_comment == 1
    assert skipped_no_match == 2
    assert conn.SERVICE_OPTS.calls == ["1"]
    assert conn._update_service.saved == [matches_palette]
    assert matches_palette.getTextValue().getValue() == "Seminal Vesicles"
    # Untouched shapes must not have been saved or had their text set.
    assert already_commented.getTextValue().getValue() == "already set"
    assert no_color.getTextValue() is None
    assert unmatched_color.getTextValue() is None


def test_process_image_dry_run_reports_without_writing(monkeypatch, caplog):
    matches_palette = _rectangle(1, rgba=(25, 20, 255))
    image = _FakeImage(image_id=362)
    conn = _FakeConn(image=image)
    monkeypatch.setattr(
        "lavlab.roi.get_rois", lambda img: [_FakeRoi([matches_palette])]
    )

    mapping = {(25, 20, 255): "Seminal Vesicles"}
    with caplog.at_level("INFO"):
        updated, skipped_has_comment, skipped_no_match = _process_image(
            conn, 362, mapping, tolerance=10, dry_run=True
        )

    # Still counted as a match, but nothing actually written.
    assert updated == 1
    assert skipped_has_comment == 0
    assert skipped_no_match == 0
    assert conn._update_service.saved == []
    assert matches_palette.getTextValue() is None
    assert any(
        "would be set to 'Seminal Vesicles'" in r.message for r in caplog.records
    )
