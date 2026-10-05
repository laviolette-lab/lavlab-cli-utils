# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os

import pytest

try:
    import pyvips  # noqa: F401
except Exception as exc:  # pragma: no cover
    pytest.skip(f"pyvips unavailable: {exc}", allow_module_level=True)

from lavlab import large_recon
from lavlab.large_recon import fetch_large_recon
from lavlab.omero_tiles import LargeReconError


class _FakeOriginalFile:
    def __init__(self, name):
        self._name = name

    def getName(self):
        return self._name


class _FakeFileAnnotation:
    def __init__(self, name, content=b"fake-recon-bytes", fail_after=None):
        self._file = _FakeOriginalFile(name)
        self._content = content
        self._fail_after = fail_after
        self._obj = object()

    def getFile(self):
        return self._file

    def getFileInChunks(self, buf=2621440):
        chunks = [self._content[:4], self._content[4:]]
        for i, chunk in enumerate(chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise OSError("boom")
            yield chunk


class _FakeImage:
    def __init__(
        self,
        image_id=1,
        name="N101_S06_HE.ome.tiff",
        annotation=None,
        list_annotations=None,
    ):
        self._id = image_id
        self._name = name
        self._list_annotations = (
            list(list_annotations)
            if list_annotations is not None
            else ([annotation] if annotation is not None else [])
        )
        self.removed = []
        self.linked = []

    def getId(self):
        return self._id

    def getName(self):
        return self._name

    def listAnnotations(self, ns=None):
        return list(self._list_annotations)

    def removeAnnotations(self, anns):
        self.removed.extend(anns)

    def linkAnnotation(self, ann):
        self.linked.append(ann)


class _FakeConn:
    def __init__(self):
        self.deleted = []
        self.uploaded = []
        self._create_should_fail = False

    def deleteObject(self, obj):
        self.deleted.append(obj)

    def createFileAnnfromLocalFile(
        self, path, origFilePathAndName=None, mimetype=None, ns=None
    ):
        if self._create_should_fail:
            raise RuntimeError("upload failed")
        self.uploaded.append((path, origFilePathAndName, mimetype, ns))
        return _FakeFileAnnotation(name=origFilePathAndName or os.path.basename(path))


def _forbid(*_args, **_kwargs):
    raise AssertionError("should not be called")


def _stub_write_recon(monkeypatch):
    def _write(img, output_path, lossless=True):
        with open(output_path, "wb") as fh:
            fh.write(b"generated")

    monkeypatch.setattr(large_recon, "write_recon", _write)


def test_annotation_tier_hit_valid_jp2(tmp_path, monkeypatch):
    ann = _FakeFileAnnotation("LR10_N101_S06_HE.jp2")
    image = _FakeImage(annotation=ann)
    conn = _FakeConn()
    monkeypatch.setattr(large_recon, "generate_over_network", _forbid)
    monkeypatch.setattr(large_recon, "load_downsampled", _forbid)

    out = tmp_path / "out.jp2"
    tier = fetch_large_recon(conn, image, 10, str(out))

    assert tier == "annotation"
    assert out.read_bytes() == b"fake-recon-bytes"
    assert conn.uploaded == []
    assert not (tmp_path / "out.jp2.part").exists()


def test_annotation_tier_respects_requested_format(tmp_path, monkeypatch):
    # Namespace holds both a .jp2 and a .jpg at the same downsample --
    # requesting a .jpg output must find the .jpg one, not whichever
    # happens to come first in the list.
    jp2_ann = _FakeFileAnnotation("LR10_N101_S06_HE.jp2", content=b"jp2-bytes")
    jpg_ann = _FakeFileAnnotation("LR10_N101_S06_HE.jpg", content=b"jpg-bytes")
    image = _FakeImage(list_annotations=[jp2_ann, jpg_ann])
    conn = _FakeConn()
    monkeypatch.setattr(large_recon, "generate_over_network", _forbid)
    monkeypatch.setattr(large_recon, "load_downsampled", _forbid)

    out = tmp_path / "out.jpg"
    tier = fetch_large_recon(conn, image, 10, str(out))

    assert tier == "annotation"
    assert out.read_bytes() == b"jpg-bytes"


def test_upload_uses_canonical_remote_name_independent_of_local_path(
    tmp_path, monkeypatch
):
    image = _FakeImage(name="N101_S06_HE.ome.tiff", annotation=None)
    conn = _FakeConn()

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: None)
    monkeypatch.setattr(large_recon, "generate_over_network", lambda c, im, d: object())
    _stub_write_recon(monkeypatch)

    out = tmp_path / "my_custom_local_name.jp2"
    tier = fetch_large_recon(conn, image, 10, str(out))

    assert tier == "network"
    assert len(conn.uploaded) == 1
    local_path, remote_name, mimetype, ns = conn.uploaded[0]
    assert local_path == str(out)
    assert remote_name == "LR10_N101_S06_HE.jp2"
    assert mimetype == "image/jp2"
    assert ns == "LargeRecon.10"


def test_upload_mimetype_matches_requested_format(tmp_path, monkeypatch):
    image = _FakeImage(annotation=None)
    conn = _FakeConn()

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: None)
    monkeypatch.setattr(large_recon, "generate_over_network", lambda c, im, d: object())
    _stub_write_recon(monkeypatch)

    out = tmp_path / "out.jpg"
    tier = fetch_large_recon(conn, image, 8, str(out))

    assert tier == "network"
    local_path, remote_name, mimetype, ns = conn.uploaded[0]
    assert remote_name == "LR8_N101_S06_HE.jpg"
    assert mimetype == "image/jpeg"
    assert ns == "LargeRecon.8"


def test_annotation_miss_legacy_jpg_falls_through_to_local(tmp_path, monkeypatch):
    old_ann = _FakeFileAnnotation("LR10_foo.jpg")
    image = _FakeImage(annotation=old_ann)
    conn = _FakeConn()

    src = tmp_path / "src.tif"
    src.write_bytes(b"whatever")
    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: str(src))
    monkeypatch.setattr(large_recon, "load_downsampled", lambda p, d: object())
    monkeypatch.setattr(large_recon, "generate_over_network", _forbid)
    _stub_write_recon(monkeypatch)

    out = tmp_path / "out.jp2"
    tier = fetch_large_recon(conn, image, 10, str(out))

    assert tier == "local"
    assert out.read_bytes() == b"generated"
    # A non-.jp2 annotation sharing the namespace (e.g. an older,
    # manually-uploaded .jpg large recon) must NOT be deleted -- only
    # something this tool itself could have written (.jp2) is "ours".
    assert image.removed == []
    assert conn.deleted == []
    assert len(conn.uploaded) == 1
    assert conn.uploaded[0][1:] == (
        "LR10_N101_S06_HE.jp2",
        "image/jp2",
        "LargeRecon.10",
    )


@pytest.mark.parametrize("src_path", [None, "/does/not/exist.tif"])
def test_local_tier_miss_falls_through_to_network(tmp_path, monkeypatch, src_path):
    image = _FakeImage(annotation=None)
    conn = _FakeConn()

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: src_path)
    monkeypatch.setattr(large_recon, "load_downsampled", _forbid)
    monkeypatch.setattr(large_recon, "generate_over_network", lambda c, im, d: object())
    _stub_write_recon(monkeypatch)

    out = tmp_path / "out.jp2"
    tier = fetch_large_recon(conn, image, 10, str(out))

    assert tier == "network"
    assert out.exists()


def test_regenerate_skips_valid_annotation(tmp_path, monkeypatch):
    ann = _FakeFileAnnotation("LR10_N101_S06_HE.jp2")
    image = _FakeImage(annotation=ann)
    conn = _FakeConn()

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: None)
    monkeypatch.setattr(large_recon, "generate_over_network", lambda c, im, d: object())
    _stub_write_recon(monkeypatch)

    out = tmp_path / "out.jp2"
    tier = fetch_large_recon(conn, image, 10, str(out), regenerate=True)

    assert tier == "network"
    assert image.removed == [ann]
    assert conn.deleted == [ann._obj]
    assert len(conn.uploaded) == 1
    assert conn.uploaded[0][1:] == (
        "LR10_N101_S06_HE.jp2",
        "image/jp2",
        "LargeRecon.10",
    )


def test_upload_preserves_other_format_annotations_sharing_the_namespace(
    tmp_path, monkeypatch
):
    # e.g. an older, manually-uploaded LR10_*.png sitting in the same
    # LargeRecon.10 namespace as a stale .jp2 this tool previously wrote.
    stale_jp2 = _FakeFileAnnotation("LR10_N101_S06_HE.jp2")
    other_png = _FakeFileAnnotation("LR10_N101_S06_HE.png")
    image = _FakeImage(list_annotations=[stale_jp2, other_png])
    conn = _FakeConn()

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: None)
    monkeypatch.setattr(large_recon, "generate_over_network", lambda c, im, d: object())
    _stub_write_recon(monkeypatch)

    out = tmp_path / "out.jp2"
    tier = fetch_large_recon(conn, image, 10, str(out), regenerate=True)

    assert tier == "network"
    assert image.removed == [stale_jp2]
    assert conn.deleted == [stale_jp2._obj]
    assert len(conn.uploaded) == 1


def test_skip_upload_true_does_not_touch_annotations(tmp_path, monkeypatch):
    image = _FakeImage(annotation=None)
    conn = _FakeConn()

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: None)
    monkeypatch.setattr(large_recon, "generate_over_network", lambda c, im, d: object())
    _stub_write_recon(monkeypatch)

    out = tmp_path / "out.jp2"
    tier = fetch_large_recon(conn, image, 10, str(out), skip_upload=True)

    assert tier == "network"
    assert conn.uploaded == []
    assert image.removed == []


def test_upload_failure_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    image = _FakeImage(annotation=None)
    conn = _FakeConn()
    conn._create_should_fail = True

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: None)
    monkeypatch.setattr(large_recon, "generate_over_network", lambda c, im, d: object())
    _stub_write_recon(monkeypatch)

    out = tmp_path / "out.jp2"
    with caplog.at_level("INFO"):
        tier = fetch_large_recon(conn, image, 10, str(out))

    assert tier == "network"
    assert out.exists()
    assert any("upload to OMERO" in r.message for r in caplog.records)


def test_network_tier_error_propagates_unwrapped(tmp_path, monkeypatch):
    image = _FakeImage(annotation=None)
    conn = _FakeConn()

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: None)

    def _raise(c, im, d):
        raise LargeReconError("boom")

    monkeypatch.setattr(large_recon, "generate_over_network", _raise)

    out = tmp_path / "out.jp2"
    with pytest.raises(LargeReconError, match="boom"):
        fetch_large_recon(conn, image, 10, str(out))


def test_download_annotation_cleans_up_partial_file_on_failure(tmp_path):
    ann = _FakeFileAnnotation("LR10_N101_S06_HE.jp2", fail_after=1)
    out = tmp_path / "out.jp2"

    with pytest.raises(IOError):
        large_recon._download_annotation(ann, str(out))

    assert not out.exists()
    assert not (tmp_path / "out.jp2.part").exists()


def test_creates_missing_parent_directory(tmp_path, monkeypatch):
    image = _FakeImage(annotation=None)
    conn = _FakeConn()

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: None)
    monkeypatch.setattr(large_recon, "generate_over_network", lambda c, im, d: object())
    _stub_write_recon(monkeypatch)

    out = tmp_path / "a" / "b" / "out.jp2"
    tier = fetch_large_recon(conn, image, 10, str(out))

    assert tier == "network"
    assert out.exists()


def test_has_cached_recon_true_when_matching_annotation_exists():
    image = _FakeImage(annotation=_FakeFileAnnotation("LR10_N101_S06_HE.jp2"))
    assert large_recon.has_cached_recon(image, 10, "jp2") is True


def test_has_cached_recon_false_when_no_annotations():
    assert large_recon.has_cached_recon(_FakeImage(annotation=None), 10, "jp2") is False


def test_has_cached_recon_is_format_specific():
    # A .jpg cached at this downsample is not a .jp2 cache hit.
    image = _FakeImage(annotation=_FakeFileAnnotation("LR10_N101_S06_HE.jpg"))
    assert large_recon.has_cached_recon(image, 10, "jp2") is False
    assert large_recon.has_cached_recon(image, 10, "jpg") is True


def test_has_cached_recon_does_not_download(monkeypatch):
    # The whole point of this check is that it's cheap -- listing
    # annotations only, never pulling bytes.
    monkeypatch.setattr(large_recon, "_download_annotation", _forbid)
    image = _FakeImage(annotation=_FakeFileAnnotation("LR10_N101_S06_HE.jp2"))
    assert large_recon.has_cached_recon(image, 10, "jp2") is True


def test_namespace_format():
    assert large_recon._namespace(10) == "LargeRecon.10"


def test_mimetype_for_ext_known_and_unknown():
    assert large_recon._mimetype_for_ext("jp2") == "image/jp2"
    assert large_recon._mimetype_for_ext("jpg") == "image/jpeg"
    assert large_recon._mimetype_for_ext("png") == "image/png"
    assert large_recon._mimetype_for_ext("bmp") == "application/octet-stream"


# --- tier 2 -> tier 3 fallback -------------------------------------------


def _local_source(tmp_path, monkeypatch, raises):
    """Wire up a readable local source whose load_downsampled raises *raises*."""
    src = tmp_path / "src.ome.tiff"
    src.write_bytes(b"whatever")

    def _boom(path, downsample):
        raise raises

    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: str(src))
    monkeypatch.setattr(large_recon, "load_downsampled", _boom)
    return src


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("<COMPRESSION.JPEG: 7> requires the 'imagecodecs' package"),
        ImportError("could not import name 'jpeg8_decode' from 'imagecodecs'"),
        OSError("Stale file handle"),
        KeyError("34712"),
        IndexError("list index out of range"),
        MemoryError(),
        large_recon.pv.Error("libvips could not open the file"),
    ],
)
def test_local_read_failure_falls_back_to_network(tmp_path, monkeypatch, caplog, exc):
    image = _FakeImage(annotation=None)
    conn = _FakeConn()
    _local_source(tmp_path, monkeypatch, exc)
    monkeypatch.setattr(large_recon, "generate_over_network", lambda c, im, d: object())
    _stub_write_recon(monkeypatch)

    out = tmp_path / "out.jp2"
    with caplog.at_level("WARNING"):
        tier = fetch_large_recon(conn, image, 10, str(out))
    assert tier == "network"
    assert out.read_bytes() == b"generated"
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "tier 2" in msg and "tier 3" in msg.lower()
    assert type(exc).__name__ in msg
    assert "src.ome.tiff" in msg


@pytest.mark.parametrize(
    "exc",
    [
        AttributeError("'NoneType' has no attribute 'shape'"),
        TypeError("unsupported operand"),
        NameError("name 'foo' is not defined"),
    ],
)
def test_local_read_bug_is_not_masked_by_fallback(tmp_path, monkeypatch, exc):
    """Programming errors must propagate, not vanish into a slow retry."""
    image = _FakeImage(annotation=None)
    conn = _FakeConn()
    _local_source(tmp_path, monkeypatch, exc)
    monkeypatch.setattr(large_recon, "generate_over_network", _forbid)
    _stub_write_recon(monkeypatch)

    with pytest.raises(type(exc)):
        fetch_large_recon(conn, image, 10, str(tmp_path / "out.jp2"))


def test_successful_local_read_does_not_warn_or_hit_network(
    tmp_path, monkeypatch, caplog
):
    image = _FakeImage(annotation=None)
    conn = _FakeConn()
    src = tmp_path / "src.ome.tiff"
    src.write_bytes(b"whatever")
    monkeypatch.setattr(large_recon, "get_source_file_path", lambda c, i: str(src))
    monkeypatch.setattr(large_recon, "load_downsampled", lambda p, d: object())
    monkeypatch.setattr(large_recon, "generate_over_network", _forbid)
    _stub_write_recon(monkeypatch)

    with caplog.at_level("WARNING"):
        tier = fetch_large_recon(conn, image, 10, str(tmp_path / "out.jp2"))

    assert tier == "local"
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
