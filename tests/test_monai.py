# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for lavlab.monai: the OMERO WSI reader, the datalist and datasets.

OMERO is faked throughout, as in test_tiling.py, but the slide is a real
three-level pyramid file with real tissue contrast, so the thumbnail, tissue
mask and both pixel tiers run for real. Tier 3 keeps the real
NetworkRegionReader and fakes only its async tile gather -- the Ice round
trip -- serving OMERO tiles cut from the same pyramid.

The datalist tests need only lavlab's own dependencies; the reader and
dataset tests are skipped without the monai extra. Import hygiene is in
test_monai_imports.py.
"""

from __future__ import annotations

import csv
import functools
import json
import math
import os
import re
import subprocess
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


# Everything below needs the real stack.
np = pytest.importorskip("numpy")
pytest.importorskip("skimage")
pytest.importorskip("omero")
try:
    import pyvips  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"pyvips unavailable: {exc}", allow_module_level=True)
tifffile = pytest.importorskip("tifffile")

import lavlab.omero_client as omero_client  # noqa: E402
import lavlab.roi as roi_module  # noqa: E402
from lavlab import tiling  # noqa: E402
from lavlab.config import ConfigError  # noqa: E402
from lavlab.monai import datalist as datalist_module  # noqa: E402
from lavlab.monai.datalist import (  # noqa: E402
    build_datalist,
    image_uri,
    parse_image_id,
    tile_params,
)

LEVELS = 3
PIXEL_SIZE = 0.5
GROUP_ID = 3


def _slide(size=1024, seed=0):
    """Glass with a textured, stained tissue block -- real tissue contrast."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), 245, dtype=np.uint8)
    base += rng.integers(0, 8, base.shape, dtype=np.uint8)
    tissue = rng.integers(0, 60, (size - 96, size - 96, 3), dtype=np.uint8)
    base[48:-48, 48:-48] = np.array([170, 50, 150], dtype=np.uint8) + tissue
    return base


def _write_pyramid(path, base):
    with tifffile.TiffWriter(str(path), bigtiff=True) as writer:
        writer.write(base, tile=(64, 64))
        for level in range(1, LEVELS):
            step = 2**level
            writer.write(base[::step, ::step], tile=(64, 64), subfiletype=1)


class _Value:
    def __init__(self, value):
        self._value = value

    def getValue(self):
        return self._value


class _Ref:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class _Pixels:
    def getPixelsType(self):
        return _Value("uint8")

    def getId(self):
        return 99


class _FakeImage:
    def __init__(self, image_id, name, size=1024, pixel_size=PIXEL_SIZE):
        self._id = image_id
        self._name = name
        self._size = size
        self._pixel_size = pixel_size
        self.details = _Ref(group=_Ref(id=_Ref(val=GROUP_ID)))

    def getId(self):
        return self._id

    def getName(self):
        return self._name

    def getSizeX(self):
        return self._size

    def getSizeY(self):
        return self._size

    def getSizeC(self):
        return 3

    def getPixelSizeX(self):
        return self._pixel_size

    def getPixelSizeY(self):
        return self._pixel_size

    def getPrimaryPixels(self):
        return _Pixels()

    def getObjectiveSettings(self):
        return _Ref(getObjective=lambda: _Ref(getNominalMagnification=lambda: 20.0))


class _Desc:
    def __init__(self, size):
        self.sizeX = size
        self.sizeY = size


class _FakeRps:
    def __init__(self, size):
        self._descs = [_Desc(size // 2**level) for level in range(LEVELS)]

    def setPixelsId(self, pixels_id, bypass):
        pass

    def getResolutionLevels(self):
        return len(self._descs)

    def getResolutionDescriptions(self):
        return self._descs

    def setResolutionLevel(self, level):
        pass

    def getTileSize(self):
        return (64, 64)

    def close(self):
        pass


class _Opts:
    def __init__(self):
        self.group = None

    def setOmeroGroup(self, group):
        self.group = group


class _FakeConn:
    """A gateway that refuses to be used outside the process that opened it."""

    def __init__(self, images, record_dir=None):
        self.owner_pid = os.getpid()
        self._images = images
        self._record_dir = record_dir
        self.SERVICE_OPTS = _Opts()
        self.closed = False
        if record_dir:
            open(os.path.join(record_dir, f"open-{self.owner_pid}-{id(self)}"), "w")

    def _check(self):
        if os.getpid() != self.owner_pid:
            raise RuntimeError("OMERO session used outside the process that opened it")

    def isConnected(self):
        self._check()
        return not self.closed

    def getObject(self, kind, image_id):
        self._check()
        return self._images.get(int(image_id))

    def createRawPixelsStore(self):
        self._check()
        return _FakeRps(1024)

    def setGroupForSession(self, group_id):
        self._check()

    def close(self):
        self._check()
        self.closed = True
        if self._record_dir:
            open(os.path.join(self._record_dir, f"closed-{self.owner_pid}"), "w")


SHAPES = [
    # (shape id, level-0 box, textValue); shape id * 10 is its OMERO ROI id
    (7, (128, 128, 576, 576), "g3"),
    (8, (576, 192, 896, 512), "gleason 4 cribriform"),
    (9, (600, 600, 896, 896), "g5"),
    (10, (320, 320, 400, 400), "exclusion roi"),
]


def _install_world(directory, setattr):
    """Two annotated slides on two cases, served from one pyramid file.

    *setattr* is ``monkeypatch.setattr`` in a test, or the builtin in a
    throwaway subprocess that needs the same fake OMERO.
    """
    base = _slide()
    source = os.path.join(str(directory), "slide.tiff")
    _write_pyramid(source, base)
    world = _Ref(
        base=base,
        source=source,
        images={
            1962: _FakeImage(1962, "N101_S08_HE.ome.tiff"),
            7775: _FakeImage(7775, "N102_S02_HE.ome.tiff"),
        },
        sources={1962: source, 7775: source},
        network_reads=0,
    )

    setattr(omero_client, "get_source_file_path", lambda conn, i: world.sources[int(i)])

    def shapes(image, img_downsample=1, include_all=True, **kw):
        d = img_downsample
        return [
            (
                sid,
                (0, 0, 0),
                text,
                [(a / d, b / d), (c / d, b / d), (c / d, e / d), (a / d, e / d)],
            )
            for sid, (a, b, c, e), text in SHAPES
        ]

    setattr(roi_module, "get_shapes_as_points", shapes)
    setattr(
        roi_module, "shape_roi_ids", lambda image: {s[0]: s[0] * 10 for s in SHAPES}
    )
    # Tier-3 thumbnails come off the same pyramid as tier-2 ones.
    setattr(
        tiling,
        "_load_thumbnail_network",
        lambda conn, image, ds: tiling._load_thumbnail_local(source, ds),
    )

    async def gather(self, keys):
        level = base[:: 2**self._plan.index, :: 2**self._plan.index]
        tile_w, tile_h = self._tile_size
        world.network_reads += 1
        for tx, ty in keys:
            x, y = tx * tile_w, ty * tile_h
            self._cache[(tx, ty)] = level[y : y + tile_h, x : x + tile_w].copy()

    setattr(tiling.NetworkRegionReader, "_gather", gather)
    world.conn = functools.partial(_FakeConn, world.images)
    return world


@pytest.fixture
def omero_world(monkeypatch, tmp_path):
    return _install_world(tmp_path, monkeypatch.setattr)


SETTINGS = dict(
    labels=["G3", "G4cg", "G5"],
    size=32,
    mpp=1.5,
    erode=0,
    min_coverage=0.75,
    background_margin=16,
    max_background_tiles=6,
    max_tiles_per_label=10,
    seed=5,
)


def _key(item):
    return (item["omero_image_id"], item["x0"], item["y0"])


def test_image_names_round_trip():
    assert image_uri(1962) == "omero://1962"
    assert parse_image_id("omero://1962") == 1962
    assert parse_image_id(1962) == 1962
    for bad in ("1962", "/data/slide.svs", "omero://abc", True, 1.5):
        with pytest.raises(ValueError):
            parse_image_id(bad)


def _cli_params(*flags):
    from lavlab.cli import build_parser
    from lavlab.commands.tile import _build_params, _validate_args

    args = build_parser().parse_args(["tile", "1", "-o", "/tmp/x", "--roi", *flags])
    _validate_args(args)
    return _build_params(args)


@pytest.mark.parametrize(
    "kwargs, flags",
    [
        (dict(all_labels=True), ["--all"]),
        (
            dict(labels=["G3", "G5"], size=512, mpp=0.5, erode=64, min_coverage=0.75),
            [
                "-t",
                "G3",
                "-t",
                "G5",
                "--size",
                "512",
                "--mpp",
                "0.5",
                "--erode",
                "64",
                "--min-coverage",
                "0.75",
            ],
        ),
        (
            dict(all_labels=True, scales=[1.0, 0.5], background_label=None),
            ["--all", "--scales", "1.0,0.5", "--no-background"],
        ),
        (
            dict(
                all_labels=True,
                downsample=4,
                exclude_text=["Burnt"],
                seed=3,
                background_margin=50,
                max_background_tiles=10,
                overlap=8,
            ),
            [
                "--all",
                "--downsample",
                "4",
                "--exclude-text",
                "Burnt",
                "--seed",
                "3",
                "--background-margin",
                "50",
                "--max-background-tiles",
                "10",
                "--overlap",
                "8",
            ],
        ),
    ],
)
def test_tile_params_matches_the_cli(kwargs, flags):
    assert tile_params(**kwargs) == _cli_params(*flags)


def test_tile_params_rejects_what_the_cli_rejects():
    with pytest.raises(ValueError, match="contradictory"):
        tile_params(labels=["G3"], all_labels=True)
    with pytest.raises(ValueError, match="all_labels=True or at least one"):
        tile_params()
    with pytest.raises(ValueError, match="at most one"):
        tile_params(all_labels=True, mpp=0.5, downsample=2)
    with pytest.raises(ValueError, match="min_coverage is a fraction"):
        tile_params(all_labels=True, min_coverage=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"labels": ["G3"], "all_labels": True},
        {"all_labels": True, "min_coverage": 2},
        {"all_labels": True, "size": 0},
    ],
)
def test_tile_params_errors_name_python_arguments_not_cli_flags(kwargs):
    """The CLI's validation runs underneath, but the caller passed kwargs."""
    with pytest.raises(ValueError) as excinfo:
        tile_params(**kwargs)

    # "--" alone is a dash in these messages; "--word" would be a CLI flag.
    assert not re.search(r"--[a-z]", str(excinfo.value))


def test_missing_credentials_point_at_creds_not_the_command_line(monkeypatch):
    for name in ("OMERO_USER", "OMERO_PASSWORD", "OMERO_HOST", "OMERO_PORT"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigError) as excinfo:
        datalist_module._connect(None)

    message = str(excinfo.value)
    assert "creds=" in message
    assert "OMERO_USER" in message
    assert "command line" not in message


def _cli_flags(**settings):
    flags = []
    for label in settings["labels"]:
        flags += ["-t", label]
    for name, flag in (
        ("size", "--size"),
        ("mpp", "--mpp"),
        ("erode", "--erode"),
        ("min_coverage", "--min-coverage"),
        ("background_margin", "--background-margin"),
        ("max_background_tiles", "--max-background-tiles"),
        ("max_tiles_per_label", "--max-tiles-per-label"),
        ("seed", "--seed"),
    ):
        flags += [flag, str(settings[name])]
    return flags


def test_datalist_matches_tile_coords_only_manifest_row_for_row(omero_world, tmp_path):
    """The parity test: same slide, seed and settings -> the same tiles."""
    params = tile_params(**SETTINGS)
    conn = omero_world.conn()
    out = tmp_path / "out"
    rows = []
    for image_id in (1962, 7775):
        tiling.tile_slide(
            conn,
            omero_world.images[image_id],
            tiling.TileParams(**{**params.__dict__, "coords_only": True}),
            str(out),
        )
    for slide in ("N101/N101_S08_HE", "N102/N102_S02_HE"):
        rows += tiling.read_manifest(str(out / slide))

    datalist = build_datalist([1962, 7775], conn=conn, **SETTINGS)

    def from_row(r):
        return (
            int(r["omero_image_id"]),
            r["case_id"],
            r["slide_id"],
            r["label"],
            int(r["x"]),
            int(r["y"]),
            int(r["x0"]),
            int(r["y0"]),
            int(r["w0"]),
            int(r["level"]),
            int(r["roi_id"] or -1),
            int(r["shape_id"] or -1),
            float(r["mpp"]),
            float(r["coverage"]),
            float(r["tissue_frac"]),
            r["tier"],
        )

    def from_item(d):
        return (
            d["omero_image_id"],
            d["case_id"],
            d["slide_id"],
            d["label_name"],
            d["x"],
            d["y"],
            d["x0"],
            d["y0"],
            d["w0"],
            d["level"],
            d["roi_id"],
            d["shape_id"],
            d["mpp"],
            d["coverage"],
            d["tissue_frac"],
            d["tier"],
        )

    assert len(rows) > 20
    assert {r["label"] for r in rows} == {"G3", "G4cg", "G5", "benign"}
    assert [from_item(d) for d in datalist] == [from_row(r) for r in rows]


def test_datalist_matches_lavlab_tile_dry_run_report(omero_world, monkeypatch, capsys):
    """And through the CLI itself: the --dry-run report counts the same tiles."""
    from collections import Counter

    import lavlab.commands.tile as tile_cmd
    from lavlab.cli import main

    monkeypatch.setattr(tile_cmd, "connect_from_args", lambda args: omero_world.conn())
    monkeypatch.setattr(tile_cmd, "group_of", lambda conn, image: GROUP_ID)
    main(
        [
            "tile",
            "1962",
            "--roi",
            "-o",
            "/tmp/unused",
            "--dry-run",
            *_cli_flags(**SETTINGS),
        ]
    )
    report = capsys.readouterr().out

    datalist = build_datalist(1962, conn=omero_world.conn(), **SETTINGS)
    counts = Counter(d["label_name"] for d in datalist)
    per_roi = Counter(
        (d["label_name"], d["roi_id"]) for d in datalist if d["roi_id"] >= 0
    )

    assert "Dry run for image 1962" in report
    for label, count in counts.items():
        assert any(
            line.split()[:2] == [label, str(count)] for line in report.splitlines()
        ), (label, count, report)
    for (label, roi_id), count in per_roi.items():
        assert f"ROI {roi_id}" in report
        assert any(
            line.split()[0] == label and line.split()[-1] == str(count)
            for line in report.splitlines()
            if f"ROI {roi_id}" in line
        )


def test_datalist_sample_shape_and_classes(omero_world):
    datalist = build_datalist(1962, conn=omero_world.conn(), **SETTINGS)
    item = next(d for d in datalist if d["label_name"] == "G3")

    assert item["image"] == "omero://1962"
    # mpp 1.5 on a 0.5 um/px slide: 3x below full res, read from level 1 (x2)
    assert item["level"] == 1
    assert item["size"] == [48, 48]
    assert item["location"] == [item["y0"], item["x0"]]
    assert item["tile_size"] == 32 and item["mpp"] == 1.5
    assert item["case_id"] == "N101" and item["roi_id"] == 70
    # classes: the requested labels in order, then background
    labels = {d["label_name"]: d["label"] for d in datalist}
    assert labels == {"G3": 0, "G4cg": 1, "G5": 2, "benign": 3}
    background = next(d for d in datalist if d["label_name"] == "benign")
    assert background["roi_id"] == -1 and background["shape_id"] == -1


def test_datalist_scales_carry_concentric_context(omero_world):
    datalist = build_datalist(
        1962, conn=omero_world.conn(), **{**SETTINGS, "mpp": None, "scales": [1.0, 2.0]}
    )

    item = datalist[0]
    assert item["scales"] == [1.0, 2.0]
    (context,) = item["extra_scales"]
    assert context["level"] == 2 and context["size"] == [32, 32]
    assert context["w0"] == 2 * item["w0"]
    assert context["x0"] + context["w0"] // 2 == item["x"]


def test_datalist_skips_unannotated_slides_and_can_skip_failures(
    omero_world, monkeypatch
):
    monkeypatch.setattr(roi_module, "get_shapes_as_points", lambda image, **kw: [])
    assert build_datalist(1962, conn=omero_world.conn(), **SETTINGS) == []

    omero_world.images[5] = _FakeImage(5, "N103_S01_HE.ome.tiff", pixel_size=None)
    with pytest.raises(tiling.MissingPixelSizeError, match="--downsample"):
        build_datalist(5, conn=omero_world.conn(), **SETTINGS)
    assert (
        build_datalist(5, conn=omero_world.conn(), skip_failed=True, **SETTINGS) == []
    )


try:
    import monai
    import torch

    from lavlab.monai import LavlabTileDataset, OmeroPatchDataset, OmeroWSIReader
    from lavlab.monai.datasets import read_manifest_samples

    _HAS_MONAI = True
except ImportError:  # pragma: no cover - depends on the install
    _HAS_MONAI = False

# The datalist tests above need only lavlab's own dependencies, so they run
# without the extra; the reader and datasets need it.
needs_monai = pytest.mark.skipif(
    not _HAS_MONAI, reason="monai extra (monai, torch) not installed"
)


def _reader(world, **kwargs):
    return OmeroWSIReader(conn_factory=world.conn, **kwargs)


@needs_monai
def test_reader_is_a_monai_wsi_reader():
    reader = OmeroWSIReader()

    assert isinstance(reader, monai.data.BaseWSIReader)
    assert reader.backend == "omero"
    assert reader.verify_suffix("omero://1962")
    assert reader.verify_suffix(["omero://1", 2])
    assert not reader.verify_suffix("/data/slide.svs")


@needs_monai
def test_reader_answers_monai_geometry_from_omero(omero_world):
    reader = _reader(omero_world)
    wsi = reader.read("omero://1962")

    assert wsi.tier == "local"
    assert reader.get_file_path(wsi) == "omero://1962"
    assert reader.get_level_count(wsi) == 3
    assert reader.get_size(wsi, 0) == (1024, 1024)
    assert reader.get_size(wsi, 2) == (256, 256)
    assert reader.get_downsample_ratio(wsi, 1) == 2.0
    assert reader.get_mpp(wsi, 0) == (0.5, 0.5)
    assert reader.get_mpp(wsi, 2) == (2.0, 2.0)
    assert reader.get_power(wsi, 1) == 10.0
    # MONAI's own mpp -> level resolution works on top of these
    assert reader.get_valid_level(wsi, None, 1.0, None) == 1


@needs_monai
def test_reader_missing_pixel_size_raises_the_cli_error(omero_world):
    omero_world.images[5] = _FakeImage(5, "N103_S01_HE.ome.tiff", pixel_size=None)
    omero_world.sources[5] = omero_world.source
    reader = _reader(omero_world)
    wsi = reader.read(5)

    with pytest.raises(tiling.MissingPixelSizeError, match=r"image 5: .*--downsample"):
        reader.get_mpp(wsi, 0)
    # level-based reads still work without a pixel size
    patch, _meta = reader.get_data(wsi, location=(0, 0), size=(16, 16), level=0)
    assert patch.shape == (3, 16, 16)


@needs_monai
def test_reader_patch_is_the_level_region_without_resize(omero_world):
    reader = _reader(omero_world)
    wsi = reader.read("omero://1962")

    patch, meta = reader.get_data(wsi, location=(100, 60), size=(40, 40), level=1)

    level1 = omero_world.base[::2, ::2]
    assert patch.dtype == np.uint8
    assert np.array_equal(np.moveaxis(patch, 0, -1), level1[50:90, 30:70])
    assert meta["tier"] == "local"
    assert meta["path"] == "omero://1962"


@needs_monai
def test_reader_resizes_with_lavlab_tiles_kernel(omero_world):
    reader = _reader(omero_world, resize_to=32)
    wsi = reader.read("omero://1962")

    patch, _meta = reader.get_data(wsi, location=(96, 96), size=(48, 48), level=1)

    level1 = omero_world.base[::2, ::2]
    expected = tiling.resize_tile(level1[48:96, 48:96].copy(), 32)
    assert np.array_equal(np.moveaxis(patch, 0, -1), expected)


@needs_monai
def test_both_tiers_read_identical_patches(omero_world):
    local = _reader(omero_world, resize_to=32)
    local_wsi = local.read("omero://1962")
    omero_world.sources[1962] = "/OMERO/ManagedRepository/not/mounted/here.ome.tiff"
    network = _reader(omero_world, resize_to=32)
    network_wsi = network.read("omero://1962")

    assert (local_wsi.tier, network_wsi.tier) == ("local", "network")
    for location, size, level in (
        ((96, 96), 48, 1),
        ((40, 300), 32, 2),
        ((7, 9), 70, 0),
    ):
        a, a_meta = local.get_data(
            local_wsi, location=location, size=(size, size), level=level
        )
        b, b_meta = network.get_data(
            network_wsi, location=location, size=(size, size), level=level
        )
        assert np.array_equal(a, b)
        assert (a_meta["tier"], b_meta["tier"]) == ("local", "network")
    assert omero_world.network_reads == 3


@needs_monai
def test_network_reads_work_inside_a_running_event_loop(omero_world):
    """In a Jupyter notebook every read happens in a thread that already runs
    an event loop, where the network tier's asyncio.run() used to refuse."""
    import asyncio

    local = _reader(omero_world, resize_to=32)
    expected, _meta = local.get_data(
        local.read("omero://1962"), location=(96, 96), size=(48, 48), level=1
    )
    omero_world.sources[1962] = "/OMERO/ManagedRepository/not/mounted/here.ome.tiff"
    network = _reader(omero_world, resize_to=32)

    async def notebook_cell():
        wsi = network.read("omero://1962")
        return network.get_data(wsi, location=(96, 96), size=(48, 48), level=1)

    patch, meta = asyncio.run(notebook_cell())

    assert meta["tier"] == "network"
    assert np.array_equal(patch, expected)


@needs_monai
def test_reader_falls_forward_when_the_source_cannot_be_opened(
    omero_world, tmp_path, caplog
):
    broken = tmp_path / "broken.tiff"
    broken.write_bytes(b"not a tiff")
    omero_world.sources[1962] = str(broken)
    reader = _reader(omero_world)

    with caplog.at_level("WARNING"):
        wsi = reader.read("omero://1962")

    assert wsi.tier == "network"
    assert reader.get_level_count(wsi) == 3
    assert any("Falling back to tier 3" in r.getMessage() for r in caplog.records)


@needs_monai
def test_reader_falls_forward_mid_read(omero_world, tmp_path, caplog):
    """A mount that goes bad after the slide was opened: same pixels, tier 3."""
    import shutil

    copy = tmp_path / "copy.tiff"
    shutil.copy(omero_world.source, copy)
    omero_world.sources[1962] = str(copy)
    reader = _reader(omero_world)
    wsi = reader.read("omero://1962")
    assert wsi.tier == "local"

    copy.write_bytes(b"truncated by a flaky mount")
    with caplog.at_level("WARNING"):
        patch, meta = reader.get_data(wsi, location=(96, 96), size=(32, 32), level=1)

    level1 = omero_world.base[::2, ::2]
    assert np.array_equal(np.moveaxis(patch, 0, -1), level1[48:80, 48:80])
    assert meta["tier"] == "network"
    assert wsi.tier == "network"
    assert any("Falling back to tier 3" in r.getMessage() for r in caplog.records)


@needs_monai
def test_jpeg2000_sources_go_over_the_network_unless_forced(
    omero_world, tmp_path, caplog
):
    jp2 = tmp_path / "slide.jp2"
    jp2.write_bytes(b"x")
    omero_world.sources[1962] = str(jp2)

    with caplog.at_level("WARNING"):
        wsi = _reader(omero_world).read("omero://1962")
    assert wsi.tier == "network"
    assert any("JPEG-2000" in r.getMessage() for r in caplog.records)

    forced = _reader(omero_world, force_local=True).read("omero://1962")
    # forced local on a file libvips can't read: falls forward, never crashes
    assert forced.tier == "network"


@needs_monai
def test_reader_rejects_what_it_cannot_read(omero_world):
    reader = _reader(omero_world)
    with pytest.raises(ValueError, match="not found"):
        reader.read("omero://404")
    with pytest.raises(ValueError, match="omero://"):
        reader.read("/data/slide.svs")
    wsi = reader.read(1962)
    with pytest.raises(ValueError, match="square"):
        reader.get_data(wsi, location=(0, 0), size=(16, 32), level=0)


@needs_monai
def test_reader_and_handles_pickle_without_their_session(omero_world):
    import pickle

    reader = _reader(omero_world, resize_to=32)
    wsi = reader.read("omero://1962")
    before, _ = reader.get_data(wsi, location=(96, 96), size=(48, 48), level=1)

    reader2, wsi2 = pickle.loads(pickle.dumps((reader, wsi)))

    assert reader2._session["conn"] is None and wsi2._live is None
    after, _ = reader2.get_data(wsi2, location=(96, 96), size=(48, 48), level=1)
    assert np.array_equal(before, after)


@needs_monai
def test_reader_reconnects_after_losing_its_session(omero_world):
    reader = _reader(omero_world)
    first = reader.connection()
    first.closed = True

    second = reader.connection()

    assert second is not first and not second.closed


def _extract(world, out, manifest, **settings):
    """Run `lavlab tile` (the library half) over both slides, writing PNGs."""
    params = tile_params(**settings)
    conn = world.conn()
    for image_id in (1962, 7775):
        result = tiling.tile_slide(
            conn, world.images[image_id], params, str(out), manifest_path=str(manifest)
        )
        assert result.status == "done"


def _by_tile(dataset):
    return {
        _key(sample): sample for sample in (dataset[i] for i in range(len(dataset)))
    }


def _all_sections(cls, **kwargs):
    return cls(**kwargs, val_frac=0.0, test_frac=0.0)


@needs_monai
def test_tile_dataset_resolves_paths_against_the_manifest_not_cwd(
    omero_world, tmp_path, monkeypatch
):
    manifest = tmp_path / "index" / "dataset.csv"
    _extract(omero_world, tmp_path / "tiles", manifest, **SETTINGS)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    dataset = LavlabTileDataset(
        os.path.relpath(manifest), section="training", val_frac=0.0, test_frac=0.0
    )
    rows = list(csv.DictReader(open(manifest)))

    assert rows[0]["tile_path"].startswith(os.path.join("..", "tiles"))
    assert len(dataset) == len(rows)
    assert dataset.class_names == ["G3", "G4cg", "G5", "benign"]
    sample = dataset[0]
    assert isinstance(sample["image"], monai.data.MetaTensor)
    assert sample["image"].shape == (3, 32, 32)
    assert sample["image"].dtype == torch.uint8
    assert sample["label"] == dataset.class_names.index(sample["label_name"])
    assert {"case_id", "roi_id", "shape_id", "mpp", "coverage", "tier"} <= set(sample)


@needs_monai
def test_tile_dataset_per_slide_manifest_needs_the_out_root(omero_world, tmp_path):
    out = tmp_path / "tiles"
    _extract(omero_world, out, tmp_path / "dataset.csv", **SETTINGS)
    per_slide = out / "N101" / "N101_S08_HE" / tiling.MANIFEST_NAME

    with pytest.raises(FileNotFoundError, match="root="):
        LavlabTileDataset(str(per_slide))
    dataset = LavlabTileDataset(str(per_slide), root=str(out), val_frac=0, test_frac=0)
    assert dataset[0]["image"].shape == (3, 32, 32)


@needs_monai
def test_live_and_extracted_datasets_yield_the_same_samples(omero_world, tmp_path):
    manifest = tmp_path / "dataset.csv"
    _extract(omero_world, tmp_path / "tiles", manifest, **SETTINGS)

    on_disk = _by_tile(_all_sections(LavlabTileDataset, manifest=str(manifest)))
    live = _by_tile(
        _all_sections(
            OmeroPatchDataset,
            image_ids=[1962, 7775],
            conn=omero_world.conn(),
            conn_factory=omero_world.conn,
            **{**SETTINGS, "tile_seed": SETTINGS["seed"], "seed": 0},
        )
    )

    assert set(live) == set(on_disk) and len(live) > 20
    for key, sample in live.items():
        other = on_disk[key]
        assert torch.equal(sample["image"].as_tensor(), other["image"].as_tensor()), key
        for name in datalist_module.SAMPLE_KEYS:
            assert sample[name] == other[name] or (
                isinstance(sample[name], float)
                and math.isnan(sample[name])
                and math.isnan(other[name])
            ), (key, name)
        assert sample["image"].meta["tier"] == "local"


@needs_monai
def test_live_and_extracted_scales_match_crop_for_crop(omero_world, tmp_path):
    settings = {**SETTINGS, "mpp": None, "scales": [1.0, 2.0], "max_tiles_per_label": 4}
    manifest = tmp_path / "dataset.csv"
    _extract(omero_world, tmp_path / "tiles", manifest, **settings)

    on_disk = _by_tile(_all_sections(LavlabTileDataset, manifest=str(manifest)))
    live = _by_tile(
        _all_sections(
            OmeroPatchDataset,
            image_ids=[1962, 7775],
            conn=omero_world.conn(),
            conn_factory=omero_world.conn,
            **{**settings, "tile_seed": settings["seed"], "seed": 0},
        )
    )

    assert set(live) == set(on_disk)
    for key, sample in live.items():
        assert sample["scales"] == on_disk[key]["scales"] == [1.0, 2.0]
        for name in ("image", "image_1"):
            assert torch.equal(sample[name].as_tensor(), on_disk[key][name].as_tensor())
        assert sample["image_1"].meta["mpp"] == 2.0


@needs_monai
def test_manifest_with_mixed_runs_needs_scales(omero_world, tmp_path):
    manifest = tmp_path / "dataset.csv"
    _extract(omero_world, tmp_path / "a", manifest, **SETTINGS)
    _extract(
        omero_world,
        tmp_path / "b",
        manifest,
        **{**SETTINGS, "mpp": None, "scales": [1.0, 2.0], "max_tiles_per_label": 4},
    )

    with pytest.raises(ValueError, match="pass scales="):
        read_manifest_samples(str(manifest))
    both, _ = read_manifest_samples(str(manifest), scales=[1.0, 2.0])
    assert both and all(s["scales"] == [1.0, 2.0] for s in both)
    single, _ = read_manifest_samples(str(manifest), scales=[1.5])
    assert single and all("scales" not in s and s["mpp"] == 1.5 for s in single)


@needs_monai
def test_transforms_and_caching_apply(omero_world, tmp_path):
    from monai.transforms import ScaleIntensityRanged

    datalist = build_datalist([1962, 7775], conn=omero_world.conn(), **SETTINGS)
    scale = ScaleIntensityRanged(keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0)
    cached = OmeroPatchDataset(
        datalist=datalist,
        conn_factory=omero_world.conn,
        transform=[scale],
        cache_rate=1.0,
        val_frac=0.0,
        test_frac=0.0,
        progress=False,
    )
    omero_world.sources[1962] = omero_world.sources[7775] = "/gone"

    sample = cached[0]

    assert sample["image"].dtype == torch.float32
    assert 0.0 <= float(sample["image"].min()) <= float(sample["image"].max()) <= 1.0
    assert all(item is not None for item in cached._cache)


@needs_monai
def test_reader_drives_a_plain_patch_wsi_dataset(omero_world):
    datalist = build_datalist(1962, conn=omero_world.conn(), **SETTINGS)
    plain = monai.data.PatchWSIDataset(
        datalist,
        reader=OmeroWSIReader(conn_factory=omero_world.conn, resize_to=32),
        center_location=False,
    )
    ours = _all_sections(
        OmeroPatchDataset, datalist=datalist, conn_factory=omero_world.conn
    )

    first = plain[0]
    assert first["image"].shape == (3, 32, 32)
    assert torch.equal(first["image"].as_tensor(), ours[0]["image"].as_tensor())


# ---------------------------------------------------------------------------
# splitting is by case, never by tile
# ---------------------------------------------------------------------------


def _many_cases_manifest(tmp_path, cases=12, tiles=5):
    from PIL import Image

    tile = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (200, 50, 150)).save(tile)
    manifest = tmp_path / "many.csv"
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tiling.MANIFEST_COLUMNS))
        writer.writeheader()
        for case in range(cases):
            for index in range(tiles):
                writer.writerow(
                    {
                        **dict.fromkeys(tiling.MANIFEST_COLUMNS, "0"),
                        "tile_path": "t.png",
                        "omero_image_id": case,
                        "case_id": f"N{100 + case}",
                        "slide_id": f"N{100 + case}_S01_HE",
                        "roi_id": "",
                        "shape_id": "",
                        "label": "G3" if index % 2 else "benign",
                        "x": index,
                        "mpp": "0.5",
                        "size": "8",
                        "tier": "local",
                    }
                )
    return str(manifest)


def _cases(dataset):
    return {dataset.data[i]["case_id"] for i in range(len(dataset))}


@needs_monai
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_sections_never_share_a_case(tmp_path, seed):
    manifest = _many_cases_manifest(tmp_path)
    sections = {
        name: LavlabTileDataset(
            manifest, section=name, seed=seed, val_frac=0.25, test_frac=0.25
        )
        for name in ("training", "validation", "test")
    }

    cases = {name: _cases(ds) for name, ds in sections.items()}
    assert not cases["training"] & cases["validation"]
    assert not cases["training"] & cases["test"]
    assert not cases["validation"] & cases["test"]
    assert set().union(*cases.values()) == {f"N{100 + c}" for c in range(12)}
    assert [len(c) for c in cases.values()] == [6, 3, 3]
    # every tile of a case lands with it
    assert all(len(ds) == 5 * len(cases[name]) for name, ds in sections.items())


@needs_monai
def test_cross_validation_folds_never_share_a_case(tmp_path):
    from monai.apps import CrossValidation

    manifest = _many_cases_manifest(tmp_path, cases=10)
    folds = CrossValidation(
        dataset_cls=LavlabTileDataset, nfolds=5, seed=3, manifest=manifest
    )

    per_fold = [_cases(folds.get_dataset(folds=k)) for k in range(5)]
    for k in range(5):
        others = folds.get_dataset(folds=[f for f in range(5) if f != k])
        assert not per_fold[k] & _cases(others)
        assert len(others) == 5 * len(_cases(others))
    assert sum(len(c) for c in per_fold) == 10
    assert set().union(*per_fold) == {f"N{100 + c}" for c in range(10)}


@needs_monai
def test_live_dataset_splits_by_case_too(omero_world):
    from monai.apps import CrossValidation

    datalist = build_datalist([1962, 7775], conn=omero_world.conn(), **SETTINGS)
    training = OmeroPatchDataset(datalist=datalist, val_frac=0.5, test_frac=0.0)
    validation = OmeroPatchDataset(
        datalist=datalist, section="validation", val_frac=0.5, test_frac=0.0
    )
    folds = CrossValidation(dataset_cls=OmeroPatchDataset, nfolds=2, datalist=datalist)

    assert len(_cases(training)) == len(_cases(validation)) == 1
    assert not _cases(training) & _cases(validation)
    assert len(training) + len(validation) == len(datalist)
    fold_cases = [_cases(folds.get_dataset(folds=k)) for k in (0, 1)]
    assert not fold_cases[0] & fold_cases[1]


# ---------------------------------------------------------------------------
# one OMERO session per process
# ---------------------------------------------------------------------------


UNMOUNTED = "/OMERO/ManagedRepository/not/mounted/here.ome.tiff"


def _unmounted(conn, image_id):
    return UNMOUNTED


def _pyramid_gather(source):
    """A NetworkRegionReader._gather serving OMERO tiles from the pyramid file."""
    levels: dict = {}

    async def gather(self, keys):
        index = self._plan.index
        if index not in levels:
            with tifffile.TiffFile(source) as tif:
                levels[index] = tif.series[0].levels[index].asarray()
        tile_w, tile_h = self._tile_size
        for tx, ty in keys:
            x, y = tx * tile_w, ty * tile_h
            self._cache[(tx, ty)] = levels[index][y : y + tile_h, x : x + tile_w].copy()

    return gather


class _SpawnedConn:
    """A connection factory that brings the fake OMERO into spawned workers.

    A spawned worker starts from a fresh interpreter, without the parent's
    patches, so on first use in any other process this points every image
    at the network tier and serves its tiles from the pyramid file.
    """

    def __init__(self, images, source, record_dir):
        self.images = images
        self.source = source
        self.record_dir = record_dir
        self.parent = os.getpid()

    def __call__(self):
        if os.getpid() != self.parent:
            omero_client.get_source_file_path = _unmounted
            tiling.NetworkRegionReader._gather = _pyramid_gather(self.source)
        return _FakeConn(self.images, self.record_dir)


def _sessions(record):
    names = os.listdir(record)
    opened = {name.split("-")[1] for name in names if name.startswith("open-")}
    closed = {name.split("-")[1] for name in names if name.startswith("closed-")}
    return opened, closed


def _batches(loader):
    seen = {}
    for batch in loader:
        for i in range(len(batch["x0"])):
            key = (
                int(batch["omero_image_id"][i]),
                int(batch["x0"][i]),
                int(batch["y0"][i]),
            )
            seen[key] = batch["image"][i]
    return seen


def _loader_scenario(context, directory, datalist_path):
    """Run a 2-worker DataLoader over the network tier; print what happened.

    Runs as the main program of a fresh interpreter (see
    :func:`_run_loader_scenario`), so for ``fork`` the parent has never run
    libvips. Ends with ``os._exit``: on some macOS Python/PyTorch builds,
    interpreter shutdown after any multi-process DataLoader hangs in
    multiprocessing's resource tracker, with or without lavlab.
    """
    from monai.data import DataLoader

    record = os.path.join(directory, "sessions")
    os.mkdir(record)
    world = _install_world(directory, setattr)
    world.sources = {1962: UNMOUNTED, 7775: UNMOUNTED}
    tiling.NetworkRegionReader._gather = _pyramid_gather(world.source)
    with open(datalist_path) as fh:
        datalist = json.load(fh)
    if context == "spawn":
        factory = _SpawnedConn(world.images, world.source, record)
    else:
        factory = functools.partial(_FakeConn, world.images, record)
    dataset = _all_sections(OmeroPatchDataset, datalist=datalist, conn_factory=factory)

    expected = None
    if context == "spawn":
        # The parent reads everything first: a live session, open readers,
        # cached WSI handles and a used libvips when the workers start.
        expected = _by_tile(dataset)
    else:
        # A live session and open WSI handles, but no pixels: after libvips
        # has run, nothing forked can use it (see the fail-fast test).
        for uri in ("omero://1962", "omero://7775"):
            dataset.patch_dataset._get_wsi_object({"image": uri})

    loader = DataLoader(
        dataset, batch_size=4, num_workers=2, multiprocessing_context=context
    )
    seen = _batches(loader)
    del loader
    if expected is None:
        expected = _by_tile(dataset)
    opened, closed = _sessions(record)
    print(
        json.dumps(
            {
                "same_keys": set(seen) == set(expected),
                "n": len(seen),
                "equal": all(
                    torch.equal(seen[k].as_tensor(), expected[k]["image"].as_tensor())
                    for k in seen
                ),
                "parent": str(os.getpid()),
                "opened": sorted(opened),
                "closed": sorted(closed),
            }
        )
    )
    sys.stdout.flush()
    os._exit(0)


def _run_loader_scenario(context, omero_world, tmp_path):
    datalist = build_datalist([1962, 7775], conn=omero_world.conn(), **SETTINGS)
    datalist_path = tmp_path / "datalist.json"
    datalist_path.write_text(json.dumps(datalist))
    directory = tmp_path / context
    directory.mkdir()
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {tests_dir!r}); import test_monai; "
            f"test_monai._loader_scenario({context!r}, {str(directory)!r}, "
            f"{str(datalist_path)!r})",
        ],
        cwd=SRC,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@needs_monai
@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_dataloader_workers_each_open_their_own_session(context, omero_world, tmp_path):
    result = _run_loader_scenario(context, omero_world, tmp_path)

    assert result["same_keys"] and result["equal"] and result["n"] > 20
    workers = set(result["opened"]) - {result["parent"]}
    assert result["parent"] in result["opened"]
    assert len(workers) == 2  # one session per worker, none shared
    assert workers <= set(result["closed"])  # each worker closed its own on exit


@needs_monai
def test_forking_after_building_the_dataset_fails_fast_instead_of_hanging(omero_world):
    from monai.data import DataLoader

    # Building the datalist cuts each slide's thumbnail with libvips.
    dataset = OmeroPatchDataset(
        image_ids=[1962],
        conn=omero_world.conn(),
        conn_factory=omero_world.conn,
        val_frac=0.0,
        test_frac=0.0,
        **{**SETTINGS, "tile_seed": SETTINGS["seed"], "seed": 0},
    )

    loader = DataLoader(
        dataset, batch_size=4, num_workers=1, multiprocessing_context="fork"
    )
    with pytest.raises(RuntimeError, match="multiprocessing_context='spawn'"):
        next(iter(loader))
