# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Local-source decode path, and the imagecodecs bundling contract.

OMERO ManagedRepository OME-TIFFs are routinely JPEG- or JPEG2000-compressed.
tifffile parses the container but hands tile decompression to imagecodecs,
which resolves each codec through a dynamic ``importlib.import_module`` that
Nuitka cannot see -- so the compiled binary needs every codec extension named
explicitly in ``IMAGECODECS_NUITKA_FLAGS``. These tests guard the two ways
that list goes wrong: a name that no longer exists upstream, and a codec that
tifffile can select but we never bundled.
"""

from __future__ import annotations

import importlib

import pytest

try:
    import numpy as np
    import pyvips  # noqa: F401
    import tifffile
except Exception as exc:  # pragma: no cover
    pytest.skip(f"imaging stack unavailable: {exc}", allow_module_level=True)

imagecodecs = pytest.importorskip("imagecodecs")

from build_native import IMAGECODECS_NUITKA_FLAGS  # noqa: E402

from lavlab.imaging import load_downsampled  # noqa: E402

_FLAGGED_MODULES = [
    flag.split("=", 1)[1]
    for flag in IMAGECODECS_NUITKA_FLAGS
    if flag.startswith("--include-module=")
]


def _write_tiff(path, compression, *, pyramid=False, size=(512, 768)):
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.zeros((h, w, 3), np.uint8)
    arr[..., 0] = (200 + 40 * np.sin(xx / 97.0)).clip(0, 255)
    arr[..., 1] = (150 + 60 * np.sin(yy / 61.0)).clip(0, 255)
    arr[..., 2] = (200 + 40 * np.cos((xx + yy) / 113.0)).clip(0, 255)
    opts = dict(tile=(256, 256), compression=compression, photometric="rgb")
    with tifffile.TiffWriter(str(path), ome=True, bigtiff=True) as tw:
        if pyramid:
            tw.write(arr, subifds=2, **opts)
            tw.write(arr[::2, ::2], subfiletype=1, **opts)
            tw.write(arr[::4, ::4], subfiletype=1, **opts)
        else:
            tw.write(arr, **opts)
    return arr


def test_flagged_modules_all_exist():
    """Every module named for Nuitka must actually exist in imagecodecs.

    A typo or an upstream rename would otherwise only surface as a
    DelayedImportError from the compiled binary, at the first compressed
    tile, on the cluster.
    """
    assert _FLAGGED_MODULES, "expected imagecodecs modules to be flagged"
    missing = []
    for module in _FLAGGED_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    assert not missing, f"flagged but not importable: {missing}"


def test_shared_cython_is_flagged():
    """_shared_cython has no Python-level importer anywhere.

    Every codec extension cimports it at the C level, so Nuitka has nothing
    to follow. Leaving it out makes all the other codecs fail to import,
    each reported as its own DelayedImportError -- a confusing symptom whose
    single cause is this one missing module. Confirmed empirically.
    """
    assert "imagecodecs._shared_cython" in _FLAGGED_MODULES


def test_jpeg_compressed_pyramid_loads(tmp_path):
    """The reported failure: a JPEG-compressed pyramidal OME-TIFF."""
    path = tmp_path / "N469_S14_HE.ome.tiff"
    _write_tiff(path, "jpeg", pyramid=True)

    with tifffile.TiffFile(str(path)) as tf:
        assert int(tf.pages[0].compression) == 7

    img = load_downsampled(str(path), 10)
    assert (img.width, img.height) == (77, 51)
    assert img.bands == 3


@pytest.mark.parametrize(
    "compression",
    ["jpeg", "jpeg2000", "lzw", "deflate", "packbits", "webp", "zstd", "png"],
)
def test_codecs_tifffile_may_select_are_decodable(tmp_path, compression):
    """Each compression scheme we bundle a codec for must actually decode.

    Parametrised rather than merged so a single unbundled codec names
    itself instead of failing the whole sweep anonymously.
    """
    path = tmp_path / f"{compression}.ome.tiff"
    _write_tiff(path, compression)
    img = load_downsampled(str(path), 10)
    assert (img.width, img.height) == (77, 51)
    assert img.bands == 3
