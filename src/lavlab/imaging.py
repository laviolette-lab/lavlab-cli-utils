# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Generic image loading/downsampling for OMERO-managed image files on disk."""

from __future__ import annotations

import logging

import numpy as np
import pyvips as pv
import tifffile

log = logging.getLogger(__name__)


def load_downsampled(src_path: str, downsample: int) -> pv.Image:
    """Return a pyvips Image at exactly 1/downsample the full-resolution dimensions."""
    with tifffile.TiffFile(src_path) as tif:
        if not tif.series:
            return _load_via_fallback(src_path, downsample)

        series = tif.series[0]
        axes = series.axes  # e.g. 'CZYX', 'CYX', 'YXC', 'YX'
        shape = series.shape

        y_ax = axes.index("Y")
        x_ax = axes.index("X")
        full_h, full_w = shape[y_ax], shape[x_ax]
        target_h = max(1, round(full_h / downsample))
        target_w = max(1, round(full_w / downsample))

        # Walk pyramid levels (index 0 = full res) and pick the smallest level
        # that is still >= the target so we always downsample, never upsample.
        best_idx = 0
        for i, lvl in enumerate(series.levels[1:], 1):
            if lvl.shape[y_ax] >= target_h and lvl.shape[x_ax] >= target_w:
                best_idx = i
            else:
                break

        log.debug(
            "Source %s: full=%dx%d target=%dx%d using level %d",
            src_path,
            full_w,
            full_h,
            target_w,
            target_h,
            best_idx,
        )

        arr = series.levels[best_idx].asarray()

    drop = sorted(
        [i for i, a in enumerate(axes) if a not in ("Y", "X", "C", "S")],
        reverse=True,
    )
    for i in drop:
        arr = np.take(arr, 0, axis=i)

    remaining = "".join(a for a in axes if a in ("Y", "X", "C", "S"))
    remaining = remaining.replace("S", "C")
    target_axes = "YX" + ("C" if "C" in remaining else "")
    if remaining != target_axes:
        perm = [remaining.index(a) for a in target_axes]
        arr = arr.transpose(perm)

    img = pv.Image.new_from_array(arr)
    del arr  # new_from_array copies the buffer; release the numpy allocation now
    return img.resize(target_w / img.width, vscale=target_h / img.height)


def _load_via_fallback(src_path: str, downsample: int) -> pv.Image:
    log.warning(
        "No TIFF series found in '%s'; falling back to pyvips for load.", src_path
    )
    try:
        img = pv.Image.tiffload(src_path, access="sequential")
        target_w = max(1, round(img.width / downsample))
        target_h = max(1, round(img.height / downsample))
        return img.resize(target_w / img.width, vscale=target_h / img.height)
    except pv.Error:
        pass

    # pyvips tiffload failed (e.g. bad seek in BigTIFF); try openslide.
    log.warning("pyvips tiffload failed for '%s'; falling back to openslide.", src_path)
    import openslide

    slide = openslide.OpenSlide(src_path)
    full_w, full_h = slide.dimensions
    target_w = max(1, round(full_w / downsample))
    target_h = max(1, round(full_h / downsample))
    best_level = slide.get_best_level_for_downsample(downsample)
    lw, lh = slide.level_dimensions[best_level]
    region = slide.read_region((0, 0), best_level, (lw, lh))
    arr = np.array(region.convert("RGB"))
    img = pv.Image.new_from_array(arr)
    del arr
    return img.resize(target_w / img.width, vscale=target_h / img.height)


def is_rgb(image_path: str) -> bool:
    """Return True if the image at *image_path* has 3 bands (RGB)."""
    try:
        return pv.Image.new_from_file(image_path, access="sequential").bands == 3
    except Exception:
        return False


def write_recon(img, output_path: str, lossless: bool = True) -> None:
    ext = output_path.lower()
    if ext.endswith(".jp2"):
        from PIL import Image as PILImage

        mem = img.write_to_memory()
        arr = np.ndarray(
            buffer=mem, dtype=np.uint8, shape=(img.height, img.width, img.bands)
        )
        if img.bands == 1:
            arr = arr[:, :, 0]
        PILImage.fromarray(arr).save(
            output_path, irreversible=not lossless, tile_size=(1024, 1024)
        )
    elif ext.endswith((".jpg", ".jpeg", ".png")):
        # Neither format's libvips saver accepts a `lossless` option -- JPEG
        # is inherently lossy and PNG is inherently lossless, so there's
        # nothing for the flag to control.
        img.write_to_file(output_path)
    else:
        img.write_to_file(output_path, lossless=lossless)
