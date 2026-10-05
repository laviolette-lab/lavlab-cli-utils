# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Cut whole-slide images into fixed-size tiles for classifier training.

Two modes: ``--whole`` (every tissue tile on the slide) and ``--roi``
(tiles labelled by the annotation they fall inside). Both share one grid,
one tissue filter and one tile-cutting function; only where the pixels
come from differs, which is what keeps the two pixel tiers producing the
same tiles.

Pixels come from one of two tiers, mirroring ``lavlab.large_recon``:

* **local** -- OMERO's managed repository is mounted here (true on the
  cluster), so regions are cropped straight out of the source file's own
  pyramid.
* **network** -- otherwise, regions are fetched through OMERO's tile API.

Everything expensive is decided at *analysis resolution*: one coarse
thumbnail per slide (~8 samples across a tile) carries the tissue mask and
the rasterised ROIs, and integral images turn "what fraction of this tile
is tissue / is G3 / is excluded" into an O(1) lookup per tile. Full-res
pixels are only ever read for tiles that already passed every filter --
a 40x whole-mount grids to ~10^5 tiles, so reading pixels to reject
background is not affordable.

Nothing here imports argparse; ``lavlab.commands.tile`` is the CLI in
front of it. ``omero``, ``pyvips``, ``skimage`` and ``tifffile`` are
imported inside the functions that need them, so the pure geometry and
labelling helpers can be exercised (and used from a notebook) with just
numpy installed.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from lavlab import __version__
from lavlab.naming import get_stem

log = logging.getLogger(__name__)


class TilingError(Exception):
    """Raised when a slide cannot be tiled with the settings given."""


class MissingPixelSizeError(TilingError):
    """Raised when ``--mpp`` was asked for but the image has no pixel size.

    Separate from :class:`TilingError` because it is recoverable by the
    user in one specific way -- passing ``--downsample`` instead -- and the
    CLI says so rather than just reporting a failure.
    """


class UnreadableSourceError(TilingError):
    """Raised when a local source file cannot serve regions.

    Caught by the tier-2 path and turned into a fall-forward to tier 3,
    exactly like ``large_recon``'s local-source handling.
    """


SUBSAMPLES_PER_TILE = 8

DEFAULT_EXCLUDE_TEXT = ("exclusion roi",)

DEFAULT_BACKGROUND_LABEL = "non-regign of interest"

JP2_SUFFIXES = (".jp2", ".j2k", ".jpf", ".jpx", ".jpc")

MANIFEST_NAME = "manifest.csv"
PARAMS_NAME = "tile_params.json"

# The first ten columns are the dataset contract: enough to find, group
# (by case/slide/ROI) and re-cut every tile. x/y are the tile's *centre* in
# level-0 pixels, so concentric crops from --scales share them. The rest
# record how the tile was chosen and read.
MANIFEST_COLUMNS = (
    "tile_path",
    "omero_image_id",
    "case_id",
    "slide_id",
    "roi_id",
    "label",
    "x",
    "y",
    "mpp",
    "size",
    "shape_id",
    "x0",
    "y0",
    "w0",
    "h0",
    "level",
    "coverage",
    "tissue_frac",
    "tier",
)

DEFAULT_JPEG_QUALITY = 90

WHOLE_LABEL = "whole"

_UNKNOWN_SUBJECT = "unknown_subject"
_SUBJECT_RE = re.compile(r"^(N\d+)(?=_)", re.IGNORECASE)
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def subject_from_image_name(image_name: str) -> str | None:
    """Extract the subject ID from an OMERO image name.

    Slide names follow ``N<subject>_<slide>_<stain>`` (e.g.
    ``N101_S08_HE``), so the subject is the leading ``N###`` token. It is
    returned verbatim (upper-cased), *not* reduced to bare digits and not
    translated to the on-disk directory name -- the lab's subject
    directories don't consistently share the image name's prefix (subject
    ``N101`` lives in a directory called ``1101``), and the bundled fs_map
    has known bugs around exactly that. This is a grouping key for
    training data, so a stable value that round-trips from the image name
    beats one that matches either convention.

    :param image_name: the OMERO image name, with or without extensions
    :type image_name: str
    :return: the subject ID (e.g. ``N101``), or None if unparseable
    :rtype: str | None
    """
    match = _SUBJECT_RE.match(image_name.strip())
    if match is None:
        return None
    return match.group(1).upper()


def subject_dir_name(image_name: str) -> str:
    """Return the subject folder for an image, warning when unparseable.

    :param image_name: the OMERO image name
    :type image_name: str
    :return: the subject ID, or ``unknown_subject``
    :rtype: str
    """
    subject = subject_from_image_name(image_name)
    if subject is None:
        log.warning(
            "Image name '%s' does not start with an N<digits>_ subject token; "
            "writing under '%s/'.",
            image_name,
            _UNKNOWN_SUBJECT,
        )
        return _UNKNOWN_SUBJECT
    return subject


def sanitize_label(text: str) -> str:
    """Make a ``textValue`` safe to use as a directory name.

    :param text: raw label text
    :type text: str
    :return: the text with runs of unsafe characters collapsed to ``_``
    :rtype: str
    """
    cleaned = _UNSAFE_CHARS.sub("_", text.strip()).strip("._-")
    return cleaned or "unlabeled"


def _default_label_map_path() -> Path:
    return Path(resources.files("lavlab.data").joinpath("default_tile_labels.yaml"))


def load_label_map(path: str | None = None) -> dict[str, list[str]]:
    """Load a ``{folder_name: [textValue aliases]}`` map.

    :param path: a custom YAML file, or None for the bundled lab default
    :type path: str | None
    :return: folder name -> list of aliases
    :rtype: dict[str, list[str]]
    :raises TilingError: if *path* is given but missing or malformed
    """
    if path is not None:
        map_path = Path(path)
        if not map_path.is_file():
            raise TilingError(f"label map '{path}' does not exist.")
    else:
        map_path = _default_label_map_path()
        if not map_path.is_file():
            return {}

    with open(map_path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TilingError(
            f"label map '{map_path}' must be a mapping of folder -> aliases."
        )

    mapping: dict[str, list[str]] = {}
    for folder, aliases in raw.items():
        if aliases is None:
            aliases = []
        if isinstance(aliases, str):
            aliases = [aliases]
        mapping[str(folder)] = [str(a) for a in aliases]
    return mapping


def build_alias_lookup(label_map: dict[str, list[str]]) -> dict[str, str]:
    """Invert a label map into ``{alias_lowercased: folder}``.

    The folder name itself is always registered as an alias, so a
    ``textValue`` that already matches the folder needs no explicit entry.

    :param label_map: folder -> aliases
    :type label_map: dict[str, list[str]]
    :return: lowercased, trimmed alias -> folder name
    :rtype: dict[str, str]
    """
    lookup: dict[str, str] = {}
    for folder, aliases in label_map.items():
        for alias in [folder, *aliases]:
            key = str(alias).strip().lower()
            if key:
                lookup.setdefault(key, folder)
    return lookup


def map_text_value(
    text: str | None,
    lookup: dict[str, str],
    *,
    include_all: bool = False,
    warned: set[str] | None = None,
) -> str | None:
    """Resolve a ``textValue`` to the folder its tiles belong in.

    :param text: the shape's textValue (already lowercased by
        ``lavlab.roi``), or None for an unlabelled shape
    :type text: str | None
    :param lookup: alias -> folder, from :func:`build_alias_lookup`
    :type lookup: dict[str, str]
    :param include_all: with ``--all``, fall back to the sanitized
        textValue rather than dropping an unmapped shape
    :type include_all: bool
    :param warned: set of values already warned about, so an unmapped
        label is reported once per slide rather than once per shape
    :type warned: set[str] | None
    :return: the folder name, or None if this shape has no class
    :rtype: str | None
    """
    if text is None:
        return None
    key = text.strip().lower()
    if not key:
        return None
    folder = lookup.get(key)
    if folder is not None:
        return folder
    if not include_all:
        return None
    if warned is not None and key not in warned:
        warned.add(key)
        log.warning(
            "textValue '%s' has no entry in the label map; using '%s/' as its "
            "folder. Add it to a --labels YAML to control the name.",
            text,
            sanitize_label(text),
        )
    return sanitize_label(text)


def selects_label(folder: str, text: str | None, text_filter: Sequence[str]) -> bool:
    """Return whether ``-t`` selected this shape.

    A ``-t`` value matches either the resolved folder name or the raw
    ``textValue``, both case-insensitively, so ``-t G3`` works whether the
    annotator typed ``G3`` or ``Gleason 3``.

    :param folder: the resolved folder name
    :type folder: str
    :param text: the raw textValue
    :type text: str | None
    :param text_filter: lowercased ``-t`` values
    :type text_filter: Sequence[str]
    :return: whether this shape is selected
    :rtype: bool
    """
    wanted = {t.strip().lower() for t in text_filter}
    if folder.strip().lower() in wanted:
        return True
    return text is not None and text.strip().lower() in wanted


def resolve_total_downsample(
    *,
    mpp: float | None = None,
    downsample: float | None = None,
    pixel_size_x: float | None = None,
    pixel_size_y: float | None = None,
) -> float:
    """Work out how far below full resolution the output tiles sit.

    :param mpp: target micrometres per pixel
    :type mpp: float | None
    :param downsample: an explicit factor, used instead of *mpp*
    :type downsample: float | None
    :param pixel_size_x: the image's physical pixel size in um
    :type pixel_size_x: float | None
    :param pixel_size_y: ditto, Y
    :type pixel_size_y: float | None
    :return: the level-0 pixels per output pixel
    :rtype: float
    :raises MissingPixelSizeError: with *mpp* but no physical pixel size
    :raises TilingError: for a non-positive factor, or a target finer than
        the slide was scanned at
    """
    if downsample is not None:
        if downsample <= 0:
            raise TilingError(f"--downsample must be positive, got {downsample}.")
        return float(downsample)

    if mpp is None:
        raise TilingError("one of --mpp or --downsample is required.")
    if not pixel_size_x or pixel_size_x <= 0:
        raise MissingPixelSizeError(
            "image has no physical pixel size recorded, so --mpp cannot be "
            "converted to a scale factor; pass --downsample N instead."
        )

    sizes = [float(pixel_size_x)]
    if pixel_size_y and pixel_size_y > 0:
        sizes.append(float(pixel_size_y))
    base = sum(sizes) / len(sizes)

    ratio = float(mpp) / base
    if ratio < 0.999:
        raise TilingError(
            f"--mpp {mpp} is finer than the slide's own {base:.4f} um/px; "
            "tiling would have to upsample. Pick a larger --mpp."
        )
    return ratio


@dataclass(frozen=True)
class LevelPlan:
    """Which pyramid level to read, and how far to shrink what comes back."""

    index: int
    downsample: float
    read_size: int
    scale: float


def select_level(
    level_widths: Sequence[int], width0: int, ds_total: float, size: int
) -> LevelPlan:
    """Pick the pyramid level to read tiles from.

    Chooses the smallest level still at or above the target resolution, so
    the final resize is always a downsample -- the same rule
    ``lavlab.imaging.load_downsampled`` and
    ``lavlab.omero_tiles.closest_resolution_level`` use, kept consistent so
    a tile cut from a local file and one cut over the network come off the
    same level of the same pyramid.

    :param level_widths: width of each level, index 0 = full resolution
    :type level_widths: Sequence[int]
    :param width0: full-resolution width
    :type width0: int
    :param ds_total: level-0 pixels per output pixel
    :type ds_total: float
    :param size: output tile edge in pixels
    :type size: int
    :return: the level, its downsample, the region size to read, and the
        resize factor to apply
    :rtype: LevelPlan
    :raises TilingError: if no usable level was supplied
    """
    usable = [(i, w) for i, w in enumerate(level_widths) if w and w > 0]
    if not usable:
        raise TilingError("image reports no usable pyramid levels.")

    best_index = usable[0][0]
    for index, width in usable:
        if width0 / width <= ds_total + 1e-9:
            best_index = index
        else:
            break

    level_downsample = width0 / dict(usable)[best_index]
    read_size = max(1, int(round(size * ds_total / level_downsample)))
    return LevelPlan(
        index=best_index,
        downsample=level_downsample,
        read_size=read_size,
        scale=size / read_size,
    )


@dataclass(frozen=True)
class GridTile:
    """One grid cell, in both output and level-0 coordinates."""

    col: int
    row: int
    xt: int
    yt: int
    x0: int
    y0: int
    w0: int
    h0: int


def build_grid(
    width0: int, height0: int, size: int, overlap: int, ds_total: float
) -> list[GridTile]:
    """Lay a tile grid over the slide.

    The grid is built in *target* (output) coordinates -- that is where
    ``size`` and ``overlap`` are meaningful -- and each cell is then mapped
    back to level-0 pixels for the manifest. Cells that would run off the
    edge in either coordinate system are dropped rather than padded or
    shrunk, so every tile written is exactly ``size`` square.

    :param width0: full-resolution width
    :type width0: int
    :param height0: full-resolution height
    :type height0: int
    :param size: output tile edge
    :type size: int
    :param overlap: output pixels shared between neighbours
    :type overlap: int
    :param ds_total: level-0 pixels per output pixel
    :type ds_total: float
    :return: the grid, in row-major order
    :rtype: list[GridTile]
    :raises TilingError: for a non-positive size or an out-of-range overlap
    """
    if size <= 0:
        raise TilingError(f"--size must be positive, got {size}.")
    if not 0 <= overlap < size:
        raise TilingError(
            f"--overlap must be at least 0 and less than --size ({size}), got {overlap}."
        )

    stride = size - overlap
    width_t = int(width0 / ds_total)
    height_t = int(height0 / ds_total)
    tile0 = int(round(size * ds_total))

    tiles: list[GridTile] = []
    row = 0
    yt = 0
    while yt + size <= height_t:
        col = 0
        xt = 0
        while xt + size <= width_t:
            x0 = int(round(xt * ds_total))
            y0 = int(round(yt * ds_total))
            if x0 + tile0 <= width0 and y0 + tile0 <= height0:
                tiles.append(GridTile(col, row, xt, yt, x0, y0, tile0, tile0))
            xt += stride
            col += 1
        yt += stride
        row += 1
    return tiles


def analysis_downsample(tile0: int, sub: int = SUBSAMPLES_PER_TILE) -> int:
    """Pick the downsample for the per-slide analysis thumbnail.

    :param tile0: tile edge in level-0 pixels
    :type tile0: int
    :param sub: samples wanted across one tile
    :type sub: int
    :return: a downsample factor giving roughly *sub* samples per tile
    :rtype: int
    """
    return max(1, int(round(tile0 / max(1, sub))))


def tile_windows(
    tiles: Sequence[GridTile], analysis_ds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map grid cells onto analysis-resolution windows.

    :param tiles: the grid
    :type tiles: Sequence[GridTile]
    :param analysis_ds: the thumbnail's downsample factor
    :type analysis_ds: int
    :return: ``(xs, ys, ws, hs)`` arrays in thumbnail pixels
    :rtype: tuple[numpy.ndarray, ...]
    """
    if not tiles:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty
    xs = np.array([t.x0 for t in tiles], dtype=np.float64) / analysis_ds
    ys = np.array([t.y0 for t in tiles], dtype=np.float64) / analysis_ds
    ws = np.array([t.w0 for t in tiles], dtype=np.float64) / analysis_ds
    hs = np.array([t.h0 for t in tiles], dtype=np.float64) / analysis_ds
    return (
        np.rint(xs).astype(np.int64),
        np.rint(ys).astype(np.int64),
        np.maximum(np.rint(ws).astype(np.int64), 1),
        np.maximum(np.rint(hs).astype(np.int64), 1),
    )


def integral_image(mask: np.ndarray) -> np.ndarray:
    """Build a summed-area table with a zero border.

    :param mask: a 2-D boolean or numeric array
    :type mask: numpy.ndarray
    :return: an ``(H+1, W+1)`` int64 summed-area table
    :rtype: numpy.ndarray
    """
    values = np.asarray(mask)
    integ = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(values.astype(np.int64), axis=0), axis=1, out=integ[1:, 1:])
    return integ


def window_fractions(
    integ: np.ndarray, xs: np.ndarray, ys: np.ndarray, ws: np.ndarray, hs: np.ndarray
) -> np.ndarray:
    """Fraction of each window that is set, from a summed-area table.

    :param integ: table from :func:`integral_image`
    :type integ: numpy.ndarray
    :param xs: window left edges
    :type xs: numpy.ndarray
    :param ys: window top edges
    :type ys: numpy.ndarray
    :param ws: window widths
    :type ws: numpy.ndarray
    :param hs: window heights
    :type hs: numpy.ndarray
    :return: one fraction in ``[0, 1]`` per window
    :rtype: numpy.ndarray
    """
    if len(xs) == 0:
        return np.zeros(0, dtype=np.float64)
    height = integ.shape[0] - 1
    width = integ.shape[1] - 1
    x0 = np.clip(xs, 0, width)
    y0 = np.clip(ys, 0, height)
    x1 = np.clip(xs + ws, 0, width)
    y1 = np.clip(ys + hs, 0, height)
    total = (integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]).astype(
        np.float64
    )
    area = np.maximum((x1 - x0) * (y1 - y0), 1).astype(np.float64)
    return total / area


def tissue_mask(thumbnail: np.ndarray, min_object_px: int = 16) -> np.ndarray:
    """Segment tissue from background on a low-resolution thumbnail.

    Otsu on HSV saturation for colour input (glass is bright and grey,
    stained tissue is not), on inverted grey otherwise, then a small
    closing and speck removal. Deliberately simple: this only has to
    decide which tiles are worth reading at full resolution, and an
    over-inclusive mask costs a read while an over-aggressive one loses
    training data.

    :param thumbnail: an ``(H, W)`` or ``(H, W, C)`` uint8 array
    :type thumbnail: numpy.ndarray
    :param min_object_px: drop connected components smaller than this
    :type min_object_px: int
    :return: a boolean tissue mask
    :rtype: numpy.ndarray
    """
    from skimage import color, filters, measure, morphology

    array = np.asarray(thumbnail)
    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[:, :, :3].astype(np.float64) / 255.0
        signal = color.rgb2hsv(rgb)[:, :, 1]
    else:
        grey = array if array.ndim == 2 else array[:, :, 0]
        signal = 1.0 - (grey.astype(np.float64) / 255.0)
    if float(signal.max() - signal.min()) < 1e-6:
        return np.zeros(signal.shape, dtype=bool)

    mask = signal > filters.threshold_otsu(signal)
    mask = np.asarray(morphology.closing(mask, morphology.disk(1)), dtype=bool)
    if min_object_px > 1 and mask.any():
        components = measure.label(mask, connectivity=1)
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        mask = (sizes >= min_object_px)[components]

    return np.asarray(mask, dtype=bool)


def rasterize_shapes(
    shapes: Iterable[tuple], shape_hw: tuple[int, int]
) -> tuple[dict[str | None, np.ndarray], np.ndarray]:
    """Rasterise ROI shapes into one boolean mask per ``textValue``.

    Uses the same even-odd polygon fill ``lavlab.roi`` uses, which is what
    makes bridged "keyhole" holes come out as holes -- the interior of a
    folded-in ring is crossed an even number of times and so stays unset,
    with no special handling here or in the caller.

    :param shapes: ``(shape_id, rgb, text_label, points_xy)`` tuples, as
        returned by ``lavlab.roi.get_shapes_as_points`` at analysis scale
    :type shapes: Iterable[tuple]
    :param shape_hw: ``(height, width)`` of the analysis image
    :type shape_hw: tuple[int, int]
    :return: ``({textValue: mask}, shape_id_map)``
    :rtype: tuple[dict[str | None, numpy.ndarray], numpy.ndarray]
    """
    from skimage import draw

    masks: dict[str | None, np.ndarray] = {}
    shape_ids = np.zeros(shape_hw, dtype=np.int64)

    for shape_id, _rgb, text_label, points in shapes:
        if len(points) < 3:
            continue
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        rr, cc = draw.polygon(ys, xs, shape=shape_hw)
        if len(rr) == 0:
            continue
        mask = masks.get(text_label)
        if mask is None:
            mask = np.zeros(shape_hw, dtype=bool)
            masks[text_label] = mask
        mask[rr, cc] = True
        shape_ids[rr, cc] = shape_id

    return masks, shape_ids


def dilate_mask(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Grow a mask by *radius_px*, for the background margin.

    :param mask: boolean mask
    :type mask: numpy.ndarray
    :param radius_px: dilation radius in mask pixels
    :type radius_px: float
    :return: the dilated mask (the input, if the radius rounds to nothing)
    :rtype: numpy.ndarray
    """
    from skimage import morphology

    radius = int(math.ceil(radius_px))
    if radius < 1 or not mask.any():
        return mask
    isotropic = getattr(morphology, "isotropic_dilation", None)
    if isotropic is not None:
        return np.asarray(isotropic(mask, radius), dtype=bool)
    return np.asarray(morphology.dilation(mask, morphology.disk(radius)), dtype=bool)


def erode_mask(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Shrink a mask by *radius_px* with a square, for ``--erode`` reporting.

    Used only to count ROIs with no room left for a tile centre; the
    per-tile test itself is :func:`center_boxes` against a summed-area
    table, which is exact at pixel boundaries.

    :param mask: boolean mask
    :type mask: numpy.ndarray
    :param radius_px: erosion radius in mask pixels
    :type radius_px: float
    :return: pixels where a ``2 * radius_px`` square around them fits in the
        mask (the input, for a radius under one pixel)
    :rtype: numpy.ndarray
    """
    from scipy import ndimage

    if radius_px < 1 or not mask.any():
        return mask
    side = max(1, int(round(2 * radius_px)))
    return ndimage.minimum_filter(
        mask.astype(np.uint8), size=side, mode="constant", cval=0
    ).astype(bool)


def tile_center(tile: GridTile) -> tuple[int, int]:
    """Return a tile's centre in level-0 pixels.

    :param tile: the grid cell
    :type tile: GridTile
    :return: ``(x, y)``
    :rtype: tuple[int, int]
    """
    return tile.x0 + tile.w0 // 2, tile.y0 + tile.h0 // 2


def center_boxes(
    tiles: Sequence[GridTile], analysis_ds: int, radius_px: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analysis-resolution squares of half-width *radius_px* around each centre.

    A tile's centre lies at least *radius_px* inside an ROI exactly when this
    square is wholly inside it -- so ``window_fractions(...) == 1`` is the
    ``--erode`` test. With the default half-tile radius the square *is* the
    tile's own window, i.e. "the tile lies wholly inside the ROI".

    :param tiles: the grid
    :type tiles: Sequence[GridTile]
    :param analysis_ds: the thumbnail's downsample factor
    :type analysis_ds: int
    :param radius_px: half-width in analysis pixels
    :type radius_px: float
    :return: ``(xs, ys, ws, hs)`` arrays in thumbnail pixels
    :rtype: tuple[numpy.ndarray, ...]
    """
    if not tiles:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty
    r0 = radius_px * analysis_ds
    cx = np.array([t.x0 + t.w0 / 2 for t in tiles], dtype=np.float64)
    cy = np.array([t.y0 + t.h0 / 2 for t in tiles], dtype=np.float64)
    side = max(1, int(round(2 * r0 / analysis_ds)))
    xs = np.rint((cx - r0) / analysis_ds).astype(np.int64)
    ys = np.rint((cy - r0) / analysis_ds).astype(np.int64)
    sides = np.full(len(tiles), side, dtype=np.int64)
    return xs, ys, sides, sides


def scale_tile(tile: GridTile, ds_total: float, size: int) -> GridTile:
    """Return the tile concentric with *tile* at another scale.

    :param tile: the primary-scale tile
    :type tile: GridTile
    :param ds_total: level-0 pixels per output pixel at the other scale
    :type ds_total: float
    :param size: output tile edge
    :type size: int
    :return: a cell with the same centre and a ``size * ds_total`` edge
    :rtype: GridTile
    """
    cx, cy = tile_center(tile)
    edge = int(round(size * ds_total))
    return GridTile(
        tile.col, tile.row, tile.xt, tile.yt, cx - edge // 2, cy - edge // 2, edge, edge
    )


def tile_fits(tile: GridTile, width0: int, height0: int) -> bool:
    """Return whether a tile lies wholly inside the slide.

    :param tile: the cell
    :type tile: GridTile
    :param width0: full-resolution width
    :type width0: int
    :param height0: full-resolution height
    :type height0: int
    :return: whether no part of it runs off the slide
    :rtype: bool
    """
    return (
        tile.x0 >= 0
        and tile.y0 >= 0
        and tile.x0 + tile.w0 <= width0
        and tile.y0 + tile.h0 <= height0
    )


@dataclass
class TileRecord:
    """One tile that survived every filter, and why it was kept."""

    tile: GridTile
    label: str
    coverage: float
    tissue_frac: float
    shape_id: int | None = None


def _dominant_shape_id(
    shape_ids: np.ndarray, x: int, y: int, w: int, h: int
) -> int | None:
    """Return a representative ROI id for a window, or None."""
    height, width = shape_ids.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    window = shape_ids[y0:y1, x0:x1]
    nonzero = window[window != 0]
    if nonzero.size == 0:
        return None
    values, counts = np.unique(nonzero, return_counts=True)
    return int(values[int(np.argmax(counts))])


def assign_labels(
    tiles: Sequence[GridTile],
    tissue_fracs: np.ndarray,
    label_fracs: dict[str, np.ndarray],
    *,
    exclude_fracs: np.ndarray | None = None,
    background_fracs: np.ndarray | None = None,
    shape_id_for: Callable[[int], int | None] | None = None,
    center_ok: dict[str, np.ndarray] | None = None,
    min_coverage: float = 0.5,
    tissue_thresh: float = 0.5,
    background_label: str | None = None,
    has_rois: bool = False,
) -> tuple[list[TileRecord], Counter]:
    """Decide each tile's class, or drop it.

    In order: a tile without enough tissue is dropped; a tile touching an
    exclusion ROI at all is dropped; a tile covered at or above
    *min_coverage* by some class -- and, with *center_ok*, whose centre lies
    in that class's eroded ROI -- takes the highest-covering one (ties
    broken by folder name, so a rerun labels it the same way); a tile that
    only failed the centre test is dropped as ``dropped_roi_edge``, never
    demoted to background; and otherwise, if a
    background label is configured, a tile that is tissue but has *zero*
    coverage from any ROI on the slide becomes background.

    That last rule is why *has_rois* matters. Annotators mark everything on
    a slide they work on -- cancer, atrophy, HGPIN, exclusions -- so tissue
    outside every ROI on an *annotated* slide is genuinely benign. On a
    slide nobody annotated, the same tissue is merely unlabelled, and
    calling it benign would poison the training set. The caller must not
    pass ``has_rois=True`` for a slide with no ROIs.

    :param tiles: the grid
    :type tiles: Sequence[GridTile]
    :param tissue_fracs: tissue fraction per tile
    :type tissue_fracs: numpy.ndarray
    :param label_fracs: folder name -> coverage fraction per tile
    :type label_fracs: dict[str, numpy.ndarray]
    :param exclude_fracs: exclusion coverage per tile, if any
    :type exclude_fracs: numpy.ndarray | None
    :param background_fracs: coverage by the margin-dilated union of every
        ROI, used to keep tiles near an annotation out of background
    :type background_fracs: numpy.ndarray | None
    :param shape_id_for: maps a tile index to the shape id that covered it
    :type shape_id_for: Callable[[int], int | None] | None
    :param center_ok: folder name -> whether each tile's centre lies inside
        that class's eroded ROI; None to skip the test (no erosion)
    :type center_ok: dict[str, numpy.ndarray] | None
    :param min_coverage: fraction of a tile that must lie in one class
    :type min_coverage: float
    :param tissue_thresh: minimum tissue fraction to keep a tile
    :type tissue_thresh: float
    :param background_label: folder for background tiles, or None
    :type background_label: str | None
    :param has_rois: whether this slide carries any ROI at all
    :type has_rois: bool
    :return: the kept tiles and a count of what happened to every tile
    :rtype: tuple[list[TileRecord], collections.Counter]
    """
    records: list[TileRecord] = []
    stats: Counter = Counter()
    names = sorted(label_fracs)

    for index, tile in enumerate(tiles):
        tissue = float(tissue_fracs[index])
        if tissue < tissue_thresh:
            stats["dropped_no_tissue"] += 1
            continue
        if exclude_fracs is not None and exclude_fracs[index] > 0:
            stats["dropped_excluded"] += 1
            continue

        best_name: str | None = None
        best_frac = 0.0
        near_edge = False
        for name in names:
            frac = float(label_fracs[name][index])
            if frac < min_coverage:
                continue
            if center_ok is not None and not center_ok[name][index]:
                near_edge = True
                continue
            if frac > best_frac:
                best_name, best_frac = name, frac

        if best_name is not None:
            shape_id = shape_id_for(index) if shape_id_for is not None else None
            records.append(TileRecord(tile, best_name, best_frac, tissue, shape_id))
            stats[best_name] += 1
            continue

        if near_edge:
            stats["dropped_roi_edge"] += 1
            continue

        if (
            background_label
            and has_rois
            and background_fracs is not None
            and background_fracs[index] <= 0
        ):
            records.append(TileRecord(tile, background_label, 0.0, tissue, None))
            stats[background_label] += 1
            continue

        stats["dropped_unlabeled"] += 1

    return records, stats


def assign_whole_labels(
    tiles: Sequence[GridTile],
    tissue_fracs: np.ndarray,
    *,
    exclude_fracs: np.ndarray | None = None,
    tissue_thresh: float = 0.5,
) -> tuple[list[TileRecord], Counter]:
    """Keep every tissue tile, under one flat label.

    :param tiles: the grid
    :type tiles: Sequence[GridTile]
    :param tissue_fracs: tissue fraction per tile
    :type tissue_fracs: numpy.ndarray
    :param exclude_fracs: exclusion coverage per tile, if any
    :type exclude_fracs: numpy.ndarray | None
    :param tissue_thresh: minimum tissue fraction to keep a tile
    :type tissue_thresh: float
    :return: the kept tiles and a count of what happened to every tile
    :rtype: tuple[list[TileRecord], collections.Counter]
    """
    records: list[TileRecord] = []
    stats: Counter = Counter()
    for index, tile in enumerate(tiles):
        tissue = float(tissue_fracs[index])
        if tissue < tissue_thresh:
            stats["dropped_no_tissue"] += 1
            continue
        if exclude_fracs is not None and exclude_fracs[index] > 0:
            stats["dropped_excluded"] += 1
            continue
        records.append(TileRecord(tile, WHOLE_LABEL, 0.0, tissue, None))
        stats[WHOLE_LABEL] += 1
    return records, stats


def sample_records(
    records: Sequence[TileRecord],
    *,
    max_tiles: int | None = None,
    max_tiles_per_label: int | None = None,
    max_background_tiles: int | None = None,
    background_label: str | None = None,
    seed: int = 0,
) -> list[TileRecord]:
    """Apply the per-label and per-slide caps, sampling at random.

    Sampling is seeded and runs over a row-major ordering, so the same
    slide with the same parameters and seed yields the same tiles on every
    machine -- a training set that silently reshuffles between runs is not
    reproducible.

    :param records: the labelled tiles
    :type records: Sequence[TileRecord]
    :param max_tiles: overall cap for the slide
    :type max_tiles: int | None
    :param max_tiles_per_label: cap for every label
    :type max_tiles_per_label: int | None
    :param max_background_tiles: cap for the background label specifically,
        which otherwise dwarfs every graded class
    :type max_background_tiles: int | None
    :param background_label: which label that cap applies to
    :type background_label: str | None
    :param seed: RNG seed
    :type seed: int
    :return: the kept tiles, back in row-major order
    :rtype: list[TileRecord]
    """
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda r: (r.tile.row, r.tile.col))

    grouped: dict[str, list[TileRecord]] = {}
    for record in ordered:
        grouped.setdefault(record.label, []).append(record)

    kept: list[TileRecord] = []
    for label in sorted(grouped):
        group = grouped[label]
        cap = max_tiles_per_label
        if background_label is not None and label == background_label:
            if max_background_tiles is not None:
                cap = max_background_tiles
        if cap is not None and len(group) > cap:
            group = rng.sample(group, cap)
        kept.extend(group)

    if max_tiles is not None and len(kept) > max_tiles:
        kept.sort(key=lambda r: (r.tile.row, r.tile.col))
        kept = rng.sample(kept, max_tiles)

    kept.sort(key=lambda r: (r.tile.row, r.tile.col))
    return kept


def _vips_to_numpy(image) -> np.ndarray:
    """Copy a pyvips image into a numpy array (uint8 only)."""
    memory = image.write_to_memory()
    return np.ndarray(
        buffer=memory, dtype=np.uint8, shape=(image.height, image.width, image.bands)
    )


def is_slow_jp2_source(path: str) -> bool:
    """Return whether *path* is a bare JPEG-2000 file.

    :param path: the source file path
    :type path: str
    :return: whether random reads from it would go through Pillow
    :rtype: bool
    """
    return path.lower().endswith(JP2_SUFFIXES)


def local_level_dims(src_path: str) -> list[tuple[int, int]]:
    """Return ``(width, height)`` per pyramid level of a source file.

    Reads the pyramid through tifffile, the same way
    ``lavlab.imaging.load_downsampled`` does, so both see the same levels
    whether the file stores them as SubIFDs (OME-TIFF) or as extra pages
    (SVS).

    :param src_path: the source file
    :type src_path: str
    :return: level dimensions, index 0 = full resolution
    :rtype: list[tuple[int, int]]
    :raises UnreadableSourceError: if no series/levels could be read
    """
    import tifffile

    with tifffile.TiffFile(src_path) as tif:
        if not tif.series:
            raise UnreadableSourceError(f"no TIFF series in '{src_path}'.")
        series = tif.series[0]
        axes = series.axes
        if "Y" not in axes or "X" not in axes:
            raise UnreadableSourceError(
                f"TIFF series in '{src_path}' has axes '{axes}' with no Y/X."
            )
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        return [
            (int(level.shape[x_axis]), int(level.shape[y_axis]))
            for level in series.levels
        ]


def open_local_level(src_path: str, level_index: int, expected_wh: tuple[int, int]):
    """Open one pyramid level for random access with pyvips.

    Pyramid levels are addressed differently depending on how the file
    stores them -- SubIFDs for OME-TIFF, extra pages for SVS -- and libvips
    exposes those as different load options. Rather than guess from the
    file's structure, both spellings are tried and the one whose dimensions
    match what tifffile reported is used.

    :param src_path: the source file
    :type src_path: str
    :param level_index: which level, 0 = full resolution
    :type level_index: int
    :param expected_wh: the ``(width, height)`` that level should have
    :type expected_wh: tuple[int, int]
    :return: a pyvips image opened with random access
    :raises UnreadableSourceError: if no spelling produced that level
    """
    import pyvips as pv

    candidates = [f"{src_path}[page={level_index},access=random]"]
    if level_index > 0:
        # A SubIFD pyramid keeps every level under page 0; subifd=-1 is the
        # full-resolution image, so level N is subifd N-1.
        candidates.insert(
            0, f"{src_path}[page=0,subifd={level_index - 1},access=random]"
        )

    failures = []
    for descriptor in candidates:
        try:
            image = pv.Image.new_from_file(descriptor)
        except pv.Error as exc:
            failures.append(f"{descriptor}: {exc}")
            continue
        if (image.width, image.height) == tuple(expected_wh):
            return image
        failures.append(
            f"{descriptor}: got {image.width}x{image.height}, "
            f"expected {expected_wh[0]}x{expected_wh[1]}"
        )

    raise UnreadableSourceError(
        f"could not open level {level_index} of '{src_path}': " + "; ".join(failures)
    )


class LocalRegionReader:
    """Tier 2: crop regions straight out of a mounted source file."""

    def __init__(self, src_path: str, plan: LevelPlan, level_wh: tuple[int, int]):
        """
        :param src_path: the source file
        :type src_path: str
        :param plan: the level and region size to read
        :type plan: LevelPlan
        :param level_wh: that level's ``(width, height)``
        :type level_wh: tuple[int, int]
        """
        self._image = open_local_level(src_path, plan.index, level_wh)
        self._plan = plan
        self._level_wh = level_wh

    def prefetch(self, regions: Sequence[tuple[int, int, int, int]]) -> None:
        """No-op: a mounted file needs no batching."""

    def read(self, x0: int, y0: int, w0: int, h0: int) -> np.ndarray:
        """Read one level-0 region, returned at the chosen pyramid level.

        :param x0: level-0 left edge
        :type x0: int
        :param y0: level-0 top edge
        :type y0: int
        :param w0: level-0 width
        :type w0: int
        :param h0: level-0 height
        :type h0: int
        :return: an ``(read_size, read_size, bands)`` uint8 array
        :rtype: numpy.ndarray
        """
        size = self._plan.read_size
        level_w, level_h = self._level_wh
        x = min(max(int(round(x0 / self._plan.downsample)), 0), max(level_w - size, 0))
        y = min(max(int(round(y0 / self._plan.downsample)), 0), max(level_h - size, 0))
        crop = self._image.crop(x, y, min(size, level_w - x), min(size, level_h - y))
        return _vips_to_numpy(crop)

    def close(self) -> None:
        self._image = None


class NetworkRegionReader:
    """Tier 3: fetch regions through OMERO's tile API.

    Fetches are batched: :meth:`prefetch` pulls every OMERO tile covering a
    whole band of the grid in one async gather across several raw-pixels
    stores, and :meth:`read` then cuts from that. One round trip per output
    tile would be six figures of round trips on a 40x whole-mount.
    """

    def __init__(self, conn, image, plan: LevelPlan, store_count: int | None = None):
        """
        :param conn: a connected gateway
        :param image: an OMERO ``ImageWrapper``
        :param plan: the level and region size to read
        :type plan: LevelPlan
        :param store_count: parallel raw-pixels stores
        :type store_count: int | None
        """
        from lavlab.omero_tiles import (
            PARALLEL_STORE_COUNT,
            _switch_group_before_stateful_service,
        )

        self._conn = conn
        self._image = image
        self._plan = plan
        self._store_count = store_count or PARALLEL_STORE_COUNT
        self._channels = list(range(image.getSizeC()))

        pixels = image.getPrimaryPixels()
        pixel_type = pixels.getPixelsType().getValue()
        if pixel_type != "uint8":
            raise TilingError(
                f"image {image.getId()} has pixel type '{pixel_type}'; network "
                "tiling only supports uint8 (RGB) images."
            )
        self._pixels_id = pixels.getId()

        _switch_group_before_stateful_service(conn, image)

        rps = conn.createRawPixelsStore()
        try:
            rps.setPixelsId(self._pixels_id, True)
            count = rps.getResolutionLevels()
            descriptions = rps.getResolutionDescriptions()
            if not 0 <= plan.index < count:
                raise TilingError(
                    f"pyramid level {plan.index} is out of range 0..{count - 1} "
                    f"for image {image.getId()}."
                )
            self._level = count - 1 - plan.index
            rps.setResolutionLevel(self._level)
            tile_w, tile_h = rps.getTileSize()
            description = descriptions[plan.index]
            self._level_wh = (int(description.sizeX), int(description.sizeY))
            if not tile_w or not tile_h:
                tile_w, tile_h = 1024, 1024
            self._tile_size = (
                min(int(tile_w), self._level_wh[0]),
                min(int(tile_h), self._level_wh[1]),
            )
        finally:
            rps.close()
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    @property
    def level_wh(self) -> tuple[int, int]:
        return self._level_wh

    def _covering(
        self, regions: Sequence[tuple[int, int, int, int]]
    ) -> list[tuple[int, int]]:
        tile_w, tile_h = self._tile_size
        level_w, level_h = self._level_wh
        needed: set[tuple[int, int]] = set()
        for x, y, w, h in regions:
            x = max(0, min(x, level_w - 1))
            y = max(0, min(y, level_h - 1))
            for ty in range(y // tile_h, min(y + h - 1, level_h - 1) // tile_h + 1):
                for tx in range(x // tile_w, min(x + w - 1, level_w - 1) // tile_w + 1):
                    needed.add((tx, ty))
        return sorted(needed)

    def prefetch(self, regions: Sequence[tuple[int, int, int, int]]) -> None:
        """Fetch every OMERO tile covering *regions*, dropping the last batch.

        :param regions: level-0 ``(x, y, w, h)`` regions about to be read
        :type regions: Sequence[tuple[int, int, int, int]]
        """
        import asyncio

        level_regions = [
            (
                int(round(x / self._plan.downsample)),
                int(round(y / self._plan.downsample)),
                self._plan.read_size,
                self._plan.read_size,
            )
            for x, y, _w, _h in regions
        ]
        keys = self._covering(level_regions)
        self._cache = {}
        if not keys:
            return
        asyncio.run(self._gather(keys))

    async def _gather(self, keys: Sequence[tuple[int, int]]) -> None:
        from lavlab.omero_asyncio import AsyncSession
        from lavlab.omero_tiles import _get_session_factory, merge_async_iters

        tile_w, tile_h = self._tile_size
        level_w, level_h = self._level_wh

        requests = []
        for tx, ty in keys:
            x = tx * tile_w
            y = ty * tile_h
            w = min(tile_w, level_w - x)
            h = min(tile_h, level_h - y)
            if w <= 0 or h <= 0:
                continue
            self._cache[(tx, ty)] = np.zeros(
                (h, w, len(self._channels)), dtype=np.uint8
            )
            for channel in self._channels:
                requests.append((0, channel, 0, (x, y, w, h)))

        if not requests:
            return

        session = AsyncSession(_get_session_factory(self._conn))
        chunks = [chunk for chunk in _chunk(requests, self._store_count) if chunk]
        jobs = [self._fetch_chunk(session, chunk) for chunk in chunks]

        async for array, (_z, channel, _t, (x, y, w, h)) in merge_async_iters(*jobs):
            self._cache[(x // tile_w, y // tile_h)][:h, :w, channel] = array

    async def _fetch_chunk(self, session, requests):
        rps = await session.createRawPixelsStore()
        closed = False
        try:
            await rps.setPixelsId(self._pixels_id, True)
            await rps.setResolutionLevel(self._level)
            for z, channel, t, (x, y, w, h) in requests:
                buf = await rps.getTile(z, channel, t, x, y, w, h)
                yield (
                    np.frombuffer(buf, dtype=np.uint8).reshape(h, w),
                    (
                        z,
                        channel,
                        t,
                        (x, y, w, h),
                    ),
                )
            await rps.close()
            closed = True
        finally:
            if not closed:
                try:
                    await rps.close()
                except Exception:
                    log.debug("rps close after abort failed", exc_info=True)

    def read(self, x0: int, y0: int, w0: int, h0: int) -> np.ndarray:
        """Read one level-0 region, returned at the chosen pyramid level.

        :param x0: level-0 left edge
        :type x0: int
        :param y0: level-0 top edge
        :type y0: int
        :param w0: level-0 width
        :type w0: int
        :param h0: level-0 height
        :type h0: int
        :return: an ``(read_size, read_size, channels)`` uint8 array
        :rtype: numpy.ndarray
        """
        size = self._plan.read_size
        level_w, level_h = self._level_wh
        x = min(max(int(round(x0 / self._plan.downsample)), 0), max(level_w - size, 0))
        y = min(max(int(round(y0 / self._plan.downsample)), 0), max(level_h - size, 0))
        width = min(size, level_w - x)
        height = min(size, level_h - y)

        missing = [
            key
            for key in self._covering([(x, y, width, height)])
            if key not in self._cache
        ]
        if missing:
            # A read outside the last prefetched band -- correct, just slower.
            import asyncio

            asyncio.run(self._gather(missing))

        tile_w, tile_h = self._tile_size
        out = np.zeros((height, width, len(self._channels)), dtype=np.uint8)
        for tx, ty in self._covering([(x, y, width, height)]):
            block = self._cache.get((tx, ty))
            if block is None:
                continue
            bx, by = tx * tile_w, ty * tile_h
            sx0, sy0 = max(x, bx), max(y, by)
            sx1 = min(x + width, bx + block.shape[1])
            sy1 = min(y + height, by + block.shape[0])
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            out[sy0 - y : sy1 - y, sx0 - x : sx1 - x] = block[
                sy0 - by : sy1 - by, sx0 - bx : sx1 - bx
            ]
        return out

    def close(self) -> None:
        self._cache = {}


def _chunk(items: Sequence, count: int) -> list[list]:
    """Split *items* into at most *count* contiguous chunks."""
    if not items:
        return []
    size = math.ceil(len(items) / max(1, count))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def resize_tile(array: np.ndarray, size: int) -> np.ndarray:
    """Resize a read region to the output tile size.

    Both tiers funnel through here rather than resizing in their own
    reader, so the resampling kernel cannot differ between a tile cut from
    a mounted file and the same tile cut over the network.

    :param array: the region as read, ``(H, W, C)`` uint8
    :type array: numpy.ndarray
    :param size: the output tile edge
    :type size: int
    :return: a ``(size, size, C)`` uint8 array
    :rtype: numpy.ndarray
    """
    if array.shape[0] == size and array.shape[1] == size:
        return array

    import pyvips as pv

    image = pv.Image.new_from_array(array)
    image = image.resize(size / image.width, vscale=size / image.height)
    return _vips_to_numpy(image)


def cut_tiles(
    reader,
    records: Sequence[TileRecord],
    size: int,
    *,
    band_rows: int = 4,
) -> Iterable[tuple[TileRecord, np.ndarray]]:
    """Yield ``(record, pixels)`` for every kept tile.

    This is the whole of the tier-independent pixel path: *reader* is any
    object with ``prefetch(regions)`` and ``read(x0, y0, w0, h0)``, so
    swapping tier 2 for tier 3 changes where bytes come from and nothing
    about which pixels end up in a tile.

    :param reader: a region reader
    :param records: the tiles to cut, in row-major order
    :type records: Sequence[TileRecord]
    :param size: the output tile edge
    :type size: int
    :param band_rows: grid rows to prefetch at a time
    :type band_rows: int
    :return: an iterator of ``(record, (size, size, C) uint8)``
    :rtype: Iterable[tuple[TileRecord, numpy.ndarray]]
    """
    band: list[TileRecord] = []
    current_band = None
    for record in records:
        band_index = record.tile.row // max(1, band_rows)
        if current_band is None:
            current_band = band_index
        if band_index != current_band:
            yield from _cut_band(reader, band, size)
            band = []
            current_band = band_index
        band.append(record)
    if band:
        yield from _cut_band(reader, band, size)


def _cut_band(reader, band: Sequence[TileRecord], size: int):
    regions = [(r.tile.x0, r.tile.y0, r.tile.w0, r.tile.h0) for r in band]
    reader.prefetch(regions)
    for record in band:
        raw = reader.read(
            record.tile.x0, record.tile.y0, record.tile.w0, record.tile.h0
        )
        yield record, resize_tile(raw, size)


def write_tile(
    array: np.ndarray,
    path: str,
    *,
    metadata: dict | None = None,
    quality: int | None = None,
) -> None:
    """Write one tile to disk, format chosen by *path*'s extension.

    Written to a hidden temp file and renamed into place, so an interrupted
    run never leaves a truncated tile that a resumed run would trust.

    *metadata* is embedded as JSON so a stray tile is still self-describing:
    a PNG ``tEXt`` chunk keyed ``lavlab`` (Pillow: ``Image.open(p).text``),
    or the EXIF ImageDescription of a JPEG (``Image.open(p).getexif()[270]``).

    :param array: ``(size, size, C)`` uint8 pixels
    :type array: numpy.ndarray
    :param path: the destination file
    :type path: str
    :param metadata: values to embed in the file, or None
    :type metadata: dict | None
    :param quality: JPEG quality (ignored for PNG)
    :type quality: int | None
    """
    import pyvips as pv

    image = pv.Image.new_from_array(array)
    is_jpeg = path.lower().endswith((".jpg", ".jpeg"))
    if metadata:
        image = image.copy()
        text = json.dumps(metadata, sort_keys=True)
        field_name = "exif-ifd0-ImageDescription" if is_jpeg else "png-comment-0-lavlab"
        image.set_type(pv.GValue.gstr_type, field_name, text)

    directory, name = os.path.split(path)
    tmp_path = os.path.join(directory, f".tmp-{name}")
    options = {"Q": int(quality)} if is_jpeg and quality else {}
    try:
        image.write_to_file(tmp_path, **options)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def tile_filename(slide_stem: str, record: TileRecord, fmt: str) -> str:
    """Build a tile's filename from its level-0 coordinates.

    :param slide_stem: the slide stem, per ``lavlab.naming.get_stem``
    :type slide_stem: str
    :param record: the tile
    :type record: TileRecord
    :param fmt: file extension without the dot
    :type fmt: str
    :return: e.g. ``N101_S08_HE_x12800_y7424.png``
    :rtype: str
    """
    return f"{slide_stem}_x{record.tile.x0}_y{record.tile.y0}.{fmt}"


def scaled_tile_filename(
    slide_stem: str, center: tuple[int, int], mpp: float, fmt: str
) -> str:
    """Build a multi-scale tile's filename from its centre and scale.

    Every scale of one tile shares the centre, so the pairing survives in the
    names alone: ``..._x76800_y112640_mpp0.5.png`` / ``..._mpp1.0.png``.

    :param slide_stem: the slide stem
    :type slide_stem: str
    :param center: level-0 ``(x, y)`` centre
    :type center: tuple[int, int]
    :param mpp: this crop's micrometres per pixel
    :type mpp: float
    :param fmt: file extension without the dot
    :type fmt: str
    :return: e.g. ``N101_S08_HE_x76800_y112640_mpp0.5.png``
    :rtype: str
    """
    return f"{slide_stem}_x{center[0]}_y{center[1]}_mpp{float(mpp)}.{fmt}"


@dataclass
class TileParams:
    """Every setting that changes what a slide's tiles look like.

    Written to ``tile_params.json`` beside the manifest so ``--skip-existing``
    can tell "already tiled the way I was asked to" from "already tiled,
    but differently".
    """

    mode: str = "roi"
    include_all: bool = False
    text_filter: list[str] = field(default_factory=list)
    mpp: float | None = 0.5
    downsample: float | None = None
    size: int = 224
    overlap: int = 0
    min_coverage: float = 0.5
    tissue_thresh: float = 0.5
    fmt: str = "png"
    coords_only: bool = False
    labels: str | None = None
    exclude_text: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_TEXT))
    background_label: str | None = DEFAULT_BACKGROUND_LABEL
    background_margin_um: float = 200.0
    max_background_tiles: int | None = 2000
    max_tiles: int | None = None
    max_tiles_per_label: int | None = None
    seed: int = 0
    # Extra concentric output scales (mpp), finest first; None = just *mpp*.
    # The finest one is *mpp* itself and decides the grid and labels.
    scales: list[float] | None = None
    # --erode: None = half a tile (the default), 0 = off, else micrometres.
    erode_um: float | None = None
    quality: int | None = None

    def output_mpps(self) -> list[float | None]:
        """Every scale this run writes, finest (the labelling scale) first.

        :return: the mpp of each output scale; ``[None]`` for --downsample
        :rtype: list[float | None]
        """
        if self.scales:
            return sorted(float(s) for s in self.scales)
        return [self.mpp]

    def signature(self) -> dict:
        """Return only the fields that affect the output.

        :return: a comparable dict, free of version and timestamp
        :rtype: dict
        """
        data = asdict(self)
        data["text_filter"] = sorted(str(t).lower() for t in self.text_filter)
        data["exclude_text"] = sorted(str(t).lower() for t in self.exclude_text)
        return data

    def to_json_dict(self, **extra) -> dict:
        """Return the full record written to ``tile_params.json``.

        :param extra: additional per-run fields (tier, counts, ...)
        :return: the JSON document
        :rtype: dict
        """
        return {
            "lavlab_version": __version__,
            "generated_at": datetime.now(UTC).isoformat(),
            "params": self.signature(),
            **extra,
        }


def read_tile_params(slide_dir: str) -> dict | None:
    """Read a slide's ``tile_params.json``, or None if absent/unreadable.

    :param slide_dir: the slide's output directory
    :type slide_dir: str
    :return: the parsed document, or None
    :rtype: dict | None
    """
    path = os.path.join(slide_dir, PARAMS_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        log.warning("Could not read '%s'; treating the slide as not yet tiled.", path)
        return None


def slide_is_complete(slide_dir: str) -> bool:
    """Return whether a slide directory holds a finished run.

    Both files must be present: the manifest is renamed into place only
    once every tile is written, so its presence is the completion marker.

    :param slide_dir: the slide's output directory
    :type slide_dir: str
    :return: whether the slide looks done
    :rtype: bool
    """
    return os.path.isfile(os.path.join(slide_dir, MANIFEST_NAME)) and os.path.isfile(
        os.path.join(slide_dir, PARAMS_NAME)
    )


def params_match(saved: dict | None, params: TileParams) -> bool:
    """Return whether a saved run used the same settings as *params*.

    :param saved: the parsed ``tile_params.json``, or None
    :type saved: dict | None
    :param params: the settings this run would use
    :type params: TileParams
    :return: whether the two agree on every output-affecting field
    :rtype: bool
    """
    if not saved:
        return False
    return saved.get("params") == params.signature()


def write_params(slide_dir: str, params: TileParams, **extra) -> None:
    """Write ``tile_params.json`` for a slide.

    :param slide_dir: the slide's output directory
    :type slide_dir: str
    :param params: the settings used
    :type params: TileParams
    :param extra: additional per-run fields
    """
    path = os.path.join(slide_dir, PARAMS_NAME)
    with open(path, "w") as fh:
        json.dump(params.to_json_dict(**extra), fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_manifest(slide_dir: str, rows: Sequence[dict]) -> str:
    """Write ``manifest.csv`` atomically.

    Written to a temp file in the same directory and renamed into place, so
    a run that dies partway through never leaves behind a manifest that
    ``--skip-existing`` would mistake for a finished slide.

    :param slide_dir: the slide's output directory
    :type slide_dir: str
    :param rows: one dict per tile, keyed by :data:`MANIFEST_COLUMNS`
    :type rows: Sequence[dict]
    :return: the manifest's path
    :rtype: str
    """
    final_path = os.path.join(slide_dir, MANIFEST_NAME)
    fd, tmp_path = tempfile.mkstemp(dir=slide_dir, prefix=".manifest-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return final_path


def read_manifest(slide_dir: str) -> list[dict]:
    """Read a slide's ``manifest.csv``, or ``[]`` if absent.

    :param slide_dir: the slide's output directory
    :type slide_dir: str
    :return: one dict per row
    :rtype: list[dict]
    """
    path = os.path.join(slide_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _manifest_key(row: dict) -> tuple:
    if row.get("tile_path"):
        return ("path", str(row["tile_path"]))
    return tuple(
        str(row.get(col, ""))
        for col in ("omero_image_id", "label", "x", "y", "mpp", "size")
    )


def append_manifest(manifest_path: str, rows: Sequence[dict], out_root: str) -> int:
    """Append one slide's rows to a combined, cross-run manifest.

    ``--manifest`` accumulates every slide of every run pointed at it into
    one CSV, so a dataset extracted over several commands (one per class,
    per group, per scale) ends up described by a single file. ``tile_path``
    is rewritten relative to the manifest's own directory, so the manifest
    and the tiles can be moved together.

    Safe for concurrent batch workers (an exclusive ``flock`` around the
    read-and-append), and idempotent: rows already present -- from a resumed
    or repeated run -- are not written twice.

    :param manifest_path: the combined manifest
    :type manifest_path: str
    :param rows: the slide's rows, ``tile_path`` relative to *out_root*
    :type rows: Sequence[dict]
    :param out_root: the ``-o`` directory those paths are relative to
    :type out_root: str
    :return: the number of rows actually appended
    :rtype: int
    :raises TilingError: if the file exists with different columns
    """
    import fcntl

    manifest_path = os.path.abspath(manifest_path)
    manifest_dir = os.path.dirname(manifest_path)
    os.makedirs(manifest_dir, exist_ok=True)

    rebased = []
    for row in rows:
        row = dict(row)
        if row.get("tile_path"):
            absolute = os.path.join(os.path.abspath(out_root), row["tile_path"])
            row["tile_path"] = os.path.relpath(absolute, manifest_dir)
        rebased.append(row)

    with open(manifest_path, "a+", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            if header is not None and tuple(header) != MANIFEST_COLUMNS:
                raise TilingError(
                    f"--manifest '{manifest_path}' has different columns from this "
                    "lavlab version's; point --manifest at a new file."
                )
            seen = {_manifest_key(existing) for existing in reader}
            fresh = [row for row in rebased if _manifest_key(row) not in seen]

            fh.seek(0, os.SEEK_END)
            writer = csv.DictWriter(
                fh, fieldnames=list(MANIFEST_COLUMNS), extrasaction="ignore"
            )
            if header is None:
                writer.writeheader()
            writer.writerows(fresh)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return len(fresh)


@dataclass
class SlideResult:
    """What happened to one slide, for the batch summary."""

    image_id: int
    status: str
    tier: str | None = None
    label_counts: Counter = field(default_factory=Counter)
    stats: Counter = field(default_factory=Counter)
    written: int = 0
    slide_dir: str | None = None
    reason: str | None = None
    dry_run: bool = False
    # Tiles per (label, OMERO ROI id), after sampling.
    roi_counts: Counter = field(default_factory=Counter)
    # Selected ROIs on the slide, and those --erode shrank to nothing.
    roi_total: int = 0
    lost_rois: list[tuple[str, int | None]] = field(default_factory=list)
    # Tiles per label before the per-label / per-slide caps.
    available: Counter = field(default_factory=Counter)
    reused: int = 0
    # The --erode inset actually applied, in micrometres (0 = off; None when
    # the image has no pixel size to convert with).
    erode_um: float | None = None


def image_pixel_sizes(image) -> tuple[float | None, float | None]:
    """Read an image's physical pixel size in micrometres.

    Tolerates both what a bare ``ImageWrapper`` returns and the units
    wrapper older omero-py handed back, and treats any failure as "not
    recorded" rather than propagating -- a missing pixel size is a normal
    condition here, answered by ``--downsample``.

    :param image: an OMERO ``ImageWrapper``
    :return: ``(x, y)`` in um, either of which may be None
    :rtype: tuple[float | None, float | None]
    """

    def _value(getter):
        try:
            value = getter()
        except Exception:
            return None
        if value is None:
            return None
        if hasattr(value, "getValue"):
            try:
                value = value.getValue()
            except Exception:
                return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    return _value(image.getPixelSizeX), _value(image.getPixelSizeY)


def choose_tier(
    conn, image_id: int, *, force_local: bool = False
) -> tuple[str, str | None]:
    """Decide where this slide's pixels should come from.

    :param conn: a connected gateway
    :param image_id: the OMERO image ID
    :type image_id: int
    :param force_local: read a JPEG-2000 source locally anyway
    :type force_local: bool
    :return: ``("local" | "network", source path or None)``
    :rtype: tuple[str, str | None]
    """
    from lavlab.omero_client import get_source_file_path

    src_path = get_source_file_path(conn, image_id)
    if src_path is None or not os.path.exists(src_path):
        return "network", src_path

    if is_slow_jp2_source(src_path) and not force_local:
        log.warning(
            "Image %d: local source '%s' is JPEG-2000. The bundled libvips has "
            "no jp2k loader, so every one of this slide's tiles would be cut by "
            "decoding the whole codestream through Pillow. Using tier 3 (OMERO "
            "tile API) instead; pass --force-local to override.",
            image_id,
            src_path,
        )
        return "network", src_path

    return "local", src_path


def _local_source_errors() -> tuple:
    """Error types that mean "this local source can't serve us".

    Mirrors ``lavlab.large_recon._LOCAL_SOURCE_ERRORS`` so tier 2 falls
    forward on the same conditions ``lr`` does, plus this module's own
    :class:`UnreadableSourceError`.
    """
    import pyvips as pv

    return (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        ImportError,
        MemoryError,
        pv.Error,
        UnreadableSourceError,
    )


def network_level_dims(conn, image) -> list[tuple[int, int]]:
    """Return ``(width, height)`` per pyramid level as OMERO reports them.

    Ordered full resolution first, matching :func:`local_level_dims`, so
    the same :func:`select_level` call picks the same level on either tier.

    :param conn: a connected gateway
    :param image: an OMERO ``ImageWrapper``
    :return: level dimensions, index 0 = full resolution
    :rtype: list[tuple[int, int]]
    """
    rps = conn.createRawPixelsStore()
    try:
        rps.setPixelsId(image.getPrimaryPixels().getId(), True)
        return [(int(d.sizeX), int(d.sizeY)) for d in rps.getResolutionDescriptions()]
    finally:
        rps.close()


def _load_thumbnail_local(src_path: str, analysis_ds: int) -> np.ndarray:
    from lavlab.imaging import load_downsampled

    return _vips_to_numpy(load_downsampled(src_path, analysis_ds))


def _load_thumbnail_network(conn, image, analysis_ds: int) -> np.ndarray:
    from lavlab.omero_tiles import generate_over_network

    return _vips_to_numpy(generate_over_network(conn, image, analysis_ds))


def erode_radius_px(
    params: TileParams,
    tile0: int,
    analysis_ds: int,
    pixel_size_x: float | None,
) -> float:
    """Convert ``--erode`` into an erosion radius in analysis pixels.

    :param params: the run's settings
    :type params: TileParams
    :param tile0: tile edge in level-0 pixels
    :type tile0: int
    :param analysis_ds: the thumbnail's downsample factor
    :type analysis_ds: int
    :param pixel_size_x: physical pixel size, or None
    :type pixel_size_x: float | None
    :return: the radius (0 means no erosion)
    :rtype: float
    """
    half_tile = (tile0 / 2) / analysis_ds
    if params.erode_um is None:
        return half_tile
    if params.erode_um <= 0:
        return 0.0
    if not pixel_size_x:
        log.warning(
            "No physical pixel size recorded, so --erode %.1f um cannot be converted "
            "to pixels; eroding by half a tile instead.",
            params.erode_um,
        )
        return half_tile
    return params.erode_um / (pixel_size_x * analysis_ds)


def analyze_slide(
    tiles: Sequence[GridTile],
    thumbnail: np.ndarray,
    shapes: Sequence[tuple] | None,
    analysis_ds: int,
    params: TileParams,
    *,
    lookup: dict[str, str] | None = None,
    pixel_size_x: float | None = None,
    details: dict | None = None,
) -> tuple[list[TileRecord], Counter, bool]:
    """Decide every tile's fate at analysis resolution.

    All the per-slide thinking happens here, on a thumbnail: tissue, ROI
    coverage, ROI erosion, exclusions and the background rule. No
    full-resolution pixel is touched, which is the only reason tiling a
    220k x 260k slide is tractable.

    :param tiles: the grid
    :type tiles: Sequence[GridTile]
    :param thumbnail: the analysis-resolution image
    :type thumbnail: numpy.ndarray
    :param shapes: ``lavlab.roi.get_shapes_as_points`` output at analysis
        scale, or None for a slide with no ROIs
    :type shapes: Sequence[tuple] | None
    :param analysis_ds: the thumbnail's downsample factor
    :type analysis_ds: int
    :param params: the run's settings
    :type params: TileParams
    :param lookup: a prebuilt alias lookup, or None to load one
    :type lookup: dict[str, str] | None
    :param pixel_size_x: physical pixel size, for the background margin and
        ``--erode``
    :type pixel_size_x: float | None
    :param details: if given, filled with ``selected_shapes`` (shape id ->
        class, for every annotation this run labels from), ``lost_shapes``
        (the subset ``--erode`` shrank to nothing) and ``erode_px``
    :type details: dict | None
    :return: ``(records, stats, has_rois)``
    :rtype: tuple[list[TileRecord], collections.Counter, bool]
    """
    shape_list = list(shapes or [])
    has_rois = bool(shape_list)
    thumb_hw = (int(thumbnail.shape[0]), int(thumbnail.shape[1]))
    if details is None:
        details = {}
    details.setdefault("selected_shapes", {})
    details.setdefault("lost_shapes", {})
    details.setdefault("erode_px", 0.0)

    xs, ys, ws, hs = tile_windows(tiles, analysis_ds)
    tissue_fracs = window_fractions(
        integral_image(tissue_mask(thumbnail)), xs, ys, ws, hs
    )

    masks, shape_ids = rasterize_shapes(shape_list, thumb_hw)

    exclude_set = {str(t).strip().lower() for t in params.exclude_text}
    exclude_mask: np.ndarray | None = None
    for text, mask in masks.items():
        if text is not None and text.strip().lower() in exclude_set:
            exclude_mask = (
                mask.copy() if exclude_mask is None else (exclude_mask | mask)
            )
    exclude_fracs = (
        window_fractions(integral_image(exclude_mask), xs, ys, ws, hs)
        if exclude_mask is not None
        else None
    )

    if params.mode == "whole":
        records, stats = assign_whole_labels(
            tiles,
            tissue_fracs,
            exclude_fracs=exclude_fracs,
            tissue_thresh=params.tissue_thresh,
        )
        return records, stats, has_rois

    if lookup is None:
        lookup = build_alias_lookup(load_label_map(params.labels))

    warned: set[str] = set()

    def resolve(text: str | None) -> str | None:
        """The class a textValue labels tiles as in this run, or None."""
        if text is not None and text.strip().lower() in exclude_set:
            return None
        # Always resolve to a folder, so a -t value naming a textValue the
        # label map has never heard of still selects it; the unmapped
        # warning is only interesting in --all mode, where nobody asked
        # for that value by name.
        folder = map_text_value(
            text,
            lookup,
            include_all=True,
            warned=warned if params.include_all else None,
        )
        if folder is None:
            return None
        if not params.include_all and not selects_label(
            folder, text, params.text_filter
        ):
            return None
        return folder

    label_masks: dict[str, np.ndarray] = {}
    for text, mask in masks.items():
        folder = resolve(text)
        if folder is None:
            continue
        existing = label_masks.get(folder)
        label_masks[folder] = mask.copy() if existing is None else (existing | mask)

    selected_shapes: dict[int, str] = {}
    for shape_id, _rgb, text, _points in shape_list:
        folder = resolve(text)
        if folder is not None:
            selected_shapes[int(shape_id)] = folder
    details["selected_shapes"] = selected_shapes

    label_integs = {name: integral_image(mask) for name, mask in label_masks.items()}
    label_fracs = {
        name: window_fractions(integ, xs, ys, ws, hs)
        for name, integ in label_integs.items()
    }

    # --erode: a tile may only take a class if its centre lies in that
    # class's ROI shrunk by the radius. Eroding the per-class union (not each
    # shape alone) keeps two touching same-class annotations from growing a
    # false boundary between them.
    center_ok: dict[str, np.ndarray] | None = None
    tile0 = tiles[0].w0 if tiles else 0
    radius = erode_radius_px(params, tile0, analysis_ds, pixel_size_x)
    details["erode_px"] = radius
    if radius >= 1 and label_masks:
        boxes = center_boxes(tiles, analysis_ds, radius)
        center_ok = {}
        lost: dict[int, str] = {}
        for name, mask in label_masks.items():
            center_ok[name] = window_fractions(label_integs[name], *boxes) >= 1.0 - 1e-9
            eroded = erode_mask(mask, radius)
            ours = {sid for sid, folder in selected_shapes.items() if folder == name}
            before = ours.intersection(np.unique(shape_ids[mask]).tolist())
            after = ours.intersection(np.unique(shape_ids[eroded]).tolist())
            for sid in before - after:
                lost[sid] = name
        details["lost_shapes"] = lost

    background_fracs = None
    if params.background_label and has_rois:
        union = np.zeros(thumb_hw, dtype=bool)
        for mask in masks.values():
            union |= mask
        margin = 0.0
        if params.background_margin_um > 0:
            if pixel_size_x:
                margin = params.background_margin_um / (pixel_size_x * analysis_ds)
            else:
                log.warning(
                    "No physical pixel size recorded, so --background-margin %.1f um "
                    "cannot be converted to pixels; background tiles are taken right "
                    "up to the ROI edge.",
                    params.background_margin_um,
                )
        background_fracs = window_fractions(
            integral_image(dilate_mask(union, margin)), xs, ys, ws, hs
        )

    def shape_id_for(index: int) -> int | None:
        return _dominant_shape_id(
            shape_ids, int(xs[index]), int(ys[index]), int(ws[index]), int(hs[index])
        )

    records, stats = assign_labels(
        tiles,
        tissue_fracs,
        label_fracs,
        exclude_fracs=exclude_fracs,
        background_fracs=background_fracs,
        shape_id_for=shape_id_for,
        center_ok=center_ok,
        min_coverage=params.min_coverage,
        tissue_thresh=params.tissue_thresh,
        background_label=params.background_label,
        has_rois=has_rois,
    )
    return records, stats, has_rois


def group_rois(
    shape_ids: Iterable[int], shape_rois: dict[int, int]
) -> dict[int | None, set[int]]:
    """Group shape ids by the OMERO ROI holding them.

    :param shape_ids: the shapes
    :type shape_ids: Iterable[int]
    :param shape_rois: shape id -> ROI id, from ``lavlab.roi.shape_roi_ids``
    :type shape_rois: dict[int, int]
    :return: ROI id -> its shapes; a shape of unknown ROI is its own group,
        keyed by its negated shape id so it cannot collide with a real ROI
    :rtype: dict[int | None, set[int]]
    """
    groups: dict[int | None, set[int]] = {}
    for sid in shape_ids:
        key = shape_rois.get(sid, -sid)
        groups.setdefault(key, set()).add(sid)
    return groups


def _manifest_row(
    image_id,
    subject,
    slide_stem,
    record,
    path,
    plan,
    params,
    mpp_effective,
    tier,
    roi_id,
) -> dict:
    x, y = tile_center(record.tile)
    return {
        "tile_path": path,
        "omero_image_id": image_id,
        "case_id": subject,
        "slide_id": slide_stem,
        "roi_id": "" if roi_id is None else roi_id,
        "label": record.label,
        "x": x,
        "y": y,
        "mpp": "" if mpp_effective is None else round(mpp_effective, 6),
        "size": params.size,
        "shape_id": "" if record.shape_id is None else record.shape_id,
        "x0": record.tile.x0,
        "y0": record.tile.y0,
        "w0": record.tile.w0,
        "h0": record.tile.h0,
        "level": plan.index,
        "coverage": round(record.coverage, 6),
        "tissue_frac": round(record.tissue_frac, 6),
        "tier": tier,
    }


def tile_slide(
    conn,
    image,
    params: TileParams,
    out_root: str,
    *,
    override: bool = False,
    skip_existing: bool = False,
    force_local: bool = False,
    manifest_path: str | None = None,
    dry_run: bool = False,
    store_count: int | None = None,
) -> SlideResult:
    """Tile one slide into ``<out_root>/<subject>/<stem>/``.

    :param conn: a connected gateway, already switched into the image's group
    :param image: an OMERO ``ImageWrapper``
    :param params: the run's settings
    :type params: TileParams
    :param out_root: the output root directory
    :type out_root: str
    :param override: re-tile a slide whose recorded parameters differ
    :type override: bool
    :param skip_existing: skip slides that already look finished, and resume
        an interrupted slide from the tiles it already wrote
    :type skip_existing: bool
    :param force_local: read a JPEG-2000 source locally anyway
    :type force_local: bool
    :param manifest_path: also append this slide's rows to a combined
        cross-run manifest
    :type manifest_path: str | None
    :param dry_run: decide and count every tile, but read and write nothing
    :type dry_run: bool
    :param store_count: parallel raw-pixels stores for network reads
    :type store_count: int | None
    :return: what happened
    :rtype: SlideResult
    :raises TilingError: for a slide that cannot be tiled as asked
    """
    image_id = int(image.getId())
    name = image.getName()
    slide_stem = get_stem(name)
    subject = subject_dir_name(name)
    slide_dir = os.path.join(out_root, subject, slide_stem)

    if skip_existing and not dry_run and slide_is_complete(slide_dir):
        saved = read_tile_params(slide_dir)
        if params_match(saved, params):
            log.info(
                "Image %d: already tiled with these parameters, skipping.", image_id
            )
            if manifest_path:
                append_manifest(manifest_path, read_manifest(slide_dir), out_root)
            return SlideResult(
                image_id,
                "skipped",
                slide_dir=slide_dir,
                reason="already tiled with these parameters",
            )
        if not override:
            log.warning(
                "Image %d: '%s' already holds tiles cut with *different* parameters; "
                "skipping rather than mixing two settings in one directory. Re-run "
                "with --override to re-tile it.",
                image_id,
                slide_dir,
            )
            return SlideResult(
                image_id, "skipped", slide_dir=slide_dir, reason="parameter mismatch"
            )

    # An interrupted run records its parameters up front but never gets as far
    # as the manifest; with the same parameters, its finished tiles are reused.
    resume = (
        skip_existing
        and not dry_run
        and not params.coords_only
        and not slide_is_complete(slide_dir)
        and params_match(read_tile_params(slide_dir), params)
    )

    width0 = int(image.getSizeX())
    height0 = int(image.getSizeY())
    pixel_x, pixel_y = image_pixel_sizes(image)
    scale_ds: list[tuple[float | None, float]] = []
    for mpp_s in params.output_mpps():
        scale_ds.append(
            (
                mpp_s,
                resolve_total_downsample(
                    mpp=mpp_s,
                    downsample=params.downsample,
                    pixel_size_x=pixel_x,
                    pixel_size_y=pixel_y,
                ),
            )
        )
    ds_total = scale_ds[0][1]
    multi_scale = len(scale_ds) > 1

    tiles = build_grid(width0, height0, params.size, params.overlap, ds_total)
    if not tiles:
        log.warning(
            "Image %d: %dx%d is smaller than a single %d px tile at this scale; "
            "nothing to do.",
            image_id,
            width0,
            height0,
            params.size,
        )
        return SlideResult(
            image_id,
            "done",
            slide_dir=slide_dir,
            reason="slide smaller than one tile",
            dry_run=dry_run,
        )

    analysis_ds = analysis_downsample(tiles[0].w0)
    tier, src_path = choose_tier(conn, image_id, force_local=force_local)

    thumbnail = None
    level_dims: list[tuple[int, int]] = []
    if tier == "local":
        try:
            level_dims = local_level_dims(src_path)
            thumbnail = _load_thumbnail_local(src_path, analysis_ds)
        except _local_source_errors() as exc:
            log.warning(
                "Image %d: tier 2 (local source) failed for '%s' -- %s: %s. Falling "
                "back to tier 3 (OMERO tile API), which is much slower. If this "
                "happens for every image, the mounted repository is not readable "
                "and should be investigated.",
                image_id,
                src_path,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            tier = "network"
            thumbnail = None

    if thumbnail is None:
        level_dims = network_level_dims(conn, image)
        thumbnail = _load_thumbnail_network(conn, image, analysis_ds)

    level_widths = [w for w, _h in level_dims]
    plans = [select_level(level_widths, width0, ds, params.size) for _m, ds in scale_ds]
    plan = plans[0]
    log.info(
        "Image %d: %d grid tiles, tier %s, pyramid level %d (x%.2f), analysis 1/%d.",
        image_id,
        len(tiles),
        tier,
        plan.index,
        plan.downsample,
        analysis_ds,
    )

    from lavlab.roi import get_shapes_as_points, shape_roi_ids

    shapes = (
        get_shapes_as_points(image, img_downsample=analysis_ds, include_all=True) or []
    )

    if params.mode == "roi" and not shapes:
        log.warning(
            "Image %d: no ROIs at all. An unannotated slide is not a benign slide, "
            "so it is skipped rather than tiled as background.",
            image_id,
        )
        return SlideResult(
            image_id,
            "unannotated",
            tier=tier,
            slide_dir=slide_dir,
            reason="no ROIs",
            dry_run=dry_run,
        )

    shape_rois = shape_roi_ids(image) if params.mode == "roi" else {}

    details: dict = {}
    records, stats, _has_rois = analyze_slide(
        tiles,
        thumbnail,
        shapes,
        analysis_ds,
        params,
        pixel_size_x=pixel_x,
        details=details,
    )
    del thumbnail

    if multi_scale:
        # Every scale is cut around the same centre, so a centre near the
        # slide edge can fit at the finest scale and run off at a coarser one.
        # Drop it rather than pad: a padded context crop is not real context.
        fitting = []
        for record in records:
            if all(
                tile_fits(scale_tile(record.tile, ds, params.size), width0, height0)
                for _m, ds in scale_ds[1:]
            ):
                fitting.append(record)
            else:
                stats["dropped_scale_edge"] += 1
        records = fitting

    available = Counter(r.label for r in records)
    records = sample_records(
        records,
        max_tiles=params.max_tiles,
        max_tiles_per_label=params.max_tiles_per_label,
        max_background_tiles=params.max_background_tiles,
        background_label=params.background_label,
        seed=params.seed,
    )

    if stats.get("dropped_excluded"):
        log.info(
            "Image %d: %d tiles dropped by exclusion ROIs.",
            image_id,
            stats["dropped_excluded"],
        )

    # Report ROIs, not shapes: that is the unit annotators think in and the
    # unit a dataset is split by.
    selected = details["selected_shapes"]
    lost_shapes = details["lost_shapes"]
    roi_groups = group_rois(selected, shape_rois)
    lost_rois = sorted(
        (
            (selected[next(iter(sids))], roi_id if roi_id >= 0 else None)
            for roi_id, sids in roi_groups.items()
            if sids and all(sid in lost_shapes for sid in sids)
        ),
        key=lambda item: (item[0], item[1] or 0),
    )
    if lost_rois:
        log.info(
            "Image %d: --erode (%.1f analysis px) left %d of %d ROIs with no room "
            "for a tile centre -- they are smaller than about one tile: %s",
            image_id,
            details["erode_px"],
            len(lost_rois),
            len(roi_groups),
            ", ".join(f"{label}:{rid}" for label, rid in lost_rois),
        )

    roi_counts: Counter = Counter()
    for record in records:
        if record.shape_id is not None:
            roi_counts[(record.label, shape_rois.get(record.shape_id))] += 1
    # List every selected ROI, including the ones that produced nothing --
    # those are the ones worth seeing in a --dry-run.
    for roi_key, sids in roi_groups.items():
        label = selected[next(iter(sids))]
        roi_counts.setdefault((label, roi_key if roi_key >= 0 else None), 0)
    label_counts = Counter(r.label for r in records)

    erode_um = (
        details["erode_px"] * analysis_ds * pixel_x
        if pixel_x
        else (0.0 if not details["erode_px"] else None)
    )

    def mpp_effective(mpp_s: float | None, ds_s: float) -> float | None:
        if mpp_s is not None:
            return mpp_s
        return pixel_x * ds_s if pixel_x else None

    common = {
        "tier": tier,
        "label_counts": label_counts,
        "stats": stats,
        "slide_dir": slide_dir,
        "roi_counts": roi_counts,
        "roi_total": len(roi_groups),
        "lost_rois": lost_rois,
        "available": available,
        "erode_um": erode_um,
    }

    if dry_run:
        log.info(
            "Image %d: dry run -- would write %d tile location(s) x %d scale(s) (%s).",
            image_id,
            len(records),
            len(scale_ds),
            ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items())) or "none",
        )
        return SlideResult(image_id, "done", dry_run=True, **common)

    os.makedirs(slide_dir, exist_ok=True)
    # This directory is about to be rewritten: a manifest left over from an
    # earlier run must not make a half-finished new one look complete.
    stale_manifest = os.path.join(slide_dir, MANIFEST_NAME)
    if os.path.exists(stale_manifest):
        os.remove(stale_manifest)
    identity = {"image_id": image_id, "subject": subject, "slide_stem": slide_stem}
    if not params.coords_only:
        write_params(slide_dir, params, status="in_progress", **identity)

    rows: list[dict] = []
    written = 0
    reused = 0

    for scale_index, (mpp_s, ds_s) in enumerate(scale_ds):
        scale_plan = plans[scale_index]
        mpp_out = mpp_effective(mpp_s, ds_s)
        scaled = (
            list(records)
            if scale_index == 0
            else [
                replace(r, tile=scale_tile(r.tile, ds_s, params.size)) for r in records
            ]
        )

        def row_for(record, path, _plan=scale_plan, _mpp=mpp_out):
            return _manifest_row(
                image_id,
                subject,
                slide_stem,
                record,
                path,
                _plan,
                params,
                _mpp,
                tier,
                shape_rois.get(record.shape_id)
                if record.shape_id is not None
                else None,
            )

        if params.coords_only:
            rows.extend(row_for(record, "") for record in scaled)
            continue

        scale_rows: list[dict | None] = [None] * len(scaled)
        todo: list[tuple[int, TileRecord, str]] = []
        for index, record in enumerate(scaled):
            label_dir = os.path.join(slide_dir, record.label)
            if multi_scale:
                filename = scaled_tile_filename(
                    slide_stem, tile_center(record.tile), mpp_s, params.fmt
                )
            else:
                filename = tile_filename(slide_stem, record, params.fmt)
            tile_path = os.path.join(label_dir, filename)
            if resume and os.path.isfile(tile_path):
                scale_rows[index] = row_for(
                    record, os.path.relpath(tile_path, out_root)
                )
                reused += 1
            else:
                todo.append((index, record, tile_path))

        if todo:
            if tier == "local":
                reader = LocalRegionReader(
                    src_path, scale_plan, level_dims[scale_plan.index]
                )
            elif store_count:
                reader = NetworkRegionReader(
                    conn, image, scale_plan, store_count=store_count
                )
            else:
                reader = NetworkRegionReader(conn, image, scale_plan)
            try:
                cut = cut_tiles(
                    reader, [record for _i, record, _p in todo], params.size
                )
                for (index, record, tile_path), (_rec, pixels) in zip(
                    todo, cut, strict=True
                ):
                    os.makedirs(os.path.dirname(tile_path), exist_ok=True)
                    row = row_for(record, os.path.relpath(tile_path, out_root))
                    metadata = {
                        key: row[key]
                        for key in (
                            "omero_image_id",
                            "case_id",
                            "slide_id",
                            "roi_id",
                            "shape_id",
                            "label",
                            "x",
                            "y",
                            "mpp",
                            "size",
                        )
                    }
                    metadata["lavlab_version"] = __version__
                    write_tile(
                        pixels, tile_path, metadata=metadata, quality=params.quality
                    )
                    scale_rows[index] = row
                    written += 1
            finally:
                reader.close()

        rows.extend(row for row in scale_rows if row is not None)

    write_manifest(slide_dir, rows)
    write_params(
        slide_dir,
        params,
        status="complete",
        **identity,
        tier=tier,
        level=plan.index,
        levels={
            str(mpp_s): p.index for (mpp_s, _ds), p in zip(scale_ds, plans, strict=True)
        },
        total_downsample=round(ds_total, 6),
        analysis_downsample=analysis_ds,
        grid_tiles=len(tiles),
        label_counts=dict(sorted(label_counts.items())),
        drop_stats={k: v for k, v in sorted(stats.items()) if k.startswith("dropped_")},
        rois_total=len(roi_groups),
        rois_lost_to_erosion=len(lost_rois),
        erode_um_effective=None if erode_um is None else round(erode_um, 3),
    )
    if manifest_path:
        append_manifest(manifest_path, rows, out_root)

    if reused:
        log.info(
            "Image %d: resumed -- reused %d tiles already on disk.", image_id, reused
        )
    log.info(
        "Image %d: wrote %d tiles to '%s' (%s).",
        image_id,
        len(rows),
        slide_dir,
        ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items())) or "none",
    )
    return SlideResult(image_id, "done", written=written, reused=reused, **common)
