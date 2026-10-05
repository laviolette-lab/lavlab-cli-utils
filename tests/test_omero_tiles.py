# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for lavlab.omero_tiles -- skipped if pyvips isn't installed (it's
in the `dev` extra, not required just to import the rest of the package).

No real OMERO/Ice connection is used anywhere here -- resolution-level
selection is tested against a fake RawPixelsStore, and the async merge is
tested against fake async generators standing in for tile-fetch chunks.
"""

from __future__ import annotations

import asyncio

import pytest

try:
    # pyvips can fail with OSError (not ImportError) when the native
    # libvips library itself is missing, which pytest.importorskip doesn't
    # catch -- handle both so this file is skipped, not errored, in an
    # environment without libvips installed.
    import pyvips  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"pyvips unavailable: {exc}", allow_module_level=True)

from lavlab.omero_tiles import (
    _chunkify,
    closest_resolution_level,
    create_full_tile_list,
    create_tile_list_2d,
    downsampled_xy,
    merge_async_iters,
)


class _FakeImage:
    def __init__(self, size_x, size_y):
        self._size_x = size_x
        self._size_y = size_y

    def getSizeX(self):
        return self._size_x

    def getSizeY(self):
        return self._size_y


def test_downsampled_xy_rounds_like_imaging_module():
    assert downsampled_xy(_FakeImage(1000, 333), 10) == (100, 33)


def test_downsampled_xy_never_returns_zero():
    assert downsampled_xy(_FakeImage(5, 5), 100) == (1, 1)


class _Desc:
    def __init__(self, size_x, size_y):
        self.sizeX = size_x
        self.sizeY = size_y


class _FakeRps:
    """Pyramid: descs[0] = full res 1000x1000, descs[1] = 500x500,
    descs[2] = 100x100 -- index 0 == full res, descending, matching real
    RawPixelsStore.getResolutionDescriptions() semantics."""

    def __init__(self, tile_size=(256, 256)):
        self._descs = [_Desc(1000, 1000), _Desc(500, 500), _Desc(100, 100)]
        self._tile_size = tile_size
        self.level_set = None

    def getResolutionLevels(self):
        return len(self._descs)

    def getResolutionDescriptions(self):
        return self._descs

    def setResolutionLevel(self, level):
        self.level_set = level

    def getTileSize(self):
        assert self.level_set is not None, (
            "getTileSize called before setResolutionLevel"
        )
        return self._tile_size


def test_closest_resolution_level_picks_smallest_level_still_covering_target():
    rps = _FakeRps()
    level, (w, h), tile_size = closest_resolution_level(rps, (400, 400))
    assert (w, h) == (500, 500)  # smallest desc still >= (400, 400)
    assert level == 1
    assert rps.level_set == level  # setResolutionLevel called before getTileSize
    assert tile_size == (256, 256)


def test_closest_resolution_level_falls_back_to_smallest_level_below_target():
    rps = _FakeRps(tile_size=(2000, 2000))
    level, (w, h), tile_size = closest_resolution_level(rps, (50, 50))
    assert (w, h) == (100, 100)  # target smaller than the smallest level
    assert level == 0
    assert tile_size == (100, 100)  # clamped to level dims, not the raw 2000x2000


def test_create_tile_list_2d_clamps_edge_tiles_without_leaking_into_next_row():
    tiles = create_tile_list_2d(0, 0, 0, size_x=10, size_y=10, tile_size=(4, 4))
    coords = [t[3] for t in tiles]
    assert (8, 8, 2, 2) in coords  # bottom-right corner: both dims clamped
    assert (
        0,
        8,
        4,
        2,
    ) in coords  # bottom row, first tile: width NOT leaked from the corner clamp
    assert (8, 0, 2, 4) in coords  # right column, first row: height NOT leaked
    for x, y, w, h in coords:
        assert x + w <= 10
        assert y + h <= 10


def test_create_full_tile_list_concatenates_channels_not_interlaced():
    tiles = create_full_tile_list(
        [0], [0, 1, 2], [0], width=4, height=4, tile_size=(4, 4)
    )
    assert [t[1] for t in tiles] == [0, 1, 2]


def test_chunkify_handles_fewer_items_than_chunks():
    chunks = _chunkify([1, 2], 4)
    assert len(chunks) == 4
    assert sum(len(c) for c in chunks) == 2


async def _fake_tile_chunk(items, fail_after=None):
    for i, item in enumerate(items):
        await asyncio.sleep(0)
        if fail_after is not None and i == fail_after:
            raise ValueError("boom")
        yield item


async def _collect(aiter):
    out = []
    async for item in aiter:
        out.append(item)
    return out


def test_merge_async_iters_terminates_with_uneven_and_empty_producers():
    """Regression test for the hang found running generate_over_network()
    against a real server: legacy's run_count-based termination signal could
    race with the consumer's queue.get(), leaving it blocked forever once
    every producer had already finished (observed as all N chunk fetches
    completing and closing their stores, then an indefinite hang). If this
    test times out, that race is back."""

    async def run():
        producers = [
            _fake_tile_chunk([1, 2, 3]),
            _fake_tile_chunk(
                []
            ),  # an empty chunk, like len(tiles) < PARALLEL_STORE_COUNT
            _fake_tile_chunk([4]),
            _fake_tile_chunk([5, 6, 7, 8, 9]),
        ]
        result = await asyncio.wait_for(
            _collect(merge_async_iters(*producers)), timeout=2.0
        )
        assert sorted(result) == list(range(1, 10))

    asyncio.run(run())


def test_merge_async_iters_propagates_producer_exceptions():
    async def run():
        producers = [
            _fake_tile_chunk([1, 2, 3, 4, 5], fail_after=2),
            _fake_tile_chunk([10, 11, 12]),
        ]
        with pytest.raises(ValueError, match="boom"):
            await asyncio.wait_for(_collect(merge_async_iters(*producers)), timeout=2.0)

    asyncio.run(run())
