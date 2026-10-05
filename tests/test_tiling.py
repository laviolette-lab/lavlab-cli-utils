# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for lavlab.tiling and the `lavlab tile` command.

The geometry, labelling and manifest halves need nothing but numpy/skimage.
The pixel-reader halves use pytest.importorskip for pyvips/tifffile the same
way test_seg.py does, and the few tests that need an OMERO object model are
skipped rather than failed when omero-py isn't installed.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import textwrap

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("skimage")

from lavlab import tiling  # noqa: E402
from lavlab.geojson.geometry import bridge_hole  # noqa: E402
from lavlab.tiling import (  # noqa: E402
    GridTile,
    MissingPixelSizeError,
    TileParams,
    TileRecord,
    TilingError,
    analyze_slide,
    assign_labels,
    assign_whole_labels,
    build_alias_lookup,
    build_grid,
    cut_tiles,
    integral_image,
    load_label_map,
    map_text_value,
    params_match,
    read_tile_params,
    resolve_total_downsample,
    sample_records,
    select_level,
    slide_is_complete,
    subject_from_image_name,
    tile_windows,
    tissue_mask,
    window_fractions,
    write_manifest,
    write_params,
)

try:
    import omero  # noqa: F401

    _HAS_OMERO = True
except Exception:  # pragma: no cover - depends on the local install
    _HAS_OMERO = False

needs_omero = pytest.mark.skipif(not _HAS_OMERO, reason="omero-py unavailable")


# ---------------------------------------------------------------------------
# subject parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("N101_S08_HE.ome.tiff", "N101"),
        ("N101_S08_HE", "N101"),
        ("N101_S08_Deeper2_HE.ome.tiff", "N101"),
        ("N101_LeftLobe_HE_Biopsy.ome.tiff", "N101"),
        ("N7_S01_HE.ome.tiff", "N7"),
        ("n101_S08_HE.ome.tiff", "N101"),
        ("something_else.tiff", None),
        ("N101.ome.tiff", None),
        ("", None),
    ],
)
def test_subject_from_image_name(name, expected):
    assert subject_from_image_name(name) == expected


def test_subject_dir_name_falls_back_and_warns(caplog):
    with caplog.at_level("WARNING"):
        assert tiling.subject_dir_name("mystery_slide.tiff") == "unknown_subject"
    assert any("unknown_subject" in r.message for r in caplog.records)


def test_sanitize_label_makes_a_usable_directory_name():
    assert tiling.sanitize_label("Seminal Vesicles") == "Seminal_Vesicles"
    assert tiling.sanitize_label("  G4/cg  ") == "G4_cg"
    assert tiling.sanitize_label("...") == "unlabeled"


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


def test_build_grid_counts_and_coordinates_without_overlap():
    tiles = build_grid(1000, 1000, size=100, overlap=0, ds_total=1.0)

    assert len(tiles) == 100
    assert tiles[0] == GridTile(0, 0, 0, 0, 0, 0, 100, 100)
    assert tiles[1].x0 == 100 and tiles[1].y0 == 0
    assert tiles[-1] == GridTile(9, 9, 900, 900, 900, 900, 100, 100)


def test_build_grid_stride_follows_overlap():
    tiles = build_grid(1000, 1000, size=100, overlap=50, ds_total=1.0)

    # stride 50 -> xt = 0, 50, ... 900 inclusive
    assert len({t.col for t in tiles}) == 19
    assert len(tiles) == 19 * 19
    assert sorted({t.x0 for t in tiles})[:3] == [0, 50, 100]


def test_build_grid_drops_edge_tiles_that_do_not_fit():
    # 250 px tall at size 100 leaves a 50 px strip that cannot hold a tile.
    tiles = build_grid(1000, 250, size=100, overlap=0, ds_total=1.0)

    assert {t.row for t in tiles} == {0, 1}
    assert max(t.y0 + t.h0 for t in tiles) == 200


def test_build_grid_maps_to_level0_at_non_integer_scale():
    tiles = build_grid(1000, 1000, size=100, overlap=0, ds_total=2.5)

    # target space is 400 px, so 4 columns of 250 level-0 px each
    assert len({t.col for t in tiles}) == 4
    assert sorted({t.x0 for t in tiles}) == [0, 250, 500, 750]
    assert {t.w0 for t in tiles} == {250}
    assert max(t.x0 + t.w0 for t in tiles) == 1000


def test_build_grid_rejects_bad_overlap_and_size():
    with pytest.raises(TilingError, match="--overlap"):
        build_grid(100, 100, size=10, overlap=10, ds_total=1.0)
    with pytest.raises(TilingError, match="--overlap"):
        build_grid(100, 100, size=10, overlap=-1, ds_total=1.0)
    with pytest.raises(TilingError, match="--size"):
        build_grid(100, 100, size=0, overlap=0, ds_total=1.0)


def test_build_grid_empty_when_slide_smaller_than_a_tile():
    assert build_grid(50, 50, size=100, overlap=0, ds_total=1.0) == []


def test_tile_windows_map_to_analysis_resolution():
    tiles = build_grid(800, 800, size=100, overlap=0, ds_total=1.0)
    xs, ys, ws, hs = tile_windows(tiles, analysis_ds=10)

    assert xs[0] == 0 and ys[0] == 0
    assert ws[0] == 10 and hs[0] == 10
    assert xs[1] == 10


# ---------------------------------------------------------------------------
# scale and level selection
# ---------------------------------------------------------------------------


def test_resolve_total_downsample_from_mpp_and_pixel_size():
    assert resolve_total_downsample(mpp=0.5, pixel_size_x=0.25) == pytest.approx(2.0)
    # X and Y are averaged when both are recorded
    assert resolve_total_downsample(
        mpp=0.5, pixel_size_x=0.2, pixel_size_y=0.3
    ) == pytest.approx(2.0)


def test_resolve_total_downsample_missing_pixel_size_points_at_downsample():
    with pytest.raises(MissingPixelSizeError, match="--downsample"):
        resolve_total_downsample(mpp=0.5, pixel_size_x=None)
    with pytest.raises(MissingPixelSizeError):
        resolve_total_downsample(mpp=0.5, pixel_size_x=0.0)


def test_resolve_total_downsample_explicit_factor_needs_no_pixel_size():
    assert resolve_total_downsample(downsample=4, pixel_size_x=None) == 4.0


def test_resolve_total_downsample_refuses_to_upsample():
    with pytest.raises(TilingError, match="finer than"):
        resolve_total_downsample(mpp=0.1, pixel_size_x=0.25)


def test_select_level_picks_smallest_level_at_or_above_target():
    plan = select_level([1000, 500, 250, 125], width0=1000, ds_total=2.5, size=100)

    assert plan.index == 1
    assert plan.downsample == pytest.approx(2.0)
    assert plan.read_size == 125
    assert plan.scale == pytest.approx(0.8)


def test_select_level_uses_full_resolution_when_no_level_is_coarse_enough():
    plan = select_level([1000, 500], width0=1000, ds_total=1.5, size=100)

    assert plan.index == 0
    assert plan.read_size == 150


def test_select_level_takes_an_exactly_matching_level():
    plan = select_level([1000, 500, 250], width0=1000, ds_total=4.0, size=100)

    assert plan.index == 2
    assert plan.read_size == 100
    assert plan.scale == pytest.approx(1.0)


def test_select_level_requires_levels():
    with pytest.raises(TilingError, match="pyramid levels"):
        select_level([], width0=1000, ds_total=2.0, size=100)


# ---------------------------------------------------------------------------
# label map
# ---------------------------------------------------------------------------


def test_bundled_label_map_covers_the_gleason_classes():
    mapping = load_label_map()

    assert {"G3", "G4cg", "G4fg", "G5"} <= set(mapping)
    lookup = build_alias_lookup(mapping)
    # the palette in lavlab/palettes.py spells these G4CG/G4FG
    assert lookup["g4cg"] == "G4cg"
    assert lookup["g4fg"] == "G4fg"
    assert lookup["gleason 3"] == "G3"


def test_alias_lookup_is_case_and_whitespace_insensitive():
    lookup = build_alias_lookup({"G3": ["Gleason 3"]})

    assert map_text_value("g3", lookup) == "G3"
    assert map_text_value("  GLEASON 3  ", lookup) == "G3"
    assert map_text_value("G3", lookup) == "G3"


def test_folder_name_is_always_its_own_alias():
    lookup = build_alias_lookup({"G5": []})
    assert map_text_value("g5", lookup) == "G5"


def test_unmapped_text_is_dropped_unless_all_is_given():
    lookup = build_alias_lookup({"G3": ["gleason 3"]})
    assert map_text_value("mystery", lookup) is None


def test_unmapped_text_with_all_uses_sanitized_value_and_warns_once(caplog):
    lookup = build_alias_lookup({"G3": ["gleason 3"]})
    warned: set[str] = set()

    with caplog.at_level("WARNING"):
        first = map_text_value("Odd Label", lookup, include_all=True, warned=warned)
        second = map_text_value("Odd Label", lookup, include_all=True, warned=warned)

    assert first == second == "Odd_Label"
    assert sum("Odd Label" in r.message for r in caplog.records) == 1


def test_load_label_map_rejects_a_missing_file():
    with pytest.raises(TilingError, match="does not exist"):
        load_label_map("/nonexistent/labels.yaml")


def test_load_label_map_accepts_a_custom_file(tmp_path):
    path = tmp_path / "labels.yaml"
    path.write_text("Tumour:\n  - carcinoma\nStroma: stromal\n")

    mapping = load_label_map(str(path))

    assert mapping == {"Tumour": ["carcinoma"], "Stroma": ["stromal"]}
    lookup = build_alias_lookup(mapping)
    assert map_text_value("CARCINOMA", lookup) == "Tumour"
    assert map_text_value("stromal", lookup) == "Stroma"


def test_selects_label_matches_folder_or_raw_text():
    assert tiling.selects_label("G3", "gleason 3", ["g3"]) is True
    assert tiling.selects_label("G3", "gleason 3", ["gleason 3"]) is True
    assert tiling.selects_label("G3", "gleason 3", ["g5"]) is False


# ---------------------------------------------------------------------------
# coverage and tissue
# ---------------------------------------------------------------------------


def _square_ring(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _shape(shape_id, ring, text):
    return (shape_id, (0, 0, 0), text, [(float(x), float(y)) for x, y in ring])


def test_window_fractions_measure_coverage():
    mask = np.zeros((16, 16), dtype=bool)
    mask[0:8, 0:8] = True
    integ = integral_image(mask)

    xs = np.array([0, 8, 4])
    ys = np.array([0, 8, 0])
    ws = np.array([8, 8, 8])
    hs = np.array([8, 8, 8])

    fracs = window_fractions(integ, xs, ys, ws, hs)

    assert fracs[0] == pytest.approx(1.0)
    assert fracs[1] == pytest.approx(0.0)
    assert fracs[2] == pytest.approx(0.5)


def test_rasterize_shapes_fills_polygons_and_rectangles():
    shapes = [
        _shape(1, _square_ring(0, 0, 8, 8), "g3"),
        _shape(2, _square_ring(10, 10, 18, 18), "g5"),
    ]

    masks, shape_ids = tiling.rasterize_shapes(shapes, (20, 20))

    assert masks["g3"][4, 4] and not masks["g3"][14, 14]
    assert masks["g5"][14, 14]
    assert shape_ids[4, 4] == 1
    assert shape_ids[14, 14] == 2


def test_rasterize_shapes_preserves_bridged_holes():
    # A hole folded into its outer ring as a keyhole slit, exactly as
    # lavlab.geojson writes them into OMERO -- even-odd filling must leave
    # the interior unset without any special handling here.
    ring = bridge_hole(_square_ring(0, 0, 20, 20), _square_ring(6, 6, 14, 14))
    shapes = [_shape(1, ring, "g3")]

    masks, _ = tiling.rasterize_shapes(shapes, (24, 24))

    assert masks["g3"][2, 2]
    assert not masks["g3"][10, 10]


def test_tissue_mask_separates_stained_tissue_from_glass():
    thumb = np.full((40, 40, 3), 250, dtype=np.uint8)
    thumb[10:30, 10:30] = (180, 60, 160)

    mask = tissue_mask(thumb, min_object_px=4)

    assert mask[20, 20]
    assert not mask[2, 2]
    assert mask.sum() == pytest.approx(400, rel=0.15)


def test_tissue_mask_on_a_blank_thumbnail_finds_nothing():
    assert not tissue_mask(np.full((20, 20, 3), 255, dtype=np.uint8)).any()


def test_tissue_mask_handles_greyscale():
    thumb = np.full((40, 40), 250, dtype=np.uint8)
    thumb[10:30, 10:30] = 40

    mask = tissue_mask(thumb, min_object_px=4)

    assert mask[20, 20] and not mask[2, 2]


# ---------------------------------------------------------------------------
# labelling
# ---------------------------------------------------------------------------


def _grid(n=4, size=8):
    return build_grid(n * size, n * size, size=size, overlap=0, ds_total=1.0)


def test_assign_labels_applies_the_min_coverage_threshold():
    tiles = _grid(2)
    tissue = np.ones(len(tiles))
    fracs = {"G3": np.array([0.9, 0.4, 0.5, 0.0])}

    records, stats = assign_labels(
        tiles, tissue, fracs, min_coverage=0.5, tissue_thresh=0.5
    )

    assert [r.tile.col for r in records] == [0, 0]
    assert [r.label for r in records] == ["G3", "G3"]
    assert stats["dropped_unlabeled"] == 2


def test_assign_labels_picks_the_highest_coverage_label():
    tiles = _grid(1)
    records, _ = assign_labels(
        tiles,
        np.ones(len(tiles)),
        {"G3": np.array([0.6]), "G5": np.array([0.9]), "G4cg": np.array([0.7])},
        min_coverage=0.5,
    )

    assert [(r.label, r.coverage) for r in records] == [("G5", 0.9)]


def test_assign_labels_breaks_exact_ties_by_name_for_reproducibility():
    tiles = _grid(1)
    records, _ = assign_labels(
        tiles,
        np.ones(1),
        {"G5": np.array([0.8]), "G3": np.array([0.8])},
        min_coverage=0.5,
    )

    assert records[0].label == "G3"


def test_assign_labels_drops_tiles_without_enough_tissue():
    tiles = _grid(2)
    records, stats = assign_labels(
        tiles,
        np.array([0.9, 0.1, 0.6, 0.0]),
        {"G3": np.ones(4)},
        min_coverage=0.5,
        tissue_thresh=0.5,
    )

    assert len(records) == 2
    assert stats["dropped_no_tissue"] == 2


def test_assign_labels_drops_any_tile_touching_an_exclusion():
    tiles = _grid(2)
    records, stats = assign_labels(
        tiles,
        np.ones(4),
        {"G3": np.ones(4)},
        exclude_fracs=np.array([0.0, 0.01, 0.5, 0.0]),
        min_coverage=0.5,
    )

    assert stats["dropped_excluded"] == 2
    assert len(records) == 2


def test_assign_labels_records_the_covering_roi_id():
    tiles = _grid(1)
    records, _ = assign_labels(
        tiles,
        np.ones(1),
        {"G3": np.array([1.0])},
        shape_id_for=lambda index: 4242,
        min_coverage=0.5,
    )

    assert records[0].shape_id == 4242


# --- the benign / background rule ------------------------------------------


def test_background_labels_tissue_outside_every_roi():
    tiles = _grid(2)
    records, _ = assign_labels(
        tiles,
        np.ones(4),
        {"G3": np.array([1.0, 0.0, 0.0, 0.0])},
        background_fracs=np.array([1.0, 0.0, 0.0, 0.0]),
        background_label="benign",
        has_rois=True,
        min_coverage=0.5,
    )

    assert [r.label for r in records] == ["G3", "benign", "benign", "benign"]


def test_background_skipped_on_a_slide_with_no_rois():
    """An unannotated slide is not a benign slide."""
    tiles = _grid(2)
    records, stats = assign_labels(
        tiles,
        np.ones(4),
        {},
        background_fracs=np.zeros(4),
        background_label="benign",
        has_rois=False,
        min_coverage=0.5,
    )

    assert records == []
    assert stats["dropped_unlabeled"] == 4


def test_background_excludes_tiles_near_an_roi():
    tiles = _grid(2)
    # tile 1 sits inside the dilated margin, so it is neither G3 nor benign
    records, stats = assign_labels(
        tiles,
        np.ones(4),
        {"G3": np.array([1.0, 0.1, 0.0, 0.0])},
        background_fracs=np.array([1.0, 0.4, 0.0, 0.0]),
        background_label="benign",
        has_rois=True,
        min_coverage=0.5,
    )

    labels = {(r.tile.row, r.tile.col): r.label for r in records}
    assert labels[(0, 0)] == "G3"
    assert (0, 1) not in labels
    assert labels[(1, 0)] == "benign"
    assert stats["dropped_unlabeled"] == 1


def test_background_can_be_turned_off():
    tiles = _grid(1)
    records, _ = assign_labels(
        tiles,
        np.ones(1),
        {},
        background_fracs=np.zeros(1),
        background_label=None,
        has_rois=True,
    )

    assert records == []


def test_assign_whole_labels_keeps_every_tissue_tile_under_one_label():
    tiles = _grid(2)
    records, stats = assign_whole_labels(
        tiles,
        np.array([1.0, 0.2, 0.9, 1.0]),
        exclude_fracs=np.array([0.0, 0.0, 0.3, 0.0]),
        tissue_thresh=0.5,
    )

    assert [r.label for r in records] == ["whole", "whole"]
    assert stats["dropped_no_tissue"] == 1
    assert stats["dropped_excluded"] == 1


# ---------------------------------------------------------------------------
# integrated analysis (thumbnail + shapes -> labels)
# ---------------------------------------------------------------------------


def _analysis_thumbnail():
    """White glass with a stained tissue block from (8, 8) to (72, 72)."""
    thumb = np.full((80, 80, 3), 250, dtype=np.uint8)
    thumb[8:72, 8:72] = (180, 60, 160)
    return thumb


def _analysis_params(**kw):
    base = dict(
        mode="roi",
        include_all=True,
        mpp=1.0,
        size=8,
        overlap=0,
        min_coverage=0.5,
        tissue_thresh=0.5,
        background_label="benign",
        background_margin_um=4.0,
        max_background_tiles=None,
        exclude_text=["exclusion roi"],
    )
    base.update(kw)
    return TileParams(**base)


def _labels_by_cell(records):
    return {(r.tile.row, r.tile.col): r.label for r in records}


def test_analyze_slide_labels_roi_margin_and_background():
    tiles = build_grid(80, 80, size=8, overlap=0, ds_total=1.0)
    shapes = [_shape(7, _square_ring(16, 16, 32, 32), "g3")]

    records, _stats, has_rois = analyze_slide(
        tiles,
        _analysis_thumbnail(),
        shapes,
        analysis_ds=1,
        params=_analysis_params(),
        lookup=build_alias_lookup({"G3": ["g3"]}),
        pixel_size_x=1.0,
    )
    labels = _labels_by_cell(records)

    assert has_rois is True
    # squarely inside the ROI
    assert labels[(2, 2)] == "G3"
    assert labels[(3, 3)] == "G3"
    # just outside it, inside the 4 um margin -> neither class nor benign
    assert (1, 1) not in labels
    assert (4, 4) not in labels
    # tissue far from any ROI -> benign
    assert labels[(7, 7)] == "benign"
    # glass -> no tile at all
    assert (0, 0) not in labels
    # a classed tile carries the ROI that covered it; background carries none
    graded = [r for r in records if r.label == "G3"]
    assert {r.shape_id for r in graded} == {7}
    assert all(r.shape_id is None for r in records if r.label == "benign")


def test_analyze_slide_reports_no_rois_so_the_caller_can_skip():
    tiles = build_grid(80, 80, size=8, overlap=0, ds_total=1.0)

    records, _stats, has_rois = analyze_slide(
        tiles,
        _analysis_thumbnail(),
        [],
        analysis_ds=1,
        params=_analysis_params(),
        lookup={},
        pixel_size_x=1.0,
    )

    assert has_rois is False
    assert records == []


def test_analyze_slide_drops_tiles_inside_an_exclusion_roi():
    tiles = build_grid(80, 80, size=8, overlap=0, ds_total=1.0)
    shapes = [
        _shape(1, _square_ring(16, 16, 32, 32), "g3"),
        # kept clear of x=24 so it lands wholly inside grid cell (2, 2):
        # a tile is dropped on *any* exclusion overlap, edge pixels included
        _shape(2, _square_ring(16, 16, 23, 23), "exclusion roi"),
    ]

    records, stats, _ = analyze_slide(
        tiles,
        _analysis_thumbnail(),
        shapes,
        analysis_ds=1,
        params=_analysis_params(),
        lookup=build_alias_lookup({"G3": ["g3"]}),
        pixel_size_x=1.0,
    )
    labels = _labels_by_cell(records)

    assert (2, 2) not in labels  # covered by the exclusion
    assert labels[(3, 3)] == "G3"  # the rest of the ROI survives
    assert stats["dropped_excluded"] >= 1


def test_analyze_slide_exclusions_apply_in_whole_mode_too():
    tiles = build_grid(80, 80, size=8, overlap=0, ds_total=1.0)
    shapes = [_shape(1, _square_ring(16, 16, 32, 32), "exclusion roi")]

    records, stats, _ = analyze_slide(
        tiles,
        _analysis_thumbnail(),
        shapes,
        analysis_ds=1,
        params=_analysis_params(mode="whole", background_label=None),
        lookup={},
        pixel_size_x=1.0,
    )
    labels = _labels_by_cell(records)

    assert (2, 2) not in labels
    assert labels[(7, 7)] == "whole"
    assert stats["dropped_excluded"] >= 1


def test_analyze_slide_text_filter_selects_a_subset():
    tiles = build_grid(80, 80, size=8, overlap=0, ds_total=1.0)
    shapes = [
        _shape(1, _square_ring(16, 16, 32, 32), "g3"),
        _shape(2, _square_ring(40, 40, 56, 56), "g5"),
    ]
    params = _analysis_params(
        include_all=False, text_filter=["g3"], background_label=None
    )

    records, _stats, _ = analyze_slide(
        tiles,
        _analysis_thumbnail(),
        shapes,
        analysis_ds=1,
        params=params,
        lookup=build_alias_lookup({"G3": ["g3"], "G5": ["g5"]}),
        pixel_size_x=1.0,
    )

    assert {r.label for r in records} == {"G3"}


def test_analyze_slide_background_still_respects_unselected_rois():
    """A ROI the -t filter ignored must still keep its tiles out of benign."""
    tiles = build_grid(80, 80, size=8, overlap=0, ds_total=1.0)
    shapes = [_shape(2, _square_ring(40, 40, 64, 64), "g5")]
    params = _analysis_params(include_all=False, text_filter=["g3"])

    records, _stats, _ = analyze_slide(
        tiles,
        _analysis_thumbnail(),
        shapes,
        analysis_ds=1,
        params=params,
        lookup=build_alias_lookup({"G3": ["g3"], "G5": ["g5"]}),
        pixel_size_x=1.0,
    )
    labels = _labels_by_cell(records)

    assert (6, 6) not in labels  # inside the unselected G5 ROI
    assert labels[(1, 1)] == "benign"  # far from it


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def _records(label, count):
    return [
        TileRecord(GridTile(i, 0, i, 0, i * 10, 0, 10, 10), label, 1.0, 1.0)
        for i in range(count)
    ]


def test_sample_records_caps_per_label():
    records = _records("G3", 10) + _records("benign", 50)

    kept = sample_records(
        records, max_background_tiles=5, background_label="benign", seed=0
    )

    counts = {}
    for record in kept:
        counts[record.label] = counts.get(record.label, 0) + 1
    assert counts == {"G3": 10, "benign": 5}


def test_sample_records_is_deterministic_for_a_seed():
    records = _records("benign", 50)

    first = sample_records(
        records, max_background_tiles=7, background_label="benign", seed=3
    )
    second = sample_records(
        records, max_background_tiles=7, background_label="benign", seed=3
    )
    other = sample_records(
        records, max_background_tiles=7, background_label="benign", seed=4
    )

    assert [r.tile.x0 for r in first] == [r.tile.x0 for r in second]
    assert [r.tile.x0 for r in first] != [r.tile.x0 for r in other]


def test_sample_records_applies_the_overall_cap_last():
    records = _records("G3", 20) + _records("G5", 20)

    kept = sample_records(records, max_tiles=6, max_tiles_per_label=5, seed=1)

    assert len(kept) == 6


def test_sample_records_returns_row_major_order():
    records = _records("G3", 20)

    kept = sample_records(records, max_tiles=5, seed=2)

    assert kept == sorted(kept, key=lambda r: (r.tile.row, r.tile.col))


# ---------------------------------------------------------------------------
# manifest and tile_params
# ---------------------------------------------------------------------------


def _row(**kw):
    base = dict.fromkeys(tiling.MANIFEST_COLUMNS, "")
    base.update(kw)
    return base


def test_write_manifest_writes_every_column(tmp_path):
    path = write_manifest(
        str(tmp_path), [_row(tile_path="a.png", omero_image_id=1, label="G3", x=10)]
    )

    contents = open(path).read().splitlines()
    assert contents[0] == ",".join(tiling.MANIFEST_COLUMNS)
    assert contents[1].startswith("a.png,1,,,,G3,10,")


def test_write_manifest_is_atomic_and_leaves_nothing_behind_on_failure(tmp_path):
    with pytest.raises(ValueError):
        write_manifest(str(tmp_path), [{"not_a_column": 1}])

    assert not (tmp_path / tiling.MANIFEST_NAME).exists()
    assert os.listdir(tmp_path) == []


def test_slide_is_complete_needs_both_files(tmp_path):
    assert slide_is_complete(str(tmp_path)) is False

    write_manifest(str(tmp_path), [])
    assert slide_is_complete(str(tmp_path)) is False

    write_params(str(tmp_path), TileParams())
    assert slide_is_complete(str(tmp_path)) is True


def test_skip_existing_accepts_a_matching_parameter_record(tmp_path):
    params = TileParams(mode="roi", include_all=True, size=224, mpp=0.5)
    write_params(str(tmp_path), params)

    saved = read_tile_params(str(tmp_path))

    assert params_match(saved, params) is True
    assert saved["lavlab_version"]
    assert saved["generated_at"]


def test_skip_existing_rejects_a_mismatched_parameter_record(tmp_path):
    write_params(str(tmp_path), TileParams(size=224, mpp=0.5))

    saved = read_tile_params(str(tmp_path))

    assert params_match(saved, TileParams(size=512, mpp=0.5)) is False
    assert params_match(saved, TileParams(size=224, mpp=1.0)) is False
    assert params_match(saved, TileParams(size=224, mpp=0.5, seed=9)) is False


def test_params_match_ignores_version_and_timestamp(tmp_path):
    params = TileParams()
    write_params(str(tmp_path), params)
    saved = read_tile_params(str(tmp_path))
    saved["lavlab_version"] = "9.9.9"
    saved["generated_at"] = "1999-01-01T00:00:00+00:00"

    assert params_match(saved, params) is True


def test_params_match_is_insensitive_to_filter_order_and_case():
    a = TileParams(text_filter=["G3", "g5"], exclude_text=["Exclusion ROI"])
    b = TileParams(text_filter=["g5", "g3"], exclude_text=["exclusion roi"])

    assert a.signature() == b.signature()


def test_read_tile_params_tolerates_a_corrupt_file(tmp_path, caplog):
    (tmp_path / tiling.PARAMS_NAME).write_text("{not json")

    with caplog.at_level("WARNING"):
        assert read_tile_params(str(tmp_path)) is None


def test_tile_params_json_records_the_background_settings(tmp_path):
    params = TileParams(
        background_label="benign",
        background_margin_um=200.0,
        max_background_tiles=2000,
        seed=7,
    )
    write_params(str(tmp_path), params, tier="local")

    saved = json.loads((tmp_path / tiling.PARAMS_NAME).read_text())

    assert saved["params"]["background_label"] == "benign"
    assert saved["params"]["background_margin_um"] == 200.0
    assert saved["params"]["max_background_tiles"] == 2000
    assert saved["params"]["seed"] == 7
    assert saved["tier"] == "local"


def test_tile_filename_uses_level0_coordinates():
    record = TileRecord(GridTile(1, 2, 0, 0, 12800, 7424, 448, 448), "G3", 1.0, 1.0)

    assert tiling.tile_filename("N101_S08_HE", record, "png") == (
        "N101_S08_HE_x12800_y7424.png"
    )


# ---------------------------------------------------------------------------
# pixel readers and tier parity
# ---------------------------------------------------------------------------

pyvips = pytest.importorskip("pyvips")
tifffile = pytest.importorskip("tifffile")


def _write_pyramid(path, base):
    """A classic multi-page pyramid, the layout SVS uses."""
    with tifffile.TiffWriter(str(path), bigtiff=True) as writer:
        writer.write(base, tile=(64, 64))
        writer.write(base[::2, ::2], tile=(64, 64), subfiletype=1)


def _synthetic_slide(size=256):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


class _BlockReader:
    """Stands in for tier 3: same interface, assembled from 64 px blocks.

    Mirrors NetworkRegionReader's coordinate handling but serves from a
    plain array, so a test can compare the two tiers without a server.
    """

    def __init__(self, level_array, plan):
        self._array = level_array
        self._plan = plan
        self.prefetched = 0

    def prefetch(self, regions):
        self.prefetched += 1

    def read(self, x0, y0, w0, h0):
        size = self._plan.read_size
        level_h, level_w = self._array.shape[:2]
        x = min(max(int(round(x0 / self._plan.downsample)), 0), max(level_w - size, 0))
        y = min(max(int(round(y0 / self._plan.downsample)), 0), max(level_h - size, 0))
        width = min(size, level_w - x)
        height = min(size, level_h - y)

        out = np.zeros((height, width, self._array.shape[2]), dtype=np.uint8)
        for by in range(y // 64 * 64, y + height, 64):
            for bx in range(x // 64 * 64, x + width, 64):
                block = self._array[by : by + 64, bx : bx + 64]
                sx0, sy0 = max(x, bx), max(y, by)
                sx1 = min(x + width, bx + block.shape[1])
                sy1 = min(y + height, by + block.shape[0])
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                out[sy0 - y : sy1 - y, sx0 - x : sx1 - x] = block[
                    sy0 - by : sy1 - by, sx0 - bx : sx1 - bx
                ]
        return out

    def close(self):
        pass


def test_local_level_dims_reads_the_pyramid(tmp_path):
    path = tmp_path / "slide.tiff"
    _write_pyramid(path, _synthetic_slide())

    assert tiling.local_level_dims(str(path)) == [(256, 256), (128, 128)]


def test_local_region_reader_crops_exactly(tmp_path):
    base = _synthetic_slide()
    path = tmp_path / "slide.tiff"
    _write_pyramid(path, base)

    plan = select_level([256, 128], width0=256, ds_total=1.0, size=32)
    reader = tiling.LocalRegionReader(str(path), plan, (256, 256))
    try:
        region = reader.read(64, 32, 32, 32)
    finally:
        reader.close()

    assert np.array_equal(region, base[32:64, 64:96])


def test_open_local_level_rejects_a_level_that_is_not_there(tmp_path):
    path = tmp_path / "slide.tiff"
    _write_pyramid(path, _synthetic_slide())

    with pytest.raises(tiling.UnreadableSourceError):
        tiling.open_local_level(str(path), 5, (8, 8))


def test_both_tiers_produce_identical_tiles(tmp_path):
    """The tier only decides where bytes come from, never which pixels."""
    base = _synthetic_slide()
    path = tmp_path / "slide.tiff"
    _write_pyramid(path, base)

    # ds_total 2.0 lands on level 1, so both tiers read the half-res level
    # and resize identically through resize_tile.
    plan = select_level([256, 128], width0=256, ds_total=2.0, size=32)
    assert plan.index == 1

    tiles = build_grid(256, 256, size=32, overlap=0, ds_total=2.0)
    records = [TileRecord(tile, "whole", 0.0, 1.0) for tile in tiles]

    local = tiling.LocalRegionReader(str(path), plan, (128, 128))
    try:
        local_tiles = [pixels for _r, pixels in cut_tiles(local, records, 32)]
    finally:
        local.close()

    network = _BlockReader(base[::2, ::2].copy(), plan)
    network_tiles = [pixels for _r, pixels in cut_tiles(network, records, 32)]

    assert len(local_tiles) == len(records) == 16
    for left, right in zip(local_tiles, network_tiles):
        assert np.array_equal(left, right)


def test_cut_tiles_prefetches_per_band_not_per_tile(tmp_path):
    tiles = build_grid(256, 256, size=32, overlap=0, ds_total=2.0)
    records = [TileRecord(tile, "whole", 0.0, 1.0) for tile in tiles]
    plan = select_level([256, 128], width0=256, ds_total=2.0, size=32)
    reader = _BlockReader(_synthetic_slide()[::2, ::2].copy(), plan)

    list(cut_tiles(reader, records, 32, band_rows=2))

    # 4 grid rows in bands of 2 -> 2 prefetches, not 16
    assert reader.prefetched == 2
    assert reader.prefetched < len(records)


def test_resize_tile_is_a_no_op_at_the_right_size():
    array = _synthetic_slide(32)
    assert tiling.resize_tile(array, 32) is array


def test_resize_tile_downsamples_to_the_output_size():
    assert tiling.resize_tile(_synthetic_slide(64), 32).shape == (32, 32, 3)


def test_write_tile_round_trips(tmp_path):
    array = _synthetic_slide(32)
    path = tmp_path / "tile.png"

    tiling.write_tile(array, str(path))

    loaded = pyvips.Image.new_from_file(str(path))
    assert (loaded.width, loaded.height) == (32, 32)


def test_is_slow_jp2_source():
    assert tiling.is_slow_jp2_source("/OMERO/x.jp2") is True
    assert tiling.is_slow_jp2_source("/OMERO/X.JP2") is True
    assert tiling.is_slow_jp2_source("/OMERO/x.ome.tiff") is False


# ---------------------------------------------------------------------------
# tier selection
# ---------------------------------------------------------------------------


@needs_omero
def test_choose_tier_local_when_the_source_is_mounted(tmp_path, monkeypatch):
    import lavlab.omero_client as omero_client

    source = tmp_path / "slide.ome.tiff"
    source.write_bytes(b"x")
    monkeypatch.setattr(omero_client, "get_source_file_path", lambda c, i: str(source))

    assert tiling.choose_tier(object(), 1) == ("local", str(source))


@needs_omero
def test_choose_tier_network_when_there_is_no_source(monkeypatch):
    import lavlab.omero_client as omero_client

    monkeypatch.setattr(omero_client, "get_source_file_path", lambda c, i: None)

    assert tiling.choose_tier(object(), 1) == ("network", None)


@needs_omero
def test_choose_tier_network_when_the_source_is_not_mounted_here(monkeypatch):
    import lavlab.omero_client as omero_client

    monkeypatch.setattr(
        omero_client, "get_source_file_path", lambda c, i: "/OMERO/nope.ome.tiff"
    )

    tier, path = tiling.choose_tier(object(), 1)
    assert tier == "network"
    assert path == "/OMERO/nope.ome.tiff"


@needs_omero
def test_choose_tier_falls_forward_for_jp2_sources(tmp_path, monkeypatch, caplog):
    import lavlab.omero_client as omero_client

    source = tmp_path / "slide.jp2"
    source.write_bytes(b"x")
    monkeypatch.setattr(omero_client, "get_source_file_path", lambda c, i: str(source))

    with caplog.at_level("WARNING"):
        tier, _path = tiling.choose_tier(object(), 1)

    assert tier == "network"
    assert any("JPEG-2000" in r.message for r in caplog.records)


@needs_omero
def test_force_local_overrides_the_jp2_fall_forward(tmp_path, monkeypatch):
    import lavlab.omero_client as omero_client

    source = tmp_path / "slide.jp2"
    source.write_bytes(b"x")
    monkeypatch.setattr(omero_client, "get_source_file_path", lambda c, i: str(source))

    assert tiling.choose_tier(object(), 1, force_local=True)[0] == "local"


# ---------------------------------------------------------------------------
# tile_slide end to end (fully mocked OMERO)
# ---------------------------------------------------------------------------


class _FakeImage:
    def __init__(
        self, image_id=1, name="N101_S08_HE.ome.tiff", size=80, pixel_size=1.0
    ):
        self._id = image_id
        self._name = name
        self._size = size
        self._pixel_size = pixel_size

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


class _FakeTileReader:
    def __init__(self, *args, **kwargs):
        pass

    def prefetch(self, regions):
        pass

    def read(self, x0, y0, w0, h0):
        return np.full((w0, h0, 3), 128, dtype=np.uint8)

    def close(self):
        pass


@pytest.fixture
def mocked_slide(monkeypatch, tmp_path):
    """Wire tile_slide onto a synthetic 80x80 slide with no server."""
    import lavlab.roi as roi_module

    source = tmp_path / "src.ome.tiff"
    source.write_bytes(b"x")

    monkeypatch.setattr(
        tiling, "choose_tier", lambda c, i, force_local=False: ("local", str(source))
    )
    monkeypatch.setattr(tiling, "local_level_dims", lambda p: [(80, 80)])
    monkeypatch.setattr(
        tiling, "_load_thumbnail_local", lambda p, d: _analysis_thumbnail()
    )
    monkeypatch.setattr(tiling, "LocalRegionReader", _FakeTileReader)

    def _shapes(image, **kwargs):
        return _shapes.value

    _shapes.value = [_shape(7, _square_ring(16, 16, 32, 32), "g3")]
    monkeypatch.setattr(roi_module, "get_shapes_as_points", _shapes)
    # shape 7 belongs to OMERO ROI 70
    monkeypatch.setattr(
        roi_module,
        "shape_roi_ids",
        lambda image: {sid: sid * 10 for sid, *_rest in _shapes.value},
    )
    return _shapes


@needs_omero
def test_tile_slide_writes_tiles_manifest_and_params(mocked_slide, tmp_path):
    out = tmp_path / "out"
    params = _analysis_params(max_background_tiles=3)

    result = tiling.tile_slide(None, _FakeImage(), params, str(out))

    assert result.status == "done"
    assert result.tier == "local"
    slide_dir = out / "N101" / "N101_S08_HE"
    assert (slide_dir / tiling.MANIFEST_NAME).is_file()
    assert (slide_dir / tiling.PARAMS_NAME).is_file()
    assert (slide_dir / "G3").is_dir()
    assert result.label_counts["G3"] > 0
    assert result.label_counts["benign"] == 3

    manifest = (slide_dir / tiling.MANIFEST_NAME).read_text().splitlines()
    assert manifest[0] == ",".join(tiling.MANIFEST_COLUMNS)
    rows = list(csv.DictReader(io.StringIO("\n".join(manifest))))
    first = rows[0]
    assert first["case_id"] == "N101"
    assert first["slide_id"] == "N101_S08_HE"
    assert first["omero_image_id"] == "1"
    assert first["tier"] == "local"
    # path is relative to the output root, not absolute
    assert not os.path.isabs(first["tile_path"])
    assert (out / first["tile_path"]).is_file()
    # a graded tile carries the real OMERO ROI id and the shape id
    graded = next(r for r in rows if r["label"] == "G3")
    assert graded["roi_id"] == "70"
    assert graded["shape_id"] == "7"
    # x/y are the tile centre
    assert int(graded["x"]) == int(graded["x0"]) + int(graded["w0"]) // 2


@needs_omero
def test_tile_slide_skips_an_unannotated_slide_in_roi_mode(
    mocked_slide, tmp_path, caplog
):
    mocked_slide.value = []
    out = tmp_path / "out"

    with caplog.at_level("WARNING"):
        result = tiling.tile_slide(None, _FakeImage(), _analysis_params(), str(out))

    assert result.status == "unannotated"
    assert not (out / "N101").exists()
    assert any("not a benign slide" in r.message for r in caplog.records)


@needs_omero
def test_tile_slide_whole_mode_tiles_an_unannotated_slide(mocked_slide, tmp_path):
    mocked_slide.value = []
    out = tmp_path / "out"

    result = tiling.tile_slide(
        None,
        _FakeImage(),
        _analysis_params(mode="whole", background_label=None),
        str(out),
    )

    assert result.status == "done"
    assert set(result.label_counts) == {"whole"}
    assert (out / "N101" / "N101_S08_HE" / "whole").is_dir()


@needs_omero
def test_tile_slide_coords_only_writes_no_images(mocked_slide, tmp_path):
    out = tmp_path / "out"

    result = tiling.tile_slide(
        None, _FakeImage(), _analysis_params(coords_only=True), str(out)
    )

    slide_dir = out / "N101" / "N101_S08_HE"
    assert result.written == 0
    assert (slide_dir / tiling.MANIFEST_NAME).is_file()
    assert not (slide_dir / "G3").exists()
    rows = (slide_dir / tiling.MANIFEST_NAME).read_text().splitlines()[1:]
    assert rows
    assert all(
        dict(zip(tiling.MANIFEST_COLUMNS, r.split(",")))["tile_path"] == ""
        for r in rows
    )


@needs_omero
def test_tile_slide_skip_existing_honours_matching_params(mocked_slide, tmp_path):
    out = tmp_path / "out"
    params = _analysis_params()
    tiling.tile_slide(None, _FakeImage(), params, str(out))

    result = tiling.tile_slide(None, _FakeImage(), params, str(out), skip_existing=True)

    assert result.status == "skipped"
    assert "already tiled" in result.reason


@needs_omero
def test_tile_slide_skip_existing_refuses_mismatched_params(
    mocked_slide, tmp_path, caplog
):
    out = tmp_path / "out"
    tiling.tile_slide(None, _FakeImage(), _analysis_params(), str(out))

    with caplog.at_level("WARNING"):
        result = tiling.tile_slide(
            None, _FakeImage(), _analysis_params(size=16), str(out), skip_existing=True
        )

    assert result.status == "skipped"
    assert result.reason == "parameter mismatch"
    assert any("--override" in r.message for r in caplog.records)


@needs_omero
def test_tile_slide_override_retiles_mismatched_params(mocked_slide, tmp_path):
    out = tmp_path / "out"
    tiling.tile_slide(None, _FakeImage(), _analysis_params(), str(out))

    result = tiling.tile_slide(
        None,
        _FakeImage(),
        _analysis_params(size=16),
        str(out),
        skip_existing=True,
        override=True,
    )

    assert result.status == "done"
    saved = read_tile_params(str(out / "N101" / "N101_S08_HE"))
    assert saved["params"]["size"] == 16


@needs_omero
def test_tile_slide_missing_pixel_size_raises_the_actionable_error(
    mocked_slide, tmp_path
):
    with pytest.raises(MissingPixelSizeError, match="--downsample"):
        tiling.tile_slide(
            None, _FakeImage(pixel_size=None), _analysis_params(), str(tmp_path / "out")
        )


@needs_omero
def test_tile_slide_unparseable_name_uses_unknown_subject(mocked_slide, tmp_path):
    out = tmp_path / "out"

    tiling.tile_slide(
        None, _FakeImage(name="mystery.ome.tiff"), _analysis_params(), str(out)
    )

    assert (out / "unknown_subject" / "mystery").is_dir()


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


from lavlab.cli import build_parser  # noqa: E402
from lavlab.commands.tile import (  # noqa: E402
    _build_params,
    _summarise_batch,
    _validate_args,
)


def _args(argv):
    return build_parser().parse_args(argv)


def _tile_args(*extra):
    return _args(["tile", "1", "-o", "/tmp/out", *extra])


def test_tile_is_registered_as_a_subcommand():
    args = _tile_args("--whole")
    assert args.command == "tile"
    assert callable(args.handler)


def test_whole_rejects_annotation_filters():
    with pytest.raises(SystemExit, match="--whole ignores"):
        _validate_args(_tile_args("--whole", "-t", "g3"))
    with pytest.raises(SystemExit, match="--whole ignores"):
        _validate_args(_tile_args("--whole", "--all"))


def test_roi_requires_all_or_a_text_filter():
    with pytest.raises(SystemExit, match="--all or at least one --text-filter"):
        _validate_args(_tile_args("--roi"))


def test_roi_rejects_all_together_with_a_text_filter():
    with pytest.raises(SystemExit, match="contradictory"):
        _validate_args(_tile_args("--roi", "--all", "-t", "g3"))


def test_mode_is_required_and_exclusive():
    with pytest.raises(SystemExit):
        _args(["tile", "1", "-o", "/tmp/out"])
    with pytest.raises(SystemExit):
        _args(["tile", "1", "-o", "/tmp/out", "--whole", "--roi"])


def test_out_is_required():
    with pytest.raises(SystemExit):
        _args(["tile", "1", "--whole"])


def test_mpp_and_downsample_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _tile_args("--whole", "--mpp", "0.5", "--downsample", "4")


def test_coords_only_rejects_an_explicit_format():
    with pytest.raises(SystemExit, match="--coords-only writes no image files"):
        _validate_args(_tile_args("--whole", "--coords-only", "--format", "png"))


def test_coords_only_alone_is_fine():
    _validate_args(_tile_args("--whole", "--coords-only"))


def test_overlap_must_be_less_than_size():
    with pytest.raises(SystemExit, match="--overlap"):
        _validate_args(_tile_args("--whole", "--size", "64", "--overlap", "64"))
    with pytest.raises(SystemExit, match="--overlap"):
        _validate_args(_tile_args("--whole", "--overlap", "-1"))


def test_fractions_must_be_between_zero_and_one():
    with pytest.raises(SystemExit, match="--min-coverage"):
        _validate_args(_tile_args("--roi", "--all", "--min-coverage", "1.5"))
    with pytest.raises(SystemExit, match="--tissue-thresh"):
        _validate_args(_tile_args("--whole", "--tissue-thresh", "-0.1"))


def test_background_flags_are_rejected_in_whole_mode():
    with pytest.raises(SystemExit, match="only applies to --roi"):
        _validate_args(_tile_args("--whole", "--background-label", "benign"))
    with pytest.raises(SystemExit, match="only applies to --roi"):
        _validate_args(_tile_args("--whole", "--background-margin", "100"))
    with pytest.raises(SystemExit, match="only applies to --roi"):
        _validate_args(_tile_args("--whole", "--no-background"))


def test_no_background_conflicts_with_naming_one():
    with pytest.raises(SystemExit, match="contradictory"):
        _validate_args(
            _tile_args(
                "--roi", "--all", "--no-background", "--background-label", "benign"
            )
        )


def test_negative_caps_are_rejected():
    with pytest.raises(SystemExit, match="--max-tiles"):
        _validate_args(_tile_args("--whole", "--max-tiles", "-1"))


def test_build_params_defaults_match_the_documented_ones():
    params = _build_params(_tile_args("--roi", "--all"))

    assert params.mode == "roi"
    assert params.mpp == 0.5
    assert params.downsample is None
    assert params.size == 224
    assert params.overlap == 0
    assert params.min_coverage == 0.5
    assert params.tissue_thresh == 0.5
    assert params.fmt == "png"
    assert params.background_label == "benign"
    assert params.background_margin_um == 200.0
    assert params.max_background_tiles == 2000
    assert params.exclude_text == ["exclusion roi"]
    assert params.seed == 0


def test_build_params_downsample_clears_mpp():
    params = _build_params(_tile_args("--whole", "--downsample", "4"))

    assert params.mpp is None
    assert params.downsample == 4.0


def test_build_params_whole_mode_has_no_background_class():
    params = _build_params(_tile_args("--whole"))

    assert params.mode == "whole"
    assert params.background_label is None


def test_build_params_no_background_turns_it_off():
    params = _build_params(_tile_args("--roi", "--all", "--no-background"))

    assert params.background_label is None


def test_build_params_lowercases_text_filters():
    params = _build_params(_tile_args("--roi", "-t", "G3", "-t", "G4CG"))

    assert params.text_filter == ["g3", "g4cg"]


def test_build_params_custom_exclusions_replace_the_default():
    params = _build_params(_tile_args("--roi", "--all", "--exclude-text", "Burnt"))

    assert params.exclude_text == ["burnt"]


# --- batch summary ---------------------------------------------------------


def _result(image_id, status, tier=None, **counts):
    from collections import Counter

    return tiling.SlideResult(image_id, status, tier=tier, label_counts=Counter(counts))


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_batch_summary_counts_every_status(caplog):
    ids = [1, 2, 3, 4]
    results = [
        _result(1, "done", "local", G3=5),
        _result(2, "skipped"),
        _result(3, "unannotated"),
        _result(4, "failed"),
    ]

    with caplog.at_level("INFO"):
        _summarise_batch(ids, results)

    assert (
        "Batch complete: 1/4 slides tiled, 1 skipped, 1 unannotated, 1 failed."
        in _messages(caplog)
    )


def test_batch_summary_reports_tiers_and_labels(caplog):
    ids = [1, 2]
    results = [
        _result(1, "done", "local", G3=5, benign=10),
        _result(2, "done", "network", G3=1, G5=2),
    ]

    with caplog.at_level("INFO"):
        _summarise_batch(ids, results)

    messages = _messages(caplog)
    assert "Served by tier: local=1, network=1" in messages
    assert "Tiles by label: G3=6, G5=2, benign=10" in messages
    assert "Total tiles: 18" in messages


def test_batch_summary_lists_problem_ids(caplog):
    ids = [1, 2, 3]
    results = [_result(1, "failed"), _result(2, "unannotated"), _result(3, "skipped")]

    with caplog.at_level("INFO"):
        _summarise_batch(ids, results)

    messages = _messages(caplog)
    assert "Failed image IDs: 1" in messages
    assert "Unannotated (no ROIs) image IDs: 2" in messages
    assert "Skipped image IDs: 3" in messages


def test_batch_summary_exits_non_zero_only_when_everything_failed():
    with pytest.raises(SystemExit, match="all 2 images failed"):
        _summarise_batch([1, 2], [_result(1, "failed"), _result(2, "failed")])

    # one bad slide in a batch must not fail a scheduled run
    _summarise_batch([1, 2], [_result(1, "failed"), _result(2, "done", "local")])


def test_batch_summary_of_an_empty_batch_exits_zero():
    _summarise_batch([], [])


def test_a_batch_of_only_unannotated_slides_is_not_a_failure():
    _summarise_batch([1, 2], [_result(1, "unannotated"), _result(2, "unannotated")])


# --- --help stays pure Python ---------------------------------------------


def test_tile_help_works_without_numpy_pyvips_or_omero():
    """--help must survive a broken native dependency (see lavlab/cli.py)."""
    program = textwrap.dedent(
        """
        import sys

        BLOCKED = {"numpy", "pyvips", "omero", "skimage", "tifffile", "SimpleITK"}

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError("blocked for this test: " + name)
                return None

        sys.meta_path.insert(0, Blocker())

        from lavlab.cli import main
        try:
            main(["tile", "--help"])
        except SystemExit as exc:
            sys.exit(exc.code or 0)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
        ),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--coords-only" in completed.stdout
    assert "--background-label" in completed.stdout


def test_root_help_works_without_heavy_imports():
    program = (
        "from lavlab.cli import build_parser; build_parser(); import sys; "
        "print(sorted(m for m in ('numpy','pyvips','omero') if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
        ),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_dilate_mask_grows_by_the_radius():
    mask = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True

    grown = tiling.dilate_mask(mask, 3.0)

    assert grown[10, 13] and grown[7, 10]
    assert not grown[10, 15]


def test_dilate_mask_is_a_no_op_without_a_margin():
    mask = np.zeros((8, 8), dtype=bool)
    mask[4, 4] = True

    assert tiling.dilate_mask(mask, 0.0) is mask
    # a sub-pixel margin still rounds up to one pixel, erring towards
    # keeping doubtful tiles out of the background class
    assert tiling.dilate_mask(mask, 0.4).sum() > 1
    # nothing to grow
    assert tiling.dilate_mask(np.zeros((8, 8), dtype=bool), 5.0).sum() == 0


def test_dilate_mask_falls_back_without_isotropic_dilation(monkeypatch):
    """Older scikit-image has no isotropic_dilation; the footprint path must match."""
    from skimage import morphology

    mask = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True
    expected = tiling.dilate_mask(mask, 3.0)

    monkeypatch.delattr(morphology, "isotropic_dilation", raising=False)
    assert np.array_equal(tiling.dilate_mask(mask, 3.0), expected)


# ---------------------------------------------------------------------------
# --erode: tile centres only inside the shrunken ROI
# ---------------------------------------------------------------------------


def test_center_boxes_at_half_a_tile_are_the_tiles_own_windows():
    tiles = build_grid(80, 80, size=8, overlap=0, ds_total=1.0)

    boxes = tiling.center_boxes(tiles, analysis_ds=1, radius_px=4.0)

    assert all(np.array_equal(b, w) for b, w in zip(boxes, tile_windows(tiles, 1)))


def test_erosion_keeps_flush_tiles_on_every_side_and_nothing_poking_out():
    """Symmetric: a tile flush against any ROI edge passes, one 1 px out fails."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[16:32, 16:32] = True
    integ = integral_image(mask)
    tiles = [
        GridTile(0, 0, 0, 0, 16, 16, 8, 8),  # flush top-left
        GridTile(0, 0, 0, 0, 24, 24, 8, 8),  # flush bottom-right
        GridTile(0, 0, 0, 0, 15, 16, 8, 8),  # 1 px out on the left
        GridTile(0, 0, 0, 0, 25, 24, 8, 8),  # 1 px out on the right
    ]

    inside = window_fractions(integ, *tiling.center_boxes(tiles, 1, 4.0)) >= 1.0

    assert inside.tolist() == [True, True, False, False]


def test_erode_mask_finds_room_only_where_a_tile_fits():
    mask = np.zeros((40, 40), dtype=bool)
    mask[16:32, 16:32] = True  # 16 px: room for an 8 px square
    mask[2:7, 2:7] = True  # 5 px: no room

    eroded = tiling.erode_mask(mask, 4.0)

    assert eroded[16:32, 16:32].any()
    assert not eroded[0:10, 0:10].any()


def test_erode_mask_is_a_no_op_under_one_pixel():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    assert tiling.erode_mask(mask, 0.0) is mask
    assert tiling.erode_mask(mask, 0.5) is mask


def _edge_case_analysis(**params):
    # ROI [16, 36) x [16, 32): column 4 (x 32..40) is half inside it.
    tiles = build_grid(80, 80, size=8, overlap=0, ds_total=1.0)
    shapes = [_shape(7, _square_ring(16, 16, 36, 32), "g3")]
    details = {}
    records, stats, _ = analyze_slide(
        tiles,
        _analysis_thumbnail(),
        shapes,
        analysis_ds=1,
        params=_analysis_params(min_coverage=0.4, **params),
        lookup=build_alias_lookup({"G3": ["g3"]}),
        pixel_size_x=1.0,
        details=details,
    )
    return _labels_by_cell(records), stats, details


def test_erosion_drops_a_tile_that_only_straddles_the_roi():
    labels, stats, details = _edge_case_analysis()

    assert labels[(2, 2)] == "G3"  # wholly inside
    assert (2, 4) not in labels  # half inside: enough coverage, centre outside
    assert stats["dropped_roi_edge"] >= 1
    assert details["erode_px"] == 4.0  # auto = half an 8 px tile


def test_erode_zero_restores_coverage_only_labelling():
    labels, stats, _ = _edge_case_analysis(erode_um=0.0)

    assert labels[(2, 4)] == "G3"
    assert stats["dropped_roi_edge"] == 0


def test_erode_in_micrometres_uses_the_pixel_size():
    # 2 um at 1 um/px = 2 px: the half-inside tile's centre (x=36) is on the
    # edge pixel, which is outside the ROI whatever the radius.
    labels, _stats, details = _edge_case_analysis(erode_um=2.0)

    assert details["erode_px"] == 2.0
    assert (2, 4) not in labels


def test_erosion_reports_rois_too_small_for_a_tile():
    tiles = build_grid(80, 80, size=8, overlap=0, ds_total=1.0)
    shapes = [
        _shape(7, _square_ring(16, 16, 40, 40), "g3"),  # big: survives
        _shape(8, _square_ring(50, 50, 55, 55), "g3"),  # 5 px: smaller than a tile
    ]
    details = {}

    analyze_slide(
        tiles,
        _analysis_thumbnail(),
        shapes,
        analysis_ds=1,
        params=_analysis_params(),
        lookup=build_alias_lookup({"G3": ["g3"]}),
        pixel_size_x=1.0,
        details=details,
    )

    assert details["selected_shapes"] == {7: "G3", 8: "G3"}
    assert details["lost_shapes"] == {8: "G3"}


def test_group_rois_groups_shapes_by_their_omero_roi():
    groups = tiling.group_rois([1, 2, 3], {1: 100, 2: 100})

    assert groups[100] == {1, 2}
    assert groups[-3] == {3}  # unknown ROI: its own group


# ---------------------------------------------------------------------------
# tile_slide: dry run, scales, metadata, combined manifest, resume
# ---------------------------------------------------------------------------


@needs_omero
def test_dry_run_counts_everything_and_writes_nothing(mocked_slide, tmp_path):
    out = tmp_path / "out"

    result = tiling.tile_slide(
        None,
        _FakeImage(),
        _analysis_params(max_background_tiles=3),
        str(out),
        dry_run=True,
        manifest_path=str(tmp_path / "dataset.csv"),
    )

    assert result.dry_run is True
    assert result.status == "done"
    assert not out.exists()
    assert not (tmp_path / "dataset.csv").exists()
    assert result.label_counts["G3"] > 0
    assert result.label_counts["benign"] == 3
    assert result.available["benign"] > 3  # before the cap
    assert result.roi_total == 1
    assert result.roi_counts[("G3", 70)] == result.label_counts["G3"]


@needs_omero
def test_dry_run_lists_an_roi_that_erosion_empties(mocked_slide, tmp_path):
    mocked_slide.value = [
        _shape(7, _square_ring(16, 16, 32, 32), "g3"),
        _shape(8, _square_ring(50, 50, 54, 54), "g5"),
    ]

    result = tiling.tile_slide(
        None, _FakeImage(), _analysis_params(), str(tmp_path / "out"), dry_run=True
    )

    assert result.roi_total == 2
    assert result.lost_rois == [("G5", 80)]
    assert result.roi_counts[("G5", 80)] == 0


@needs_omero
def test_scales_write_concentric_pairs(mocked_slide, tmp_path):
    out = tmp_path / "out"
    params = _analysis_params(scales=[1.0, 2.0], background_label=None)

    result = tiling.tile_slide(None, _FakeImage(), params, str(out))

    rows = list(
        csv.DictReader(open(out / "N101" / "N101_S08_HE" / tiling.MANIFEST_NAME))
    )
    by_scale = {}
    for row in rows:
        by_scale.setdefault(row["mpp"], []).append(row)
    assert set(by_scale) == {"1.0", "2.0"}
    fine = {(r["x"], r["y"]) for r in by_scale["1.0"]}
    coarse = {(r["x"], r["y"]) for r in by_scale["2.0"]}
    assert fine == coarse  # same centres at both scales
    assert result.written == len(rows) == 2 * sum(result.label_counts.values())
    sample = by_scale["2.0"][0]
    assert sample["tile_path"].endswith(f"_x{sample['x']}_y{sample['y']}_mpp2.0.png")
    assert int(sample["w0"]) == 16  # 8 px at 2 um/px on a 1 um/px slide
    assert (out / sample["tile_path"]).is_file()


def test_output_mpps_lists_the_finest_scale_first():
    assert TileParams(mpp=0.5).output_mpps() == [0.5]
    assert TileParams(mpp=0.5, scales=[1.0, 0.5]).output_mpps() == [0.5, 1.0]
    assert TileParams(mpp=None, downsample=4.0).output_mpps() == [None]


def test_scale_tile_is_concentric():
    tile = GridTile(0, 0, 0, 0, 100, 200, 64, 64)

    bigger = tiling.scale_tile(tile, ds_total=2.0, size=64)

    assert tiling.tile_center(bigger) == tiling.tile_center(tile)
    assert (bigger.w0, bigger.h0) == (128, 128)
    assert not tiling.tile_fits(tiling.scale_tile(tile, 10.0, 64), 1000, 1000)


def test_write_tile_embeds_metadata_in_png_and_jpeg(tmp_path):
    from PIL import Image

    pixels = np.full((16, 16, 3), 100, dtype=np.uint8)
    meta = {"omero_image_id": 5, "roi_id": 9, "mpp": 0.5}

    tiling.write_tile(pixels, str(tmp_path / "t.png"), metadata=meta)
    tiling.write_tile(pixels, str(tmp_path / "t.jpg"), metadata=meta, quality=90)

    assert json.loads(Image.open(tmp_path / "t.png").text["lavlab"]) == meta
    assert json.loads(Image.open(tmp_path / "t.jpg").getexif()[270]) == meta
    # atomic: no temp files left behind
    assert sorted(os.listdir(tmp_path)) == ["t.jpg", "t.png"]


def test_write_tile_quality_changes_jpeg_size(tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    tiling.write_tile(pixels, str(tmp_path / "lo.jpg"), quality=20)
    tiling.write_tile(pixels, str(tmp_path / "hi.jpg"), quality=95)

    assert (tmp_path / "lo.jpg").stat().st_size < (tmp_path / "hi.jpg").stat().st_size


def _manifest_rows(n, image_id=1, prefix="N101/N101_S08_HE/G3/t"):
    rows = []
    for i in range(n):
        row = dict.fromkeys(tiling.MANIFEST_COLUMNS, "")
        row.update(tile_path=f"{prefix}{i}.png", omero_image_id=image_id, label="G3")
        rows.append(row)
    return rows


def test_append_manifest_accumulates_without_duplicates(tmp_path):
    out = tmp_path / "tiles"
    manifest = tmp_path / "index" / "dataset.csv"

    assert tiling.append_manifest(str(manifest), _manifest_rows(3), str(out)) == 3
    # a repeated / resumed run adds nothing new
    assert tiling.append_manifest(str(manifest), _manifest_rows(3), str(out)) == 0
    # a second command adds its own rows
    assert (
        tiling.append_manifest(
            str(manifest), _manifest_rows(2, 2, "X/Y/G5/u"), str(out)
        )
        == 2
    )

    rows = list(csv.DictReader(open(manifest)))
    assert len(rows) == 5
    # paths are rebased onto the manifest's own folder
    assert rows[0]["tile_path"] == os.path.join(
        "..", "tiles", "N101/N101_S08_HE/G3/t0.png"
    )


def test_append_manifest_refuses_a_file_with_other_columns(tmp_path):
    manifest = tmp_path / "old.csv"
    manifest.write_text("image_id,path\n1,a.png\n")

    with pytest.raises(TilingError, match="different columns"):
        tiling.append_manifest(str(manifest), _manifest_rows(1), str(tmp_path))


@needs_omero
def test_tile_slide_appends_to_the_combined_manifest(mocked_slide, tmp_path):
    out = tmp_path / "out"
    manifest = tmp_path / "dataset.csv"
    params = _analysis_params()

    tiling.tile_slide(None, _FakeImage(), params, str(out), manifest_path=str(manifest))
    first = list(csv.DictReader(open(manifest)))
    # a skipped (already finished) slide still lands in a fresh manifest
    other = tmp_path / "other.csv"
    tiling.tile_slide(
        None,
        _FakeImage(),
        params,
        str(out),
        skip_existing=True,
        manifest_path=str(other),
    )

    assert first
    assert len(list(csv.DictReader(open(other)))) == len(first)
    assert all((tmp_path / r["tile_path"]).is_file() for r in first)


@needs_omero
def test_skip_existing_resumes_an_interrupted_slide(mocked_slide, tmp_path):
    out = tmp_path / "out"
    params = _analysis_params()
    first = tiling.tile_slide(None, _FakeImage(), params, str(out))
    slide_dir = out / "N101" / "N101_S08_HE"

    # simulate an interruption: no manifest yet, one tile never written
    rows = tiling.read_manifest(str(slide_dir))
    (slide_dir / tiling.MANIFEST_NAME).unlink()
    (out / rows[0]["tile_path"]).unlink()
    marker = out / rows[1]["tile_path"]
    marker.write_bytes(b"kept")

    result = tiling.tile_slide(None, _FakeImage(), params, str(out), skip_existing=True)

    assert result.status == "done"
    assert result.written == 1
    assert result.reused == first.written - 1
    assert marker.read_bytes() == b"kept"  # reused, not re-read
    assert slide_is_complete(str(slide_dir))


@needs_omero
def test_an_interrupted_rerun_never_looks_finished(mocked_slide, tmp_path, monkeypatch):
    """A crash mid-rewrite must not leave the old manifest beside new params."""
    out = tmp_path / "out"
    tiling.tile_slide(None, _FakeImage(), _analysis_params(), str(out))

    def boom(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(tiling, "write_tile", boom)
    with pytest.raises(RuntimeError):
        tiling.tile_slide(
            None, _FakeImage(), _analysis_params(size=16), str(out), override=True
        )

    assert not slide_is_complete(str(out / "N101" / "N101_S08_HE"))


# ---------------------------------------------------------------------------
# CLI: the dataset flags
# ---------------------------------------------------------------------------


def test_scales_build_a_multi_scale_run_at_the_finest_mpp():
    params = _build_params(_tile_args("--roi", "--all", "--scales", "1.0,0.5"))

    assert params.mpp == 0.5
    assert params.scales == [0.5, 1.0]


def test_a_single_scale_is_just_mpp():
    params = _build_params(_tile_args("--roi", "--all", "--scales", "0.25"))

    assert params.mpp == 0.25
    assert params.scales is None


def test_scales_reject_nonsense():
    with pytest.raises(SystemExit):
        _tile_args("--whole", "--scales", "0.5,abc")
    with pytest.raises(SystemExit):
        _tile_args("--whole", "--scales", "0.5,0.5")
    with pytest.raises(SystemExit):
        _tile_args("--whole", "--scales", "0.5", "--mpp", "0.5")


def test_erode_defaults_to_auto_and_accepts_micrometres_or_zero():
    assert _build_params(_tile_args("--roi", "--all")).erode_um is None
    assert (
        _build_params(_tile_args("--roi", "--all", "--erode", "auto")).erode_um is None
    )
    assert (
        _build_params(_tile_args("--roi", "--all", "--erode", "128")).erode_um == 128.0
    )
    assert _build_params(_tile_args("--roi", "--all", "--erode", "0")).erode_um == 0.0
    with pytest.raises(SystemExit):
        _tile_args("--roi", "--all", "--erode", "-5")


def test_erode_is_rejected_in_whole_mode():
    with pytest.raises(SystemExit, match="only applies to --roi"):
        _validate_args(_tile_args("--whole", "--erode", "100"))


def test_stride_is_the_other_way_to_say_overlap():
    params = _build_params(_tile_args("--whole", "--size", "512", "--stride", "256"))

    assert params.overlap == 256
    with pytest.raises(SystemExit, match="--stride"):
        _validate_args(_tile_args("--whole", "--size", "64", "--stride", "65"))
    with pytest.raises(SystemExit):
        _tile_args("--whole", "--stride", "10", "--overlap", "2")


def test_jpeg_format_and_quality():
    params = _build_params(_tile_args("--whole", "--format", "jpeg"))
    assert params.fmt == "jpg"
    assert params.quality == 90

    params = _build_params(_tile_args("--whole", "--format", "jpg", "--quality", "80"))
    assert params.quality == 80

    assert _build_params(_tile_args("--whole")).quality is None
    with pytest.raises(SystemExit, match="--quality only applies"):
        _validate_args(_tile_args("--whole", "--quality", "80"))
    with pytest.raises(SystemExit, match="1-100"):
        _validate_args(_tile_args("--whole", "--format", "jpg", "--quality", "0"))


def test_tissue_threshold_is_an_alias():
    args = _tile_args("--whole", "--tissue-threshold", "0.25")

    assert _build_params(args).tissue_thresh == 0.25


def test_dry_run_manifest_and_connections_parse():
    args = _tile_args(
        "--roi", "--all", "--dry-run", "--manifest", "d.csv", "--connections", "8"
    )
    _validate_args(args)

    assert args.dry_run and args.manifest == "d.csv" and args.connections == 8
    with pytest.raises(SystemExit, match="--connections"):
        _validate_args(_tile_args("--whole", "--connections", "0"))


def test_batch_summary_reports_rois_lost_to_erosion(caplog):
    from collections import Counter

    results = [
        tiling.SlideResult(
            1,
            "done",
            tier="local",
            label_counts=Counter(G3=5),
            roi_total=3,
            roi_counts=Counter({("G3", 1): 5, ("G5", 2): 0, ("G5", 3): 0}),
            lost_rois=[("G5", 2)],
            dry_run=True,
            available=Counter(G3=9),
        )
    ]

    with caplog.at_level("INFO"):
        _summarise_batch([1], results)

    messages = _messages(caplog)
    assert "Dry run: nothing was written." in messages
    assert "Available before caps: G3=9" in messages
    assert (
        "ROIs: 3 selected, 2 gave no tiles, 1 too small for a tile after --erode "
        "(G5=1)" in messages
    )


@needs_omero
def test_scales_cut_real_pixels_from_the_right_pyramid_levels(monkeypatch, tmp_path):
    """End to end over a real two-level pyramid: no fake reader, real files."""
    import pyvips as pv

    import lavlab.roi as roi_module

    base = _synthetic_slide(256)
    source = tmp_path / "slide.tiff"
    _write_pyramid(source, base)
    size = 16
    analysis_ds = tiling.analysis_downsample(size)  # 16 px tiles at 1 um/px -> 1/2
    # glass all round, stained tissue in the middle (a uniform image has no
    # contrast, so the tissue detector would rightly find nothing)
    thumb = np.full((256 // analysis_ds, 256 // analysis_ds, 3), 250, np.uint8)
    thumb[8:-8, 8:-8] = (180, 60, 160)

    monkeypatch.setattr(
        tiling, "choose_tier", lambda c, i, force_local=False: ("local", str(source))
    )
    monkeypatch.setattr(tiling, "_load_thumbnail_local", lambda p, d: thumb)
    # one G3 ROI covering level-0 [32, 224), given at analysis scale
    ring = _square_ring(
        32 // analysis_ds, 32 // analysis_ds, 224 // analysis_ds, 224 // analysis_ds
    )
    monkeypatch.setattr(
        roi_module, "get_shapes_as_points", lambda image, **kw: [_shape(7, ring, "g3")]
    )
    monkeypatch.setattr(roi_module, "shape_roi_ids", lambda image: {7: 70})

    out = tmp_path / "out"
    params = _analysis_params(
        scales=[1.0, 2.0], size=size, background_label=None, max_tiles_per_label=6
    )
    result = tiling.tile_slide(None, _FakeImage(size=256), params, str(out))

    rows = tiling.read_manifest(str(out / "N101" / "N101_S08_HE"))
    assert result.written == len(rows) == 12  # 6 centres x 2 scales
    level1 = base[::2, ::2]
    for row in rows:
        tile = pv.Image.new_from_file(str(out / row["tile_path"])).numpy()
        x0, y0, w0 = int(row["x0"]), int(row["y0"]), int(row["w0"])
        if row["mpp"] == "1.0":
            assert row["level"] == "0"
            expected = base[y0 : y0 + size, x0 : x0 + size]
        else:
            assert row["level"] == "1" and w0 == 2 * size
            expected = level1[y0 // 2 : y0 // 2 + size, x0 // 2 : x0 // 2 + size]
        assert np.array_equal(tile[:, :, :3], expected)
        assert row["roi_id"] == "70"


# --- making --erode visible ------------------------------------------------


def test_erode_zero_warns_about_label_noise(caplog):
    from lavlab.commands.tile import _explain_erosion

    with caplog.at_level("INFO"):
        _explain_erosion(_build_params(_tile_args("--roi", "--all", "--erode", "0")))

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "label noise" in warnings[0].getMessage()


def test_default_erosion_says_nothing_for_a_single_scale(caplog):
    from lavlab.commands.tile import _explain_erosion

    with caplog.at_level("INFO"):
        _explain_erosion(_build_params(_tile_args("--roi", "--all")))
        _explain_erosion(_build_params(_tile_args("--whole")))

    assert caplog.records == []


def test_scales_explain_the_inset_and_how_to_keep_every_scale_inside(caplog):
    from lavlab.commands.tile import _explain_erosion

    args = _tile_args("--roi", "--all", "--size", "512", "--scales", "0.5,1.0")
    with caplog.at_level("INFO"):
        _explain_erosion(_build_params(args))

    message = caplog.records[0].getMessage()
    assert "kept 128 um inside" in message
    assert "reach 256 um" in message
    assert "pass --erode 256" in message


@needs_omero
def test_the_applied_erosion_is_reported_in_micrometres(mocked_slide, tmp_path):
    out = tmp_path / "out"

    # 8 px tiles at 1 um/px: the default inset is half a tile, 4 um
    auto = tiling.tile_slide(
        None, _FakeImage(), _analysis_params(), str(out), dry_run=True
    )
    off = tiling.tile_slide(
        None, _FakeImage(), _analysis_params(erode_um=0.0), str(out), dry_run=True
    )
    done = tiling.tile_slide(None, _FakeImage(), _analysis_params(), str(out))

    assert auto.erode_um == 4.0
    assert off.erode_um == 0.0
    saved = read_tile_params(done.slide_dir)
    assert saved["erode_um_effective"] == 4.0
