# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for lavlab.naming's fs_map resolution -- pure filesystem/regex
logic, no OMERO required.

Covers the subject_glob mechanism and the bundled default_fs_map.yaml's
HE / other-stain / biopsy entries, ported from legacy/batch_lr.py's
proven path resolution logic.
"""

from __future__ import annotations

import dataclasses

import pytest

from lavlab.config import ConfigError, FsMapEntry, FsMapGroup, load_fs_map
from lavlab.naming import resolve_fs_map_dir


def test_subject_glob_resolves_inconsistent_prefix(tmp_path):
    (tmp_path / "1101").mkdir()
    (tmp_path / "717").mkdir()

    entry = FsMapEntry(
        match=__import__("re").compile(r"^N(?P<subject>\d+)_X\.ome\.tiff$"),
        base_dir=str(tmp_path),
        formatted_dir="out",
        subject_glob="subject",
    )
    group = FsMapGroup(group_id="1", name="test", maps=[entry])
    fs_map = {"1": group}

    resolved = resolve_fs_map_dir(fs_map, 1, "N101_X.ome.tiff")
    assert resolved == str(tmp_path / "1101" / "out")

    resolved_3digit = resolve_fs_map_dir(fs_map, 1, "N717_X.ome.tiff")
    assert resolved_3digit == str(tmp_path / "717" / "out")


def test_subject_glob_no_match_falls_through_to_next_entry(tmp_path):
    (tmp_path / "1101").mkdir()

    entry_no_match = FsMapEntry(
        match=__import__("re").compile(r"^N(?P<subject>\d+)_X\.ome\.tiff$"),
        base_dir=str(tmp_path),
        formatted_dir="out",
        subject_glob="subject",
    )
    entry_fallback = FsMapEntry(
        match=__import__("re").compile(r"^N(?P<subject>\d+)_X\.ome\.tiff$"),
        base_dir=str(tmp_path),
        formatted_dir="fallback",
        subject_glob=None,
    )
    group = FsMapGroup(group_id="1", name="test", maps=[entry_no_match, entry_fallback])
    fs_map = {"1": group}

    # subject "999" has no matching glob under tmp_path -> first entry skipped,
    # second entry (no subject_glob) used directly against base_dir.
    resolved = resolve_fs_map_dir(fs_map, 1, "N999_X.ome.tiff")
    assert resolved == str(tmp_path / "fallback")


def test_missing_base_dir_raises_config_error(tmp_path):
    entry = FsMapEntry(
        match=__import__("re").compile(r"^N(?P<subject>\d+)_X\.ome\.tiff$"),
        base_dir=str(tmp_path / "does_not_exist"),
        formatted_dir="out",
    )
    group = FsMapGroup(group_id="1", name="test", maps=[entry])
    fs_map = {"1": group}

    with pytest.raises(ConfigError):
        resolve_fs_map_dir(fs_map, 1, "N101_X.ome.tiff")


def _bundled_entries_against(tmp_path):
    """Load the real bundled default_fs_map.yaml, but redirect every entry's
    base_dir to tmp_path so it's testable without /Volumes/Siren mounted."""
    fs_map = load_fs_map(None)
    group = fs_map["1"]
    retargeted = [
        dataclasses.replace(entry, base_dir=str(tmp_path)) for entry in group.maps
    ]
    return {"1": dataclasses.replace(group, maps=retargeted)}


def test_bundled_fs_map_he_slide_has_no_stain_subdir(tmp_path):
    (tmp_path / "1101").mkdir()
    fs_map = _bundled_entries_against(tmp_path)

    resolved = resolve_fs_map_dir(fs_map, 1, "N101_S06_HE.ome.tiff")
    assert resolved == str(tmp_path / "1101" / "Hist" / "6" / "Huron")


def test_bundled_fs_map_other_stain_gets_subdir(tmp_path):
    (tmp_path / "1101").mkdir()
    fs_map = _bundled_entries_against(tmp_path)

    resolved = resolve_fs_map_dir(fs_map, 1, "N101_S06_CD3.ome.tiff")
    assert resolved == str(tmp_path / "1101" / "Hist" / "6" / "Huron" / "CD3")


def test_bundled_fs_map_strips_leading_zeros_and_keeps_deeper_suffix(tmp_path):
    (tmp_path / "1101").mkdir()
    fs_map = _bundled_entries_against(tmp_path)

    resolved = resolve_fs_map_dir(fs_map, 1, "N101_S06_Deeper2_HE.ome.tiff")
    assert resolved == str(tmp_path / "1101" / "Hist" / "6_Deeper2" / "Huron")


def test_bundled_fs_map_biopsy_he(tmp_path):
    (tmp_path / "1101").mkdir()
    fs_map = _bundled_entries_against(tmp_path)

    resolved = resolve_fs_map_dir(fs_map, 1, "N101_LeftLobe_HE_Biopsy.ome.tiff")
    assert resolved == str(tmp_path / "1101" / "Hist" / "LeftLobe" / "Huron")


def test_bundled_fs_map_biopsy_other_stain(tmp_path):
    (tmp_path / "1101").mkdir()
    fs_map = _bundled_entries_against(tmp_path)

    resolved = resolve_fs_map_dir(fs_map, 1, "N101_RightLobe_CD3_TURP.ome.tiff")
    assert resolved == str(tmp_path / "1101" / "Hist" / "RightLobe" / "Huron" / "CD3")


def test_bundled_fs_map_unmatched_name_returns_none(tmp_path):
    (tmp_path / "1101").mkdir()
    fs_map = _bundled_entries_against(tmp_path)

    assert resolve_fs_map_dir(fs_map, 1, "not_a_recognised_name.ome.tiff") is None
