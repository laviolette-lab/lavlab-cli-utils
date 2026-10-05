# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest
import yaml

from lavlab.palettes import _parse_hex, load_color_mapping, match_color


def test_load_color_mapping_builtin_default():
    mapping = load_color_mapping("default")
    assert mapping[(25, 20, 255)] == "Seminal Vesicles"
    assert len(mapping) == 10


def test_load_color_mapping_missing_raises_file_not_found():
    with pytest.raises(FileNotFoundError, match="default"):
        load_color_mapping("nonexistent_palette_xyz")


def test_load_color_mapping_custom_yaml_rgb_and_hex(tmp_path):
    path = tmp_path / "palette.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {"rgb": [10, 20, 30], "text": "Test Region"},
                {"hex": "#00FF00", "text": "Green Region"},
            ]
        )
    )

    mapping = load_color_mapping(str(path))

    assert mapping == {(10, 20, 30): "Test Region", (0, 255, 0): "Green Region"}


def test_load_color_mapping_custom_json(tmp_path):
    path = tmp_path / "palette.json"
    path.write_text(json.dumps([{"rgb": [1, 2, 3], "text": "JSON Region"}]))

    mapping = load_color_mapping(str(path))

    assert mapping == {(1, 2, 3): "JSON Region"}


def test_load_color_mapping_entry_missing_rgb_and_hex_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"text": "No color"}]))

    with pytest.raises(ValueError, match="rgb"):
        load_color_mapping(str(path))


def test_parse_hex_with_and_without_leading_hash():
    assert _parse_hex("#FF7A00") == (255, 122, 0)
    assert _parse_hex("FF7A00") == (255, 122, 0)


def test_match_color_exact_hit():
    mapping = {(25, 20, 255): "Seminal Vesicles"}
    assert match_color((25, 20, 255), mapping) == "Seminal Vesicles"


def test_match_color_within_tolerance():
    mapping = {(25, 20, 255): "Seminal Vesicles"}
    assert match_color((30, 25, 250), mapping, tolerance=10) == "Seminal Vesicles"


def test_match_color_outside_tolerance_returns_none():
    mapping = {(25, 20, 255): "Seminal Vesicles"}
    assert match_color((30, 25, 250), mapping, tolerance=3) is None


def test_match_color_no_entries_returns_none():
    assert match_color((1, 2, 3), {}) is None
