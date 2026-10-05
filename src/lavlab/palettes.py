# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Color-to-text-label palettes used by ``lavlab meta roi textvalue``.

A palette maps an (r, g, b) tuple to a label. Palettes can be a built-in
name, or a path to a YAML/JSON file containing a list of entries like::

    - rgb: [25, 20, 255]
      text: "Seminal Vesicles"
    - hex: "#FF7A00"
      text: "HGPIN"
"""

from __future__ import annotations

import json
import os

import yaml

ColorMap = dict[tuple[int, int, int], str]

# The lab's historical hardcoded palette (from the ROI-comment notebook).
BUILTIN_PALETTES: dict[str, ColorMap] = {
    "default": {
        (25, 20, 255): "Seminal Vesicles",
        (0, 0, 0): "Atrophy",
        (255, 122, 0): "HGPIN",
        (48, 255, 50): "G3",
        (255, 250, 20): "G4FG",
        (254, 22, 255): "G4CG",
        (33, 255, 255): "G5",
        (181, 187, 253): "Vessel",
        (128, 128, 128): "Urethra",
        (255, 16, 0): "Exclusion ROI",
    }
}


def _parse_hex(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)


def _load_custom_palette(path: str) -> ColorMap:
    with open(path) as f:
        if path.endswith(".json"):
            entries = json.load(f)
        else:
            entries = yaml.safe_load(f)

    mapping: ColorMap = {}
    for entry in entries:
        text = entry["text"]
        if "rgb" in entry:
            rgb = tuple(entry["rgb"])
        elif "hex" in entry:
            rgb = _parse_hex(entry["hex"])
        else:
            raise ValueError(f"Palette entry missing 'rgb' or 'hex': {entry}")
        mapping[rgb] = text
    return mapping


def load_color_mapping(name_or_path: str) -> ColorMap:
    if name_or_path in BUILTIN_PALETTES:
        return BUILTIN_PALETTES[name_or_path]
    if not os.path.isfile(name_or_path):
        raise FileNotFoundError(
            f"'{name_or_path}' is not a built-in palette ({', '.join(BUILTIN_PALETTES)}) "
            "or an existing YAML/JSON file."
        )
    return _load_custom_palette(name_or_path)


def match_color(
    rgb: tuple[int, int, int], mapping: ColorMap, tolerance: int = 10
) -> str | None:
    r, g, b = rgb
    for (pr, pg, pb), label in mapping.items():
        if (
            abs(r - pr) <= tolerance
            and abs(g - pg) <= tolerance
            and abs(b - pb) <= tolerance
        ):
            return label
    return None
