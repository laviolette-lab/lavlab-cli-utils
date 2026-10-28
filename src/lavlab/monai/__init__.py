# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""MONAI data layer for OMERO slides and ``lavlab tile`` output.

Two ways to train on the tiles ``lavlab tile`` chooses:

* :class:`OmeroPatchDataset` -- read patches from OMERO on demand. No
  extraction step, but every patch pays OMERO's latency every epoch.
* :class:`LavlabTileDataset` -- read tiles ``lavlab tile`` already wrote,
  via its manifest. The one to train on.

plus the parts they are built from: :class:`OmeroWSIReader` (a
``monai.data.BaseWSIReader``) and :func:`build_datalist`.

Needs the ``monai`` extra (``pip install 'lavlab-cli-utils[monai]'``) for
everything except :func:`build_datalist` and :func:`tile_params`. Names are
imported on first use, so ``import lavlab.monai`` never imports MONAI or
PyTorch by itself, and nothing else in ``lavlab`` imports this package.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_datalist": "lavlab.monai.datalist",
    "image_uri": "lavlab.monai.datalist",
    "parse_image_id": "lavlab.monai.datalist",
    "tile_params": "lavlab.monai.datalist",
    "OmeroWSI": "lavlab.monai.reader",
    "OmeroWSIReader": "lavlab.monai.reader",
    "LavlabTileDataset": "lavlab.monai.datasets",
    "OmeroPatchDataset": "lavlab.monai.datasets",
    "group_by_case": "lavlab.monai.datasets",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'lavlab.monai' has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
