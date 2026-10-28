# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""The one error raised when MONAI or PyTorch is not installed."""

from __future__ import annotations

EXTRA_HINT = (
    "lavlab.monai needs MONAI and PyTorch, which are not installed with the CLI. "
    "Install the optional extra: pip install 'lavlab-cli-utils[monai]'"
)


def is_missing_extra(exc: ImportError) -> bool:
    """Return whether *exc* means MONAI or PyTorch is simply not installed.

    An ImportError from *inside* an installed MONAI or PyTorch (a broken
    build, a missing native library) names some other module, and is
    re-raised as is: telling someone to install what they already have
    would send them the wrong way.

    :param exc: the error from importing ``monai`` or ``torch``
    :type exc: ImportError
    :return: whether to raise :data:`EXTRA_HINT` instead
    :rtype: bool
    """
    return (exc.name or "").split(".")[0] in {"monai", "torch"}
