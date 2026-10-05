# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Execute the bundled native lavlab binary without importing runtime packages."""

from __future__ import annotations

import os
import stat
import sys
from importlib.resources import files


def main() -> None:
    # A Nuitka --standalone build is a directory: the executable sits next to
    # the shared libraries it resolves through RPATH=$ORIGIN, so it has to be
    # run from where it was installed rather than copied out on its own.
    binary = files("lavlab").joinpath("bin", "dist", "lavlab-bin")
    if not binary.is_file():
        raise RuntimeError(
            "The lavlab native binary is missing. This package must be installed "
            "from a platform wheel, not a source checkout."
        )

    path = str(binary)
    # Wheels do not reliably preserve the executable bit through every
    # install path, and a non-executable binary would fail here as a bare
    # PermissionError with no indication of why.
    mode = os.stat(path).st_mode
    if not mode & stat.S_IXUSR:
        os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    os.execv(path, [path, *sys.argv[1:]])
