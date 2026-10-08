# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Nuitka hooks for compact Linux binaries and onefile exit statuses."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from nuitka.plugins.PluginBase import NuitkaPluginBase


class LavlabNuitkaPlugin(NuitkaPluginBase):
    plugin_name = "lavlab-nuitka"

    def onStandaloneDistributionFinished(self, dist_dir: str) -> None:  # noqa: N802
        """Remove unused data and debug symbols before Nuitka packs onefile."""
        dist_path = Path(dist_dir)
        for libvips in dist_path.rglob("libvips*"):
            if libvips.is_file():
                libvips.unlink()

        if not sys.platform.startswith("linux"):
            return

        for unused_data in (
            dist_path / "skimage" / "data",
            dist_path / "pydicom" / "data" / "test_files",
        ):
            if unused_data.is_dir():
                shutil.rmtree(unused_data)

        for path in dist_path.rglob("*"):
            if not path.is_file():
                continue
            try:
                with path.open("rb") as binary:
                    is_elf = binary.read(4) == b"\x7fELF"
            except OSError:
                continue
            if is_elf:
                subprocess.run(["strip", "--strip-unneeded", str(path)], check=True)

    def onGeneratedSourceCode(self, source_dir: str, onefile: bool) -> None:  # noqa: N802
        """Make Nuitka propagate child crashes instead of reporting success."""
        if not onefile:
            return

        bootstrap = Path(source_dir) / "static_src" / "OnefileBootstrap.c"
        source = bootstrap.read_text(encoding="utf-8")
        old = """        } else {
            exit_code = WEXITSTATUS(status);
        }
"""
        new = """        } else if (WIFEXITED(status)) {
            exit_code = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            exit_code = 128 + WTERMSIG(status);
        } else {
            exit_code = 2;
        }
"""
        if new in source:
            return
        if source.count(old) != 1:
            raise RuntimeError(
                "Unexpected Nuitka onefile bootstrap: cannot apply the child signal "
                "exit-status fix. Check nuitka_plugin.py against the pinned Nuitka."
            )
        bootstrap.write_text(source.replace(old, new), encoding="utf-8")
