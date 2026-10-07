# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Nuitka hooks for preserving Linux libvips and onefile exit statuses."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from nuitka.plugins.PluginBase import NuitkaPluginBase


class LavlabNuitkaPlugin(NuitkaPluginBase):
    plugin_name = "lavlab-nuitka"

    def onStandaloneDistributionFinished(self, dist_dir: str) -> None:  # noqa: N802
        """Restore the pristine pyvips libvips before Nuitka packs onefile."""
        if not sys.platform.startswith("linux"):
            return

        dist_path = Path(dist_dir)
        patched = sorted(dist_path.glob("libvips*.so.*"))
        if not patched:
            return

        pristine = [
            path
            for search_path in map(Path, sys.path)
            if search_path.is_dir()
            for libs_dir in ("pyvips_binary.libs", "pyvips.libs")
            for path in search_path.glob(f"{libs_dir}/libvips*.so.*")
        ]
        if not pristine:
            raise RuntimeError(
                f"Nuitka bundled {patched[0].name} but no pristine copy was found in "
                "pyvips_binary.libs/ or pyvips.libs/. Refusing to ship a possibly "
                "corrupted libvips."
            )

        by_name = {path.name: path for path in pristine}
        for target in patched:
            source = by_name.get(target.name, pristine[0])
            shutil.copy2(source, target)
            print(f"Restored unpatched {target.name} from {source}")  # noqa: T201

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
