# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Import hygiene for lavlab.monai: the CLI never touches MONAI or PyTorch.

Each check runs in a fresh interpreter, so modules imported by other tests
cannot leak into sys.modules. Needs nothing beyond the standard library.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _run(program: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        cwd=SRC,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# import hygiene: the CLI never touches MONAI or torch
# ---------------------------------------------------------------------------


def test_importing_lavlab_and_tile_help_import_neither_monai_nor_torch():
    completed = _run(
        """
        import sys
        import lavlab
        import lavlab.tiling
        import lavlab.monai
        from lavlab.cli import main
        try:
            main(["tile", "--help"])
        except SystemExit:
            pass
        heavy = sorted(m for m in sys.modules if m.split(".")[0] in ("monai", "torch"))
        print("HEAVY", heavy, file=sys.stderr)
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert "--coords-only" in completed.stdout
    assert "HEAVY []" in completed.stderr


def test_every_cli_command_parses_without_monai_or_torch():
    completed = _run(
        """
        import sys
        from lavlab.cli import build_parser
        parser = build_parser()
        for argv in (["tile", "1", "-o", "x", "--roi", "--all"], ["lr", "1"],
                     ["roi", "1"], ["meta", "1"]):
            try:
                parser.parse_args(argv)
            except SystemExit:
                pass
        print(sorted(m for m in sys.modules if m.split(".")[0] in ("monai", "torch")))
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_missing_monai_names_the_extra_to_install():
    completed = _run(
        """
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("monai", "torch"):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None

        sys.meta_path.insert(0, Blocker())
        import lavlab.monai
        try:
            lavlab.monai.OmeroPatchDataset
        except ImportError as exc:
            print("ERROR", exc)
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert "ERROR" in completed.stdout
    assert "pip install 'lavlab-cli-utils[monai]'" in completed.stdout


def test_build_datalist_is_importable_without_monai():
    completed = _run(
        """
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("monai", "torch"):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None

        sys.meta_path.insert(0, Blocker())
        from lavlab.monai import build_datalist, tile_params
        print(callable(build_datalist), tile_params(all_labels=True).size)
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True 224"
