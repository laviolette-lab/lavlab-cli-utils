# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Build the dependency-free wheel containing the Nuitka executable."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.build_py import build_py as _build_py

from build_native import (
    IMAGECODECS_NUITKA_FLAGS,
    PYDICOM_NUITKA_FLAGS,
    native_target,
    normalize_binary_name,
    omero_ice_modules,
    restore_unpatched_vips,
)


class build_py(_build_py):
    """Stage the native binary before setuptools copies package data."""

    def run(self):
        native_target()
        project_dir = Path(__file__).parent
        # Build outside the package. Nuitka's --include-package-data=lavlab
        # sweeps everything under src/lavlab/, so building straight into
        # src/lavlab/bin/ lets a previous build's dist folder get swept into the
        # next one (observed locally as a nested lavlab/bin/__main__.dist/
        # inside the dist itself).
        build_root = project_dir / "build" / "nuitka"
        staged_dist = project_dir / "src" / "lavlab" / "bin" / "dist"
        # Both have to go before Nuitka starts, not just before they are
        # rewritten: a previous run's staged dist lives under src/lavlab/, so
        # leaving it in place would feed it straight back into this build.
        for stale in (build_root, staged_dist):
            if stale.exists():
                shutil.rmtree(stale)
        prebuilt_dist = os.environ.get("LAVLAB_PREBUILT_DIST")
        if prebuilt_dist:
            nuitka_dist = Path(prebuilt_dist).resolve()
            if not (nuitka_dist / "lavlab-bin").is_file():
                raise RuntimeError(
                    f"LAVLAB_PREBUILT_DIST must contain lavlab-bin: {nuitka_dist}"
                )
        else:
            if sys.version_info[:2] != (3, 12):
                raise RuntimeError(
                    "The standalone binary must be compiled with Python 3.12. "
                    "Set LAVLAB_PREBUILT_DIST to package an existing build."
                )

            build_root.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "nuitka",
                "--standalone",
                "--assume-yes-for-downloads",
                f"--output-dir={build_root}",
                "--output-filename=lavlab-bin",
                "--include-package=lavlab",
                "--include-package-data=lavlab",
                *PYDICOM_NUITKA_FLAGS,
                *IMAGECODECS_NUITKA_FLAGS,
                str(project_dir / "src" / "lavlab" / "__main__.py"),
            ]
            command[3:3] = [
                f"--include-module={name}" for name in omero_ice_modules()
            ]

            extra_args = os.environ.get("LAVLAB_NUITKA_ARGS", "")
            if extra_args:
                command[3:3] = extra_args.split()

            # src layout: --include-package=lavlab resolves through sys.path,
            # and the repo root (the cwd) no longer contains the package.
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(project_dir / "src"), env.get("PYTHONPATH")])
            )
            subprocess.run(command, check=True, env=env)

            nuitka_dist = build_root / "__main__.dist"
            compiled_binary = normalize_binary_name(nuitka_dist)
            restore_unpatched_vips(nuitka_dist)

            if not compiled_binary.is_file():
                raise RuntimeError(
                    f"Nuitka did not produce the expected binary: {compiled_binary}"
                )

        # Keep one stable path for both freshly compiled and prebuilt binaries.
        staged_dist.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(nuitka_dist, staged_dist, symlinks=True)
        staged_in_build_lib = Path(self.build_lib) / "lavlab" / "bin"
        if staged_in_build_lib.exists():
            shutil.rmtree(staged_in_build_lib)

        super().run()


def native_platform_tag() -> str:
    """Wheel platform tag for the machine the Nuitka binary is compiled on.

    The binary only runs on the OS/arch it was built on, and at best on that
    OS release (macOS) or glibc (Linux) and newer -- so tag for the build host.
    """
    target = native_target()
    if sys.platform == "darwin":
        major = platform.mac_ver()[0].split(".")[0]
        return f"macosx_{major}_0_arm64"
    if sys.platform.startswith("linux"):
        libc, version = platform.libc_ver()
        if libc != "glibc":
            raise RuntimeError(f"Unsupported libc for a manylinux tag: {libc!r}")
        major, minor = version.split(".")[:2]
        return f"manylinux_{major}_{minor}_{target.split('-')[-1]}"
    raise RuntimeError(f"No wheel platform tag for {sys.platform}")


class bdist_wheel(_bdist_wheel):
    """Tag the wheel py3-none-<platform> instead of py3-none-any.

    The embedded binary is platform-specific and independent of the supported
    Python 3.11-3.14 installer runtime. Pass --plat-name to override the
    platform part.
    """

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        if self.plat_name_supplied and self.plat_name:
            plat = self.plat_name.replace("-", "_").replace(".", "_")
        else:
            plat = native_platform_tag()
        return "py3", "none", plat


setup(cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel})