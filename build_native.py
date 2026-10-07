# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Build the Nuitka onefile executable with OMERO's dynamic Ice modules."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PYDICOM_NUITKA_FLAGS = [
    "--include-module=pydicom.pixels.decoders.gdcm",
    "--include-module=pydicom.pixels.decoders.pillow",
    "--include-module=pydicom.pixels.decoders.pyjpegls",
    "--include-module=pydicom.pixels.decoders.pylibjpeg",
    "--include-module=pydicom.pixels.decoders.rle",
    "--include-module=pydicom.pixels.encoders.gdcm",
    "--include-module=pydicom.pixels.encoders.native",
    "--include-module=pydicom.pixels.encoders.pyjpegls",
    "--include-module=pydicom.pixels.encoders.pylibjpeg",
    "--include-package-data=pydicom",
]
IMAGECODECS_NUITKA_FLAGS = [
    "--include-module=imagecodecs._shared",
    "--include-module=imagecodecs._shared_cython",
    "--include-module=imagecodecs._imcd",
    "--include-module=imagecodecs._jpeg8",
    "--include-module=imagecodecs._ljpeg",
    "--include-module=imagecodecs._jpegsof3",
    "--include-module=imagecodecs._jpeg2k",
    "--include-module=imagecodecs._ccitt",
    "--include-module=imagecodecs._deflate",
    "--include-module=imagecodecs._zlib",
    "--include-module=imagecodecs._lzma",
    "--include-module=imagecodecs._zstd",
    "--include-module=imagecodecs._webp",
    "--include-module=imagecodecs._png",
    "--include-module=imagecodecs._lerc",
    "--include-module=imagecodecs._jpegxr",
    "--include-module=imagecodecs._jpegxl",
]

# scikit-image resolves its submodules through lazy_loader -- skimage/__init__.py
# defines a module-level __getattr__ rather than importing them -- so
# `from skimage import draw` is a *runtime* lookup that Nuitka's static import
# following does not see. Every submodule the package touches has to be named
# here or it is simply absent from the build, and the failure only shows up
# when a command reaches for it at runtime on the user's machine.
#
# draw: lavlab/roi.py and lavlab/tiling.py (polygon rasterization).
# color, filters, measure, morphology: lavlab/tiling.py (tissue mask and the
# background margin).
SKIMAGE_NUITKA_FLAGS = [
    "--include-module=skimage.color",
    "--include-module=skimage.draw",
    "--include-module=skimage.filters",
    "--include-module=skimage.measure",
    "--include-module=skimage.morphology",
]


def native_target() -> str:
    """Validate that this host can build one of the two supported targets."""
    machine = platform.machine().lower()
    machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(machine, machine)
    if sys.platform == "darwin" and machine == "arm64":
        detected = "macos-arm64"
    elif sys.platform.startswith("linux") and machine == "x86_64":
        detected = "linux-x86_64"
    else:
        raise RuntimeError(
            f"Unsupported build host: {sys.platform}/{platform.machine()}. "
            "Build only on macOS arm64 or Linux x86_64."
        )

    requested = os.environ.get("LAVLAB_TARGET", detected)
    if requested != detected:
        raise RuntimeError(
            f"LAVLAB_TARGET={requested!r} does not match this build host "
            f"({detected}); cross-compilation is not supported."
        )
    return detected


def normalize_binary_name(output_dir: Path) -> Path:
    """Give Nuitka's onefile executable the stable launcher name."""
    binary = output_dir / "lavlab-bin"
    if binary.is_file():
        return binary

    suffixed_binary = output_dir / "lavlab-bin.bin"
    if suffixed_binary.is_file():
        suffixed_binary.rename(binary)
        return binary

    raise RuntimeError(
        f"Nuitka did not produce the expected onefile executable in {output_dir} "
        "(lavlab-bin or lavlab-bin.bin)."
    )


def omero_ice_modules() -> list[str]:
    """Find generated OMERO Ice modules that IceImport loads dynamically."""
    modules = set()
    for search_path in map(Path, sys.path):
        if not search_path.is_dir():
            continue
        modules.update(path.stem for path in search_path.glob("*_ice.py"))
    return sorted(modules)


def build_onefile(project_dir: Path, output_dir: Path) -> Path:
    """Compile the application to one self-contained Nuitka executable."""
    native_target()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The onefile binary must be compiled with Python 3.12.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        output_dir / "lavlab-bin",
        output_dir / "lavlab-bin.bin",
        output_dir / "lavlab.dist",
        output_dir / "lavlab.onefile-build",
        output_dir / "__main__.dist",
        output_dir / "__main__.onefile-build",
    ):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()

    # Keep LTO enabled for the compact onefile binary, but cap macOS
    # parallelism to avoid multiplying peak linker memory use.
    jobs = max(1, (os.cpu_count() or 2) - 1)
    if sys.platform == "darwin":
        jobs = min(jobs, 2)

    command = [
        sys.executable,
        "-m",
        "nuitka",
        "--onefile",
        "--assume-yes-for-downloads",
        f"--output-dir={output_dir}",
        "--output-filename=lavlab-bin",
        "--include-package=lavlab",
        "--include-package-data=lavlab",
        "--include-package-data=highdicom",
        f"--user-plugin={project_dir / 'nuitka_plugin.py'}",
        "--noinclude-pytest-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        "--noinclude-custom-mode=unittest:error",
        "--nofollow-import-to=unittest",
        "--nofollow-import-to=tkinter",
        "--nofollow-import-to=matplotlib",
        "--nofollow-import-to=IPython",
        "--lto=yes",
        f"--jobs={jobs}",
        *PYDICOM_NUITKA_FLAGS,
        *IMAGECODECS_NUITKA_FLAGS,
        *SKIMAGE_NUITKA_FLAGS,
        *[f"--include-module={name}" for name in omero_ice_modules()],
        str(project_dir / "src" / "lavlab" / "__main__.py"),
    ]

    extra_args = os.environ.get("LAVLAB_NUITKA_ARGS", "")
    if extra_args:
        command[3:3] = extra_args.split()

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(project_dir / "src"), env.get("PYTHONPATH")])
    )
    print(f"Including {len(omero_ice_modules())} generated OMERO Ice modules.")
    subprocess.run(command, check=True, env=env)
    return normalize_binary_name(output_dir)


def main() -> None:
    project_dir = Path(__file__).parent
    output_dir = Path(os.environ.get("LAVLAB_OUTPUT_DIR", project_dir / "native"))
    staged_dist = project_dir / "src" / "lavlab" / "bin" / "dist"
    if staged_dist.exists():
        shutil.rmtree(staged_dist)
    binary = build_onefile(project_dir, output_dir)
    print(f"Built onefile executable: {binary}")


if __name__ == "__main__":
    main()
