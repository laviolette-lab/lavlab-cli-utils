# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Build the standalone Nuitka executable with OMERO's dynamic Ice modules."""

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


def normalize_binary_name(dist_dir: Path) -> Path:
    """Give Nuitka's platform-suffixed executable the stable launcher name."""
    binary = dist_dir / "lavlab-bin"
    if binary.is_file():
        return binary

    suffixed_binary = dist_dir / "lavlab-bin.bin"
    if suffixed_binary.is_file():
        suffixed_binary.rename(binary)
        return binary

    raise RuntimeError(
        f"Nuitka did not produce the expected executable in {dist_dir} "
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


def restore_unpatched_vips(dist_dir: Path) -> None:
    """Undo Nuitka's RPATH rewrite of pyvips' bundled libvips.

    Nuitka runs ``patchelf --set-rpath '$ORIGIN'`` over every shared object it
    bundles. On Linux that rewrite corrupts pyvips' statically linked libvips:
    the patched library segfaults inside its ELF constructor as soon as
    anything dlopens it, which takes down every command that imports pyvips.
    Copying the pristine wheel copy back over Nuitka's patched one fixes it.
    Dropping the RPATH costs nothing: libvips links only against system
    libraries (libc, libstdc++, libm, libdl, libpthread, libgcc_s, libresolv),
    so it has nothing to resolve out of the dist directory in the first place.

    macOS is unaffected. Nuitka rewrites ``libvips.42.dylib`` there too, but
    the Mach-O rewrite produces a library that still loads, so this is a
    deliberate no-op on Darwin rather than an unhandled platform.
    """
    if sys.platform == "darwin":
        print("Skipping libvips restore: macOS rewrite is not corrupting.")
        return

    patched = sorted(dist_dir.glob("libvips*.so.*"))
    if not patched:
        return

    # pyvips[binary] has shipped its libraries under both names.
    pristine = [
        path
        for search_path in map(Path, sys.path)
        if search_path.is_dir()
        for libs_dir in ("pyvips_binary.libs", "pyvips.libs")
        for path in search_path.glob(f"{libs_dir}/libvips*.so.*")
    ]
    if not pristine:
        # Failing loudly matters here: silently shipping the patched library is
        # exactly the bug this function exists to prevent, and a corrupt
        # libvips only shows up as a crash at runtime on the user's machine.
        raise RuntimeError(
            f"Nuitka bundled {patched[0].name} but no pristine copy was found in "
            "pyvips_binary.libs/ or pyvips.libs/ to restore it from. Refusing to "
            "ship a libvips that patchelf may have corrupted."
        )

    by_name = {path.name: path for path in pristine}
    for target in patched:
        source = by_name.get(target.name, pristine[0])
        shutil.copy2(source, target)
        print(f"Restored unpatched {target.name} from {source}")


def main() -> None:
    native_target()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The standalone binary must be compiled with Python 3.12.")

    project_dir = Path(__file__).parent
    output_dir = Path(os.environ.get("LAVLAB_OUTPUT_DIR", project_dir / "native"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # A previous `setup.py bdist_wheel` stages its dist under the package, and
    # --include-package-data=lavlab would sweep it into this build (Nuitka then
    # dies resolving the stale copy's dylibs). setup.py clears it the same way.
    staged_dist = project_dir / "src" / "lavlab" / "bin" / "dist"
    if staged_dist.exists():
        shutil.rmtree(staged_dist)

    command = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        f"--output-dir={output_dir}",
        "--output-filename=lavlab-bin",
        "--include-package=lavlab",
        "--include-package-data=lavlab",
        "--noinclude-pytest-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        "--noinclude-custom-mode=unittest:error",
        "--nofollow-import-to=unittest",
        "--nofollow-import-to=tkinter",
        "--nofollow-import-to=matplotlib",
        "--nofollow-import-to=IPython",
        "--lto=yes",
        f"--jobs={max(1, (os.cpu_count() or 2) - 1)}",
        *PYDICOM_NUITKA_FLAGS,
        *IMAGECODECS_NUITKA_FLAGS,
        *SKIMAGE_NUITKA_FLAGS,
        *[f"--include-module={name}" for name in omero_ice_modules()],
        str(project_dir / "src" / "lavlab" / "__main__.py"),
    ]

    extra_args = os.environ.get("LAVLAB_NUITKA_ARGS", "")
    if extra_args:
        command[3:3] = extra_args.split()

    # src layout: --include-package=lavlab resolves through sys.path, and
    # the repo root (the cwd) no longer contains the package.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(project_dir / "src"), env.get("PYTHONPATH")])
    )

    print(f"Including {len(omero_ice_modules())} generated OMERO Ice modules.")
    subprocess.run(command, check=True, env=env)
    dist_dir = output_dir / "__main__.dist"
    normalize_binary_name(dist_dir)
    restore_unpatched_vips(dist_dir)


if __name__ == "__main__":
    main()