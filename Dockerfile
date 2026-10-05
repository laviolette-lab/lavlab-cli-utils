# Reproducible build environment for lavlab-cli-utils' Nuitka wheel.
#
# This image is NOT a runtime image -- it does not contain the compiled
# lavlab binary. It's a pinned, disposable environment with the C compiler,
# Nuitka, and every runtime dependency (numpy, pyvips, omero-py, highdicom,
# SimpleITK, ...) pre-installed, so the Nuitka/Ice build doesn't depend on
# whatever happens to be installed on a given laptop. The actual compile
# runs against your live source tree, mounted in at `docker run` time --
# see usage below.
#
# ---------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------
#
#   1. Get the lab's Ice wheel for linux-x86_64 (the platform this image
#      builds for) and place it in ./vendor/, e.g.:
#        mkdir -p vendor
#        curl -L -o vendor/zeroc_ice-<version>-<platform>.whl <lab-internal-URL>
#
#   2. Build the image once (slow -- installs the whole toolchain):
#        docker build -t lavlab-builder .
#
#   3. Run it against your current source tree to actually build the wheel
#      (fast to re-run after edits -- the image itself doesn't change):
#        docker run --rm -v "$(pwd)":/src -v "$(pwd)/dist":/out lavlab-builder
#
#      The resulting wheel(s) land in ./dist/ on your host, same as running
#      `python setup.py bdist_wheel` directly would -- this just guarantees
#      the environment it ran in.
#
# The project source is intentionally NOT baked into the image (no `COPY .
# .` here) -- it's mounted at run time so you don't have to rebuild the
# image after every source change, only after a dependency change.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# --- ZeroC/Glencoe Ice wheel omero-py needs -----------------------------
# This cannot be fetched from PyPI or baked into this Dockerfile directly:
# it comes from a lab-internal URL (possibly access-restricted), not a
# public package index. Place it in ./vendor/ before building this image
# -- see the usage note at the top of this file. vendor/.gitkeep exists so
# this COPY never fails even if you haven't added a wheel yet; the RUN
# step below fails loudly (not silently) if none is present.
COPY vendor/ /tmp/vendor/
RUN if ls /tmp/vendor/*.whl >/dev/null 2>&1; then \
        pip install --no-cache-dir /tmp/vendor/*.whl; \
    else \
        echo "ERROR: no *.whl found in ./vendor/ -- see the comment at the" >&2; \
        echo "top of this Dockerfile for how to get the Ice wheel." >&2; \
        exit 1; \
    fi

# --- everything else the build needs ------------------------------------
COPY build-requirements.txt .
RUN pip install --no-cache-dir -r build-requirements.txt

# Source is mounted at /src at `docker run` time, not copied in here.
VOLUME ["/src", "/out"]
WORKDIR /src

# Not `pip wheel .` -- that builds in an isolated PEP 517 environment that
# can't see the Ice wheel installed above (it would try to rebuild
# zeroc-ice from source) and fails with "ModuleNotFoundError: No module
# named 'build_native'" since the repo root isn't on sys.path under pip's
# build hooks. `setup.py bdist_wheel` uses this image's environment
# directly, the same way a non-Docker build must (see the README's
# "Building the compiled wheel" section).
ENTRYPOINT ["python", "setup.py", "bdist_wheel", "--dist-dir", "/out"]
