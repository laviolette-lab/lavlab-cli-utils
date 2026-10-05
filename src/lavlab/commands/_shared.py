# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Shared helpers for the ``lr`` and ``roi`` command groups."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile

from lavlab.config import ConfigError, load_fs_map, resolve_creds

# lavlab.omero_client pulls in omero-py (and through it Ice). It is imported
# inside the functions that need a connection so that building the argument
# parser -- and therefore --help -- stays pure Python.

log = logging.getLogger(__name__)


def make_temp_path(fmt: str) -> str:
    """Create an empty temp file with a *fmt* extension and return its path.

    Used by ``--skip-local``: OMERO's upload API reads from a real path
    rather than raw bytes, so "don't keep a local copy" still needs a file
    to exist for the duration of the upload. The caller is responsible for
    deleting it afterward.

    :param fmt: file extension, without the dot
    :type fmt: str
    :return: path to a new empty file
    :rtype: str
    """
    fd, path = tempfile.mkstemp(suffix=f".{fmt}")
    os.close(fd)
    return path


def add_creds_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-u", "--user", help="OMERO username (or set OMERO_USER).")
    parser.add_argument(
        "-w", "--password", help="OMERO password (or set OMERO_PASSWORD)."
    )
    parser.add_argument("-s", "--host", help="OMERO server host (or set OMERO_HOST).")
    parser.add_argument(
        "-p", "--port", type=int, help="OMERO server port (or set OMERO_PORT)."
    )


def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--downsample", type=int, default=10, help="Downsample factor (default: 10)."
    )
    parser.add_argument(
        "--fs-map", dest="fs_map", help="Path to a custom fs_map YAML file."
    )


def connect_from_args(args: argparse.Namespace):
    from lavlab.omero_client import connect

    try:
        creds = resolve_creds(args.user, args.password, args.host, args.port)
    except ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None
    return connect(creds)


def load_fs_map_from_args(args: argparse.Namespace):
    try:
        return load_fs_map(args.fs_map)
    except ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None


def parse_target(target: str) -> int | str:
    """Return the image id as an int, or 'batch' verbatim."""
    if target == "batch":
        return "batch"
    try:
        return int(target)
    except ValueError:
        raise SystemExit(
            f"error: expected an image ID or 'batch', got '{target}'"
        ) from None


def group_of(conn, obj) -> int:
    from lavlab.omero_client import switch_to_object_group

    switch_to_object_group(conn, obj)
    return obj.details.group.id.val


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
