# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""``lavlab`` CLI entry point.

Entry point for the Nuitka --standalone build (see ``build_native.py``).

Building the parser must stay pure Python: the command modules import
numpy/pyvips/omero/SimpleITK inside their handlers rather than at module
scope, so that ``--help`` cannot be broken by a problem in a native
dependency the chosen subcommand never touches.
"""

from __future__ import annotations

import argparse
import logging
import sys

from lavlab.commands import geojson, lr, meta, roi_cmd, seg, tile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lavlab", description="LAVLab OMERO CLI utilities."
    )
    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Write over existing output files (default: False).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (very chatty -- logs a repr of every OMERO tile "
        "response, which measurably slows a network large-recon fetch; leave "
        "off unless you're diagnosing something).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    lr.add_parser(subparsers)
    roi_cmd.add_parser(subparsers)
    tile.add_parser(subparsers)
    meta.add_parser(subparsers)
    geojson.add_parser(subparsers)
    seg.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args.handler(args)


if __name__ == "__main__":
    main()
