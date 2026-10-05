# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""``lavlab lr`` -- pull large-recon (downsampled) images from OMERO."""

from __future__ import annotations

import argparse
import contextlib
import logging
import multiprocessing
import os
from collections import Counter

from lavlab.commands._shared import (
    add_common_output_args,
    add_creds_args,
    connect_from_args,
    group_of,
    load_fs_map_from_args,
    make_temp_path,
    parse_target,
)
from lavlab.config import ConfigError
from lavlab.naming import resolve_output_path

# lavlab.large_recon (pyvips, tifffile, numpy) and lavlab.omero_client
# (omero-py, Ice) are imported inside the functions that use them, so that
# building the argument parser -- and therefore --help -- stays pure Python.

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "lr", help="Pull large-recon downsampled images from OMERO."
    )
    parser.add_argument(
        "target", help="An OMERO image ID, or 'batch' to pull all images."
    )
    parser.add_argument(
        "-o", "--output", help="Output file (single) or directory (batch)."
    )
    parser.add_argument(
        "-g", "--group", type=int, help="OMERO group ID (batch mode only)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for batch mode (default: 8).",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Ignore any existing OMERO large-recon annotation and regenerate fresh, "
        "re-uploading the result (combine with --override to also force "
        "regeneration when the local output file already exists).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Don't upload the generated large-recon back to OMERO as an annotation.",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Don't keep a local copy -- fetch/generate to a temporary file, upload it to "
        "OMERO, then delete it. Incompatible with -o and with --skip-upload (that "
        "combination would do nothing).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any image that already has a large-recon annotation for this "
        "downsample and format, without downloading or regenerating it. Useful for "
        "backfilling a whole group: cached images cost one cheap existence check, "
        "only the missing ones do real work. Incompatible with --regenerate.",
    )
    parser.add_argument(
        "--max-failed",
        type=int,
        default=None,
        metavar="N",
        help="Batch mode: exit non-zero if more than N images failed. By default "
        "a batch exits non-zero only when *every* image failed, since one "
        "bad slide in a few thousand shouldn't fail a scheduled run. Set "
        "this to make the tolerance explicit (e.g. --max-failed 0 to fail "
        "on any error at all).",
    )
    parser.add_argument(
        "--format",
        choices=["jp2", "jpg", "jpeg", "png", "tif", "tiff"],
        default="jp2",
        help="Output format when the filename isn't fixed by an explicit -o path "
        "(default: jp2). Also selects which format is searched for/uploaded as "
        "an OMERO large-recon annotation.",
    )
    add_common_output_args(parser)
    add_creds_args(parser)
    parser.set_defaults(handler=run)


def _validate_args(args: argparse.Namespace) -> None:
    if args.skip_existing and args.regenerate:
        raise SystemExit(
            "error: --skip-existing and --regenerate are contradictory -- one skips "
            "images that already have a recon, the other regenerates them regardless."
        )
    if args.skip_local and args.skip_upload:
        raise SystemExit(
            "error: --skip-local and --skip-upload together would fetch/generate an "
            "image and then discard it -- drop one or the other."
        )
    if args.skip_local and args.output:
        raise SystemExit(
            "error: --skip-local doesn't write anywhere, so -o has nothing to do; "
            "drop one or the other."
        )
    if args.max_failed is not None:
        if args.target != "batch":
            raise SystemExit(
                "error: --max-failed is a batch-mode tolerance; a single image "
                "already exits non-zero when it fails."
            )
        if args.max_failed < 0:
            raise SystemExit("error: --max-failed must be zero or greater.")


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    target = parse_target(args.target)
    if target == "batch":
        _run_batch(args)
    else:
        _run_single(args, target)


def _render_and_write(conn, image, args: argparse.Namespace, output_path: str) -> str:
    from lavlab.large_recon import fetch_large_recon

    return fetch_large_recon(
        conn,
        image,
        args.downsample,
        output_path,
        regenerate=args.regenerate,
        skip_upload=args.skip_upload,
    )


def _run_single(args: argparse.Namespace, image_id: int) -> None:
    from lavlab.large_recon import has_cached_recon

    conn = connect_from_args(args)
    try:
        image = conn.getObject("Image", image_id)
        if image is None:
            raise SystemExit(f"error: image {image_id} not found.")
        if args.skip_existing and has_cached_recon(image, args.downsample, args.format):
            print(f"Image {image_id}: already has a large-recon, skipping.")
            return

        group_id = group_of(conn, image)
        name = image.getName()

        if args.skip_local:
            output_path = make_temp_path(args.format)
        else:
            fs_map = load_fs_map_from_args(args)
            try:
                output_path = resolve_output_path(
                    args.output,
                    fs_map,
                    group_id,
                    name,
                    args.downsample,
                    ext=args.format,
                    batch=False,
                )
            except ConfigError as exc:
                raise SystemExit(f"error: {exc}") from None

            if os.path.exists(output_path) and not args.override:
                print(
                    f"Already exists, skipping (use --override to replace): {output_path}"
                )
                return

        try:
            _render_and_write(conn, image, args, output_path)
            shown = (
                "uploaded to OMERO (not stored locally)"
                if args.skip_local
                else output_path
            )
            print(f"Completed image {image_id}: {shown}")
        finally:
            if args.skip_local:
                with contextlib.suppress(OSError):
                    os.remove(output_path)
    finally:
        conn.close()


# Per-worker state, populated by the pool initializer since Pool.imap only
# forwards a single positional argument to the worker function.
_WORKER_STATE: dict = {}


def _init_worker(args: argparse.Namespace, fs_map) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "conn": connect_from_args(args),
        "args": args,
        "fs_map": fs_map,
    }


def _process_one(image_id: int):
    from lavlab.large_recon import has_cached_recon
    from lavlab.omero_client import is_conn_error

    args = _WORKER_STATE["args"]
    fs_map = _WORKER_STATE["fs_map"]
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            conn = _WORKER_STATE["conn"]
            if not conn.isConnected():
                conn = connect_from_args(args)
                _WORKER_STATE["conn"] = conn

            image = conn.getObject("Image", image_id)
            if image is None:
                log.warning("Image %d not found, skipping.", image_id)
                return None

            if args.skip_existing and has_cached_recon(
                image, args.downsample, args.format
            ):
                print(f"Image {image_id}: already has a large-recon, skipping.")
                return (image_id, None, None)

            group_id = group_of(conn, image)
            name = image.getName()

            if args.skip_local:
                output_path = make_temp_path(args.format)
            else:
                output_path = resolve_output_path(
                    args.output,
                    fs_map,
                    group_id,
                    name,
                    args.downsample,
                    ext=args.format,
                    batch=True,
                )
                if output_path is None:
                    log.warning(
                        "Image %d: no usable output directory, skipping.", image_id
                    )
                    return None

                if os.path.exists(output_path) and not args.override:
                    return (image_id, output_path, None)

            try:
                tier = _render_and_write(conn, image, args, output_path)
                shown = (
                    "uploaded to OMERO (not stored locally)"
                    if args.skip_local
                    else output_path
                )
                print(f"Completed image {image_id}: {shown}")
                return (image_id, None if args.skip_local else output_path, tier)
            finally:
                if args.skip_local:
                    with contextlib.suppress(OSError):
                        os.remove(output_path)
        except ConfigError:
            raise
        except Exception as exc:
            if is_conn_error(exc) and attempt < max_attempts:
                log.warning("Image %d: connection error, retrying: %s", image_id, exc)
                _WORKER_STATE["conn"] = connect_from_args(args)
                continue
            log.exception("Image %d: unhandled error.", image_id)
            return None
    return None


def _summarise_batch(image_ids, results, max_failed) -> None:
    """Report what the batch achieved, and fail the run if it achieved nothing.

    Split out from ``_run_batch`` so the reporting and exit-code rules can be
    exercised without standing up a worker pool and an OMERO connection.

    *results* holds one entry per image: ``None`` for a failure, otherwise
    ``(image_id, output_path, tier)`` where *tier* is the tier that actually
    served the image, or ``None`` if it was skipped without doing work.
    """
    completed = [r for r in results if r is not None]
    failed = len(image_ids) - len(completed)

    # Which tier actually served each image. Logged at INFO -- the default
    # level, since -v only raises it to DEBUG -- so it lands in `kubectl
    # logs` with a timestamp like every other run-level message. A run
    # reporting success with network=<everything> is a run whose local
    # source silently stopped working; without this line it looks identical
    # to a healthy one.
    tiers = Counter(r[2] for r in completed if r[2] is not None)
    skipped = sum(1 for r in completed if r[2] is None)

    log.info(
        "Batch complete: %d/%d images processed, %d failed.",
        len(completed),
        len(image_ids),
        failed,
    )
    if tiers or skipped:
        parts = [f"{tier}={count}" for tier, count in sorted(tiers.items())]
        if skipped:
            parts.append(f"skipped={skipped}")
        log.info("Served by tier: %s", ", ".join(parts))

    # The exit code carries the outcome, so a scheduled run that accomplished
    # nothing shows up as a failed Job rather than a green one.
    if image_ids and not completed:
        raise SystemExit(
            f"error: all {len(image_ids)} images failed; see the per-image "
            "errors above."
        )
    if max_failed is not None and failed > max_failed:
        raise SystemExit(
            f"error: {failed} of {len(image_ids)} images failed, over the "
            f"--max-failed {max_failed} threshold."
        )


def _run_batch(args: argparse.Namespace) -> None:
    # Importing omero_client is not Ice *usage*, and this is no earlier than
    # the module-level import it replaced, so the fork-before-Ice ordering
    # this function depends on is unchanged.
    from lavlab.omero_client import iter_image_ids

    fs_map = load_fs_map_from_args(args)
    log.info("Starting %d workers.", args.workers)

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(
        args.workers, initializer=_init_worker, initargs=(args, fs_map)
    ) as pool:
        conn = connect_from_args(args)
        try:
            image_ids = list(iter_image_ids(conn, args.group))
        finally:
            conn.close()

        log.info("Found %d images.", len(image_ids))
        results = list(pool.imap_unordered(_process_one, image_ids))

    _summarise_batch(image_ids, results, args.max_failed)
