# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""``lavlab roi`` -- pull ROI mask images (RGB color mask or single-channel
palette) from OMERO."""

from __future__ import annotations

import argparse
import contextlib
import logging
import multiprocessing
import os

from lavlab.commands._shared import (
    add_common_output_args,
    add_creds_args,
    connect_from_args,
    ensure_parent_dir,
    group_of,
    load_fs_map_from_args,
    make_temp_path,
    parse_target,
)
from lavlab.config import ConfigError
from lavlab.naming import resolve_output_path

# pyvips, lavlab.imaging and lavlab.roi (numpy, skimage, omero) are imported
# inside the functions that use them, so that building the argument parser --
# and therefore --help -- stays pure Python.

log = logging.getLogger(__name__)

DEFAULT_SUFFIX = "_annot"


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("roi", help="Pull ROI mask images from OMERO.")
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
        "-a",
        "--all",
        action="store_true",
        help="Include every annotation, regardless of textValue.",
    )
    parser.add_argument(
        "-t",
        "--text-filter",
        type=str.lower,
        action="append",
        default=[],
        help="Whitelist of textValue annotations to include (repeatable).",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help=f"Filename suffix (default: '{DEFAULT_SUFFIX}').",
    )
    parser.add_argument(
        "--palette",
        action="store_true",
        help="Emit a single-channel label mask ordered by --text-filter instead of an RGB color mask.",
    )
    parser.add_argument(
        "--format",
        choices=["jp2", "jpg", "jpeg", "png", "tif", "tiff"],
        default="jp2",
        help="Output format when the filename isn't fixed by an explicit -o path "
        "(default: jp2). Not compatible with --palette, since jpg/jpeg's lossy "
        "compression would corrupt exact label values.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Also attach the rendered mask to the image in OMERO, under the "
        "LargeRecon.<downsample>.roi namespace (the same one legacy batch_roi.py "
        "used). Off by default -- roi otherwise only ever reads from OMERO.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any image that already has this exact mask attached in OMERO "
        "(same downsample, suffix and format), without re-rendering it. Useful "
        "for backfilling a whole group.",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Don't keep a local copy -- render to a temporary file, upload it, then "
        "delete it. Requires --upload (otherwise the mask would be rendered and "
        "thrown away) and is incompatible with -o.",
    )
    add_common_output_args(parser)
    add_creds_args(parser)
    parser.set_defaults(handler=run)


def _is_lossy_format(args: argparse.Namespace) -> bool:
    # An explicit -o file extension overrides --format (same precedence
    # write_recon() itself uses), so check that first.
    if args.output and not os.path.isdir(args.output):
        ext = os.path.splitext(args.output)[1].lstrip(".").lower()
        if ext:
            return ext in ("jpg", "jpeg")
    return args.format in ("jpg", "jpeg")


def _validate_selection(args: argparse.Namespace) -> None:
    if args.all and args.palette:
        raise SystemExit(
            "error: --all is not supported with --palette; manually define --text-filter values instead."
        )
    if args.palette and _is_lossy_format(args):
        raise SystemExit(
            "error: --palette produces exact label values (0, 1, 2, ...) that jpg/jpeg's "
            "lossy compression would corrupt; use --format jp2/png/tif (or an -o path with "
            "one of those extensions) instead."
        )
    if not args.all and not args.text_filter:
        raise SystemExit("error: specify --all or at least one --text-filter value.")
    if args.skip_local and not args.upload:
        raise SystemExit(
            "error: --skip-local without --upload would render a mask and then throw "
            "it away; add --upload, or drop --skip-local."
        )
    if args.skip_local and args.output:
        raise SystemExit(
            "error: --skip-local doesn't write anywhere, so -o has nothing to do; "
            "drop one or the other."
        )


def run(args: argparse.Namespace) -> None:
    _validate_selection(args)
    target = parse_target(args.target)
    if target == "batch":
        _run_batch(args)
    else:
        _run_single(args, target)


def _render_and_write(
    conn, image, args: argparse.Namespace, output_path: str
) -> int | None:
    """Render and write the ROI mask, uploading it if --upload was given;
    returns how many shapes were rendered, or None if nothing matched the
    given filters."""
    import pyvips as pv

    from lavlab.imaging import write_recon
    from lavlab.roi import get_roi_mask, upload_mask

    mask, shape_count = get_roi_mask(
        image,
        args.downsample,
        include_all=args.all,
        text_filter=args.text_filter,
        palette=args.palette,
    )
    if mask.size == 0:
        log.info("Image %d: no ROIs matched the given filters.", image.getId())
        return None
    ensure_parent_dir(output_path)
    roi_img = pv.Image.new_from_array(mask)
    del mask
    write_recon(roi_img, output_path, lossless=True)

    if args.upload:
        # A failed upload shouldn't throw away a mask that rendered fine --
        # same call the lr uploader makes, same reasoning.
        try:
            remote_name = upload_mask(
                conn, image, output_path, args.downsample, args.suffix, args.format
            )
        except Exception:
            log.warning(
                "Image %d: mask written to '%s' but upload to OMERO failed.",
                image.getId(),
                output_path,
                exc_info=True,
            )
        else:
            log.info("Image %d: uploaded mask as %s.", image.getId(), remote_name)

    return shape_count


def _run_single(args: argparse.Namespace, image_id: int) -> None:
    from lavlab.roi import has_uploaded_mask

    conn = connect_from_args(args)
    try:
        image = conn.getObject("Image", image_id)
        if image is None:
            raise SystemExit(f"error: image {image_id} not found.")

        # Cheap existence check first -- listing annotations is a read, so it
        # works before the group switch below, and skipping here avoids
        # rasterizing a mask nobody is going to use.
        if args.skip_existing and has_uploaded_mask(
            image, args.downsample, args.suffix, args.format
        ):
            print(f"Image {image_id}: mask already attached in OMERO, skipping.")
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
                    suffix=args.suffix,
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
            shape_count = _render_and_write(conn, image, args, output_path)
            if shape_count is None:
                raise SystemExit(
                    f"error: no ROIs found for image {image_id} under the given filters."
                )
            shown = (
                "uploaded to OMERO (not stored locally)"
                if args.skip_local
                else output_path
            )
            print(
                f"Completed ROI for image {image_id}: {shown} ({shape_count} shape(s))"
            )
        finally:
            if args.skip_local:
                with contextlib.suppress(OSError):
                    os.remove(output_path)
    finally:
        conn.close()


_WORKER_STATE: dict = {}


def _init_worker(args: argparse.Namespace, fs_map) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "conn": connect_from_args(args),
        "args": args,
        "fs_map": fs_map,
    }


def _process_one(image_id: int):
    from lavlab.omero_client import is_conn_error
    from lavlab.roi import has_uploaded_mask

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

            if args.skip_existing and has_uploaded_mask(
                image, args.downsample, args.suffix, args.format
            ):
                print(f"Image {image_id}: mask already attached in OMERO, skipping.")
                return (image_id, None)

            # Always switch the connection into the image's own group --
            # args.group only scopes which images iter_image_ids() listed,
            # it doesn't substitute for group_of()'s side effect of setting
            # SERVICE_OPTS. This matters directly now that --upload writes:
            # reads tolerate the dummy group (-1), writes don't. See the
            # identical fix/comment in lavlab/commands/lr.py.
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
                    suffix=args.suffix,
                    ext=args.format,
                    batch=True,
                )
                if output_path is None:
                    log.warning(
                        "Image %d: no usable output directory, skipping.", image_id
                    )
                    return None

                if os.path.exists(output_path) and not args.override:
                    return (image_id, output_path)

            try:
                shape_count = _render_and_write(conn, image, args, output_path)
                if shape_count is None:
                    return None

                shown = (
                    "uploaded to OMERO (not stored locally)"
                    if args.skip_local
                    else output_path
                )
                print(
                    f"Completed ROI for image {image_id}: {shown} ({shape_count} shape(s))"
                )
                return (image_id, None if args.skip_local else output_path)
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


def _run_batch(args: argparse.Namespace) -> None:
    # Importing omero_client is not Ice *usage*, and this is no earlier than
    # the module-level import it replaced, so it does not affect the
    # fork-before-Ice ordering described below.
    from lavlab.omero_client import iter_image_ids

    # Fork the worker pool before this (parent) process ever touches
    # OMERO/Ice -- see the identical comment in lavlab/commands/lr.py's
    # _run_batch for the confirmed root cause (fork() after any Ice usage
    # in the forking process, even a closed connection, breaks every
    # child's own subsequent Ice usage -- source retries-and-fails,
    # compiled Nuitka onefile segfaults). Fork first, while this process
    # is still Ice-virgin; the parent's own connection below (to list
    # image_ids) and each worker's _init_worker() connection (post-fork,
    # in-process) are both then a process's first-ever Ice usage -- the
    # confirmed-safe case.
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

    completed = [r for r in results if r is not None]
    print(f"Batch complete: {len(completed)}/{len(image_ids)} images processed.")
