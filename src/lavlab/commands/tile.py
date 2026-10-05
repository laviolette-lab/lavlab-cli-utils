# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""``lavlab tile`` -- cut whole-slide images into fixed-size training tiles."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import random
import time
from collections import Counter

from lavlab.commands._shared import (
    add_creds_args,
    connect_from_args,
    group_of,
    parse_target,
)

log = logging.getLogger(__name__)

DEFAULT_MPP = 0.5
DEFAULT_SIZE = 224
DEFAULT_MIN_COVERAGE = 0.5
DEFAULT_TISSUE_THRESH = 0.5
DEFAULT_BACKGROUND_LABEL = "benign"
DEFAULT_BACKGROUND_MARGIN = 200
DEFAULT_MAX_BACKGROUND_TILES = 2000
DEFAULT_EXCLUDE_TEXT = ("exclusion roi",)
DEFAULT_JPEG_QUALITY = 90
DEFAULT_CONNECTIONS = 4

EXAMPLES = """\
examples:
  # How many tiles would I get? Counts per class and per ROI, writes nothing.
  lavlab tile 12345 --roi --all --dry-run -o ./tiles

  # Training tiles from the Gleason classes, 512 px at 0.5 um/px.
  lavlab tile 12345 --roi -t G3 -t G4cg -t G4fg -t G5 --size 512 -o ./tiles

  # A whole group, 8 workers, resumable, one combined manifest for the dataset.
  lavlab tile batch -g 3 --roi --all --size 512 --skip-existing \\
      --manifest ./tiles/dataset.csv -o ./tiles

  # Two concentric scales per tile (detail + context), as JPEG.
  lavlab tile batch -g 3 --roi --all --size 512 --scales 0.5,1.0 \\
      --format jpg -o ./tiles

  # Every tissue tile, half-overlapping, for whole-slide inference.
  lavlab tile 12345 --whole --size 512 --stride 256 -o ./inference
"""


def _parse_scales(text: str) -> list[float]:
    try:
        values = [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated micrometres per pixel, e.g. 0.5,1.0; got '{text}'"
        ) from None
    if not values:
        raise argparse.ArgumentTypeError("expected at least one scale")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"every scale must be positive, got '{text}'")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"scales repeat a value: '{text}'")
    return sorted(values)


def _parse_erode(text: str) -> float | str:
    if text.strip().lower() == "auto":
        return "auto"
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected micrometres (e.g. 128), 0 to turn it off, or 'auto'; got '{text}'"
        ) from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"--erode must be 0 or more, got {value}")
    return value


def _format(text: str) -> str:
    return "jpg" if text.lower() == "jpeg" else text.lower()


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "tile",
        help="Cut whole-slide images into fixed-size tiles for training.",
        description="Cut whole-slide images into fixed-size tiles for training "
        "or inference. Start with --dry-run to see what a run would produce.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target", help="An OMERO image ID, or 'batch' to tile a whole group."
    )
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        metavar="DIR",
        help="Output root directory (required).",
    )

    what = parser.add_argument_group("what to tile")
    mode = what.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--whole",
        action="store_true",
        help="Tile the whole slide (tissue only), under a flat 'whole/' label. "
        "For inference or unannotated slides -- its output is never "
        "treated as a benign class.",
    )
    mode.add_argument(
        "--roi",
        action="store_true",
        help="Tile only inside ROIs, labelling each tile by the annotation "
        "it falls in.",
    )
    what.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="--roi: include every annotation regardless of textValue.",
    )
    what.add_argument(
        "-t",
        "--text-filter",
        type=str.lower,
        action="append",
        default=[],
        help="--roi: whitelist a textValue or class folder name "
        "(repeatable, case-insensitive). Mutually exclusive with --all.",
    )
    what.add_argument(
        "--labels",
        metavar="PATH",
        help="Custom textValue -> class folder YAML (default: the bundled "
        "lavlab/data/default_tile_labels.yaml).",
    )

    geometry = parser.add_argument_group("tile size and scale")
    scale = geometry.add_mutually_exclusive_group()
    scale.add_argument(
        "--mpp",
        type=float,
        default=None,
        help=f"Target micrometres per pixel (default: {DEFAULT_MPP}). Read "
        "against the image's own pixel size to pick a pyramid level "
        "and resize factor. Errors out if the image has no pixel "
        "size recorded -- use --downsample instead.",
    )
    scale.add_argument(
        "--scales",
        type=_parse_scales,
        default=None,
        metavar="MPP,MPP",
        help="Write concentric crops at several scales in one pass, e.g. "
        "0.5,1.0: the same centre at 0.5 um/px (detail) and 1.0 um/px "
        "(context). The finest scale decides the grid and labels. Files "
        "are named ..._x<cx>_y<cy>_mpp<scale>.<fmt> so pairs stay matched. "
        "Instead of --mpp.",
    )
    scale.add_argument(
        "--downsample",
        type=float,
        default=None,
        metavar="N",
        help="Use an explicit level-0-pixels-per-output-pixel factor "
        "instead of --mpp. The only option for an image with no "
        "physical pixel size.",
    )
    geometry.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Tile edge in output pixels (default: {DEFAULT_SIZE}).",
    )
    spacing = geometry.add_mutually_exclusive_group()
    spacing.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between neighbouring tiles in output pixels; stride "
        "is size - overlap (default: 0).",
    )
    spacing.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Distance between neighbouring tile origins in output pixels "
        "(default: --size, i.e. no overlap). --stride 256 with --size 512 "
        "is half-overlapping. Instead of --overlap.",
    )

    filters = parser.add_argument_group("which tiles are kept")
    filters.add_argument(
        "--erode",
        type=_parse_erode,
        default=None,
        metavar="UM",
        help="--roi: only place tile centres inside each ROI shrunk inward by "
        "this many micrometres, so a tile can't take a class it only half "
        "overlaps. 'auto' (the default) is half a tile, which keeps whole "
        "tiles inside the ROI; 0 turns it off. ROIs too small to survive "
        "are counted in the summary.",
    )
    filters.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="--roi: fraction of a tile that must lie inside one class's "
        f"ROI area for it to take that class (default: {DEFAULT_MIN_COVERAGE}). "
        "The highest-covering qualifying class wins.",
    )
    filters.add_argument(
        "--tissue-thresh",
        "--tissue-threshold",
        dest="tissue_thresh",
        type=float,
        default=DEFAULT_TISSUE_THRESH,
        help="Minimum tissue fraction to keep a tile, in both modes "
        f"(default: {DEFAULT_TISSUE_THRESH}). Lower it (e.g. 0.25) if "
        "glandular patterns with large lumens, like cribriform, are being "
        "dropped as background.",
    )
    filters.add_argument(
        "--exclude-text",
        type=str.lower,
        action="append",
        default=None,
        help="textValue marking an exclusion region; any tile overlapping "
        "one at all is dropped, in both modes (repeatable, default: "
        f"{', '.join(DEFAULT_EXCLUDE_TEXT)}).",
    )
    filters.add_argument(
        "--background-label",
        default=None,
        metavar="NAME",
        help="--roi: folder for tissue tiles inside no ROI on an annotated "
        f"slide (default: {DEFAULT_BACKGROUND_LABEL}). Annotators mark "
        "everything, so such tissue is genuinely benign -- but only on "
        "a slide that has ROIs, so slides with none are skipped instead.",
    )
    filters.add_argument(
        "--no-background",
        action="store_true",
        help="--roi: don't emit background tiles at all.",
    )
    filters.add_argument(
        "--background-margin",
        type=int,
        default=None,
        metavar="UM",
        help="--roi: keep background tiles this many micrometres clear of "
        f"every ROI (default: {DEFAULT_BACKGROUND_MARGIN}), so tiles "
        "straddling an annotation edge aren't called benign.",
    )

    sampling = parser.add_argument_group("how many tiles")
    sampling.add_argument(
        "--max-background-tiles",
        type=int,
        default=None,
        metavar="N",
        help="--roi: cap background tiles per slide, sampled at random "
        f"(default: {DEFAULT_MAX_BACKGROUND_TILES}). Background "
        "otherwise vastly outnumbers every graded class.",
    )
    sampling.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        metavar="N",
        help="Cap tiles per slide, sampled at random. Handy for a quick test run.",
    )
    sampling.add_argument(
        "--max-tiles-per-label",
        type=int,
        default=None,
        metavar="N",
        help="Cap tiles per class per slide, sampled at random.",
    )
    sampling.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for every random sample, so a slide tiled twice with the "
        "same settings yields the same tiles (default: 0).",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--format",
        type=_format,
        choices=["png", "jpg"],
        default=None,
        help="Tile image format: png (default, lossless) or jpg/jpeg (much "
        "smaller; standard for histology datasets). Rejected with "
        "--coords-only, which writes no images at all.",
    )
    output.add_argument(
        "--quality",
        type=int,
        default=None,
        help=f"JPEG quality 1-100 (default: {DEFAULT_JPEG_QUALITY}). Only with "
        "--format jpg.",
    )
    output.add_argument(
        "--manifest",
        metavar="PATH",
        default=None,
        help="Also append every tile to this one CSV, across slides and across "
        "runs -- point several commands at the same file to build one "
        "dataset index. Rows already present aren't repeated. Tile paths "
        "are relative to the CSV's folder. (Each slide always gets its own "
        "manifest.csv as well.)",
    )
    output.add_argument(
        "--coords-only",
        action="store_true",
        help="Write only the manifest and parameters, no image files. A "
        "40x whole-mount grids to 80-100k tiles, so this is how you "
        "survey a slide without materialising them.",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Work out every tile and print counts per class and per ROI -- "
        "and which ROIs --erode leaves no room in -- without writing "
        "anything. The cheap way to ask 'do I have enough G4 cribriform?'",
    )

    running = parser.add_argument_group("running")
    running.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip slides whose manifest.csv and tile_params.json show a "
        "finished run with these same parameters, and resume an "
        "interrupted slide from the tiles it already wrote. A slide tiled "
        "with *different* parameters is warned about and skipped too, "
        "unless the global --override is given (which re-tiles it).",
    )
    running.add_argument(
        "--force-local",
        action="store_true",
        help="Read a JPEG-2000 source locally anyway. By default a bare .jp2 "
        "falls forward to the network tier, because the bundled libvips "
        "has no jp2k loader and every tile would decode the whole "
        "codestream through Pillow.",
    )
    running.add_argument(
        "-g", "--group", type=int, help="OMERO group ID (batch mode only)."
    )
    running.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Slides tiled in parallel in batch mode (default: 8).",
    )
    running.add_argument(
        "--connections",
        type=int,
        default=None,
        metavar="N",
        help="Parallel OMERO pixel connections per slide when tiles come over "
        f"the network (default: {DEFAULT_CONNECTIONS}). Raise it if fetches "
        "dominate and the server has headroom.",
    )
    add_creds_args(parser)
    parser.set_defaults(handler=run)


def _effective_overlap(args: argparse.Namespace) -> int:
    if getattr(args, "stride", None) is not None:
        return args.size - args.stride
    return args.overlap


def _validate_args(args: argparse.Namespace) -> None:
    """Reject flag combinations that cannot mean anything, before connecting."""
    if args.whole:
        if args.all or args.text_filter:
            raise SystemExit(
                "error: --all and --text-filter select annotations, which --whole "
                "ignores; use --roi, or drop them."
            )
        for flag, value in (
            ("--background-label", args.background_label),
            ("--background-margin", args.background_margin),
            ("--max-background-tiles", args.max_background_tiles),
            ("--erode", args.erode),
        ):
            if value is not None:
                raise SystemExit(
                    f"error: {flag} only applies to --roi; --whole writes every tissue "
                    "tile under 'whole/' and has no ROIs to label from."
                )
        if args.no_background:
            raise SystemExit(
                "error: --no-background only applies to --roi; --whole has no "
                "background class to turn off."
            )
    else:
        if args.all and args.text_filter:
            raise SystemExit(
                "error: --all and --text-filter are contradictory -- one takes every "
                "annotation, the other a whitelist. Pick one."
            )
        if not args.all and not args.text_filter:
            raise SystemExit(
                "error: --roi needs to know which annotations to use: pass --all or at "
                "least one --text-filter value."
            )
        if args.no_background and args.background_label is not None:
            raise SystemExit(
                "error: --no-background and --background-label are contradictory -- one "
                "turns the background class off, the other names it."
            )

    if args.size <= 0:
        raise SystemExit(f"error: --size must be positive, got {args.size}.")
    if args.stride is not None and not 0 < args.stride <= args.size:
        raise SystemExit(
            f"error: --stride must be between 1 and --size ({args.size}), "
            f"got {args.stride}."
        )
    if not 0 <= args.overlap < args.size:
        raise SystemExit(
            f"error: --overlap must be at least 0 and less than --size ({args.size}), "
            f"got {args.overlap}."
        )
    if not 0.0 <= args.min_coverage <= 1.0:
        raise SystemExit(
            f"error: --min-coverage is a fraction between 0 and 1, got {args.min_coverage}."
        )
    if not 0.0 <= args.tissue_thresh <= 1.0:
        raise SystemExit(
            f"error: --tissue-thresh is a fraction between 0 and 1, got {args.tissue_thresh}."
        )
    if args.coords_only and args.format is not None:
        raise SystemExit(
            "error: --coords-only writes no image files, so --format has nothing to "
            "apply to; drop one or the other."
        )
    if args.quality is not None:
        if args.format != "jpg":
            raise SystemExit(
                "error: --quality only applies to --format jpg; PNG is lossless."
            )
        if not 1 <= args.quality <= 100:
            raise SystemExit(f"error: --quality must be 1-100, got {args.quality}.")
    if args.mpp is not None and args.mpp <= 0:
        raise SystemExit(f"error: --mpp must be positive, got {args.mpp}.")
    if args.downsample is not None and args.downsample <= 0:
        raise SystemExit(
            f"error: --downsample must be positive, got {args.downsample}."
        )
    if args.background_margin is not None and args.background_margin < 0:
        raise SystemExit(
            f"error: --background-margin must be zero or greater, got {args.background_margin}."
        )
    for flag, value in (
        ("--max-tiles", args.max_tiles),
        ("--max-tiles-per-label", args.max_tiles_per_label),
        ("--max-background-tiles", args.max_background_tiles),
    ):
        if value is not None and value < 0:
            raise SystemExit(f"error: {flag} must be zero or greater, got {value}.")
    if args.connections is not None and args.connections < 1:
        raise SystemExit(
            f"error: --connections must be 1 or more, got {args.connections}."
        )
    if args.workers < 1:
        raise SystemExit(f"error: --workers must be 1 or more, got {args.workers}.")


def _build_params(args: argparse.Namespace):
    """Turn the parsed arguments into a :class:`lavlab.tiling.TileParams`."""
    from lavlab.tiling import TileParams

    scales = None
    if args.downsample is not None:
        mpp = None
        downsample = float(args.downsample)
    elif args.scales is not None:
        scales = list(args.scales) if len(args.scales) > 1 else None
        mpp = float(args.scales[0])
        downsample = None
    else:
        mpp = DEFAULT_MPP if args.mpp is None else float(args.mpp)
        downsample = None

    if args.whole or args.no_background:
        background_label = None
        background_margin = 0.0
        max_background_tiles = None
    else:
        background_label = args.background_label or DEFAULT_BACKGROUND_LABEL
        background_margin = float(
            DEFAULT_BACKGROUND_MARGIN
            if args.background_margin is None
            else args.background_margin
        )
        max_background_tiles = (
            DEFAULT_MAX_BACKGROUND_TILES
            if args.max_background_tiles is None
            else args.max_background_tiles
        )

    exclude_text = (
        list(DEFAULT_EXCLUDE_TEXT)
        if args.exclude_text is None
        else list(args.exclude_text)
    )

    if args.whole:
        erode_um = 0.0
    elif args.erode is None or args.erode == "auto":
        erode_um = None
    else:
        erode_um = float(args.erode)

    fmt = args.format or "png"
    quality = None
    if fmt == "jpg" and not args.coords_only:
        quality = args.quality or DEFAULT_JPEG_QUALITY

    return TileParams(
        mode="whole" if args.whole else "roi",
        include_all=bool(args.all),
        text_filter=list(args.text_filter),
        mpp=mpp,
        downsample=downsample,
        size=args.size,
        overlap=_effective_overlap(args),
        min_coverage=args.min_coverage,
        tissue_thresh=args.tissue_thresh,
        fmt=fmt,
        coords_only=bool(args.coords_only),
        labels=args.labels,
        exclude_text=exclude_text,
        background_label=background_label,
        background_margin_um=background_margin,
        max_background_tiles=max_background_tiles,
        max_tiles=args.max_tiles,
        max_tiles_per_label=args.max_tiles_per_label,
        seed=args.seed,
        scales=scales,
        erode_um=erode_um,
        quality=quality,
    )


def _explain_erosion(params) -> None:
    """Say what --erode will do when it's easy to get wrong.

    * ``--erode 0`` brings back coverage-only labelling, where a tile
      straddling an ROI edge can still take that ROI's class -- the label
      noise erosion exists to remove -- so it is never silent.
    * With ``--scales``, the default inset is half a tile *at the labelling
      (finest) scale*. The coarser crops are context and reach further out,
      past the ROI edge; say by how much, and what to pass to keep every
      scale inside instead.
    """
    if params.mode != "roi":
        return
    if params.erode_um == 0:
        log.warning(
            "--erode 0: tiles are labelled by ROI coverage alone, so a tile "
            "straddling an ROI edge can still take that class (label noise). Only "
            "use this if you know you want it; the default keeps whole tiles "
            "inside their ROI."
        )
        return
    if params.erode_um is None and params.scales:
        finest, coarsest = min(params.scales), max(params.scales)
        inset = params.size * finest / 2
        reach = params.size * coarsest / 2
        log.info(
            "--erode auto: tile centres are kept %g um inside their ROI (half a "
            "%d px tile at %g um/px, the labelling scale). The %g um/px context "
            "crops reach %g um from the centre, so they can extend past the ROI "
            "edge -- usually the point of context. To keep every scale inside the "
            "ROI, pass --erode %g.",
            inset,
            params.size,
            finest,
            coarsest,
            reach,
            reach,
        )


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    _explain_erosion(_build_params(args))
    target = parse_target(args.target)
    if target == "batch":
        _run_batch(args)
    else:
        _run_single(args, target)


def _tile_one(conn, image, args: argparse.Namespace, params):
    from lavlab.tiling import tile_slide

    return tile_slide(
        conn,
        image,
        params,
        args.out,
        override=args.override,
        skip_existing=args.skip_existing,
        force_local=args.force_local,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        store_count=args.connections,
    )


_DROP_NAMES = {
    "dropped_no_tissue": "not enough tissue",
    "dropped_excluded": "in an exclusion ROI",
    "dropped_roi_edge": "centre too near an ROI edge (--erode)",
    "dropped_unlabeled": "partly inside an ROI (below --min-coverage, or near one)",
    "dropped_scale_edge": "coarser --scales crop runs off the slide",
}


def _roi_name(roi_id) -> str:
    return "ROI ?" if roi_id is None else f"ROI {roi_id}"


def _print_report(result, params) -> None:
    """Print a single slide's per-class / per-ROI breakdown."""
    scales = params.output_mpps()
    if len(scales) > 1:
        print(
            "  Scales: "
            + ", ".join(f"{s} um/px" for s in scales)
            + " -- each tile location is written once per scale."
        )

    if params.mode == "roi":
        if result.erode_um == 0:
            print(
                "  Erosion: off (--erode 0) -- tiles can straddle ROI edges and "
                "still take the class."
            )
        elif result.erode_um is not None:
            print(
                f"  Erosion: tile centres at least {result.erode_um:g} um inside "
                "their ROI."
            )

    labels = sorted(set(result.available) | set(result.label_counts))
    if labels:
        width = max(len(label) for label in labels)
        print("  Tiles by class (kept / available before caps):")
        for label in labels:
            print(
                f"    {label:<{width}}  {result.label_counts.get(label, 0):>7}"
                f" / {result.available.get(label, 0)}"
            )
    else:
        print("  No tiles.")

    if result.roi_counts:
        print("  Tiles per ROI:")
        width = max(len(label) for label, _rid in result.roi_counts)
        for (label, roi_id), count in sorted(
            result.roi_counts.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)
        ):
            note = "   <- no tiles" if count == 0 else ""
            print(f"    {label:<{width}}  {_roi_name(roi_id):<12} {count:>7}{note}")

    if result.lost_rois:
        print(
            f"  {len(result.lost_rois)} of {result.roi_total} ROIs are smaller than "
            "about one tile after --erode, so no tile centre fits: "
            + ", ".join(f"{label} {_roi_name(rid)}" for label, rid in result.lost_rois)
        )

    drops = [
        f"{_DROP_NAMES.get(key, key)}: {value}"
        for key, value in sorted(result.stats.items())
        if key.startswith("dropped_") and value
    ]
    if drops:
        print("  Dropped -- " + "; ".join(drops))


def _run_single(args: argparse.Namespace, image_id: int) -> None:
    from lavlab.tiling import TilingError

    params = _build_params(args)
    conn = connect_from_args(args)
    try:
        image = conn.getObject("Image", image_id)
        if image is None:
            raise SystemExit(f"error: image {image_id} not found.")
        group_of(conn, image)

        try:
            result = _tile_one(conn, image, args, params)
        except TilingError as exc:
            raise SystemExit(f"error: image {image_id}: {exc}") from None
        except Exception as exc:
            from lavlab.omero_client import describe_error, is_resource_error

            if not is_resource_error(exc):
                raise
            raise SystemExit(
                f"error: image {image_id}: OMERO could not read this image's pixels "
                "-- its file is likely missing or corrupt on the server. "
                f"({describe_error(exc)})"
            ) from None

        if result.status == "skipped":
            print(f"Image {image_id}: skipped ({result.reason}).")
        elif result.status == "unannotated":
            print(f"Image {image_id}: no ROIs, nothing tiled.")
        elif result.dry_run:
            print(f"Dry run for image {image_id} -- nothing was written.")
            _print_report(result, params)
        else:
            counts = ", ".join(
                f"{k}={v}" for k, v in sorted(result.label_counts.items())
            )
            print(
                f"Completed tile for image {image_id}: {result.slide_dir} "
                f"({counts or 'no tiles'}, tier {result.tier})"
            )
            if result.reused:
                print(f"  Resumed: reused {result.reused} tiles already on disk.")
            if result.lost_rois:
                print(
                    f"  {len(result.lost_rois)} of {result.roi_total} ROIs too small "
                    "for a tile after --erode (run with --dry-run for the list)."
                )
            if args.manifest:
                print(f"  Appended to {args.manifest}")
    finally:
        conn.close()


_WORKER_STATE: dict = {}

# The most a worker waits before its first login, so N workers starting
# together don't all hit the session service in the same instant.
_MAX_STARTUP_STAGGER = 5.0


def _close_quietly(conn) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        log.debug("closing a dead connection failed", exc_info=True)


def _init_worker(args: argparse.Namespace, params) -> None:
    """Pool initializer. Must never raise.

    A ``multiprocessing.Pool`` whose initializer raises silently respawns the
    worker, which runs the initializer again -- forever. With logins failing
    that is a reconnect storm with no end, so a failed first login is logged
    and left for the worker's first slide to retry.
    """
    global _WORKER_STATE
    _WORKER_STATE = {"conn": None, "args": args, "params": params}
    time.sleep(random.uniform(0, min(_MAX_STARTUP_STAGGER, 0.5 * args.workers)))
    try:
        _WORKER_STATE["conn"] = connect_from_args(args)
    except (Exception, SystemExit) as exc:
        log.warning(
            "Worker could not log in at start-up (%s); it will retry on its first "
            "slide.",
            exc,
        )


def _worker_conn(args: argparse.Namespace):
    """This worker's connection, logging in again if it has none or lost it.

    The old connection is closed first: an abandoned one keeps its server
    session alive until it times out, and enough of those is exactly what
    makes the session service start refusing logins.
    """
    conn = _WORKER_STATE.get("conn")
    if conn is not None and conn.isConnected():
        return conn
    _close_quietly(conn)
    _WORKER_STATE["conn"] = None
    conn = connect_from_args(args)
    _WORKER_STATE["conn"] = conn
    return conn


def _process_one(image_id: int):
    """Tile one slide in a batch worker. Never raises.

    Anything escaping a worker is re-raised by the pool in the parent and
    ends the whole batch, so every failure -- expected or not -- becomes a
    ``failed`` :class:`lavlab.tiling.SlideResult` here, and the batch moves
    on to the next slide.
    """
    from lavlab.omero_client import describe_error
    from lavlab.tiling import SlideResult

    try:
        return _tile_with_retries(image_id)
    except (Exception, SystemExit) as exc:
        reason = describe_error(exc)
        log.warning(
            "Image %d: failed (%s); continuing with the next slide.",
            image_id,
            reason,
            exc_info=True,
        )
        return SlideResult(image_id, "failed", reason=reason)


def _tile_with_retries(image_id: int):
    """Tile one slide, reconnecting only for genuine connection failures."""
    from lavlab.omero_client import describe_error, is_conn_error, is_resource_error
    from lavlab.tiling import SlideResult, TilingError

    args = _WORKER_STATE["args"]
    params = _WORKER_STATE["params"]
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        conn = None
        try:
            # Inside the try: a login that fails here is handled below like
            # any other error, instead of escaping the worker.
            conn = _worker_conn(args)
            image = conn.getObject("Image", image_id)
            if image is None:
                log.warning("Image %d not found, skipping.", image_id)
                return SlideResult(image_id, "failed", reason="not found")
            group_of(conn, image)
            return _tile_one(conn, image, args, params)
        except TilingError as exc:
            log.warning("Image %d: %s", image_id, exc)
            return SlideResult(image_id, "failed", reason=str(exc))
        except Exception as exc:
            if is_resource_error(exc):
                reason = describe_error(exc)
                log.warning(
                    "Image %d: OMERO could not read this image's pixels -- its file "
                    "is likely missing or corrupt on the server. Skipping it; the "
                    "connection is fine, so no reconnect. Server said: %s",
                    image_id,
                    reason,
                )
                return SlideResult(image_id, "failed", reason=reason)
            if is_conn_error(exc) and attempt < max_attempts:
                log.warning(
                    "Image %d: connection error, reconnecting (attempt %d/%d): %s",
                    image_id,
                    attempt,
                    max_attempts,
                    describe_error(exc),
                )
                _close_quietly(conn)
                _WORKER_STATE["conn"] = None
                continue
            raise
    return SlideResult(image_id, "failed", reason="retries exhausted")


def _format_ids(results, status: str) -> str:
    return ", ".join(str(r.image_id) for r in results if r.status == status)


def _summarise_batch(image_ids, results) -> None:
    """Report what the batch achieved, and fail the run if it achieved nothing.

    Split out from ``_run_batch`` so the reporting and exit-code rules can be
    exercised without standing up a worker pool and an OMERO connection.

    *results* holds one :class:`lavlab.tiling.SlideResult` per image. The
    per-label totals matter as much as the per-slide ones: a run that
    "succeeded" on every slide while producing no G5 tiles at all is a
    broken run, and only the label line shows it.
    """
    by_status = Counter(r.status for r in results)
    done = by_status.get("done", 0)
    skipped = by_status.get("skipped", 0)
    unannotated = by_status.get("unannotated", 0)
    failed = by_status.get("failed", 0)
    dry_run = any(getattr(r, "dry_run", False) for r in results)

    if dry_run:
        log.info("Dry run: nothing was written.")
    log.info(
        "Batch complete: %d/%d slides tiled, %d skipped, %d unannotated, %d failed.",
        done,
        len(image_ids),
        skipped,
        unannotated,
        failed,
    )

    tiers = Counter(r.tier for r in results if r.status == "done" and r.tier)
    if tiers:
        log.info(
            "Served by tier: %s",
            ", ".join(f"{tier}={count}" for tier, count in sorted(tiers.items())),
        )

    labels: Counter = Counter()
    available: Counter = Counter()
    for result in results:
        labels.update(result.label_counts)
        available.update(getattr(result, "available", Counter()))
    if labels:
        log.info(
            "Tiles by label: %s",
            ", ".join(f"{label}={count}" for label, count in sorted(labels.items())),
        )
        log.info("Total tiles: %d", sum(labels.values()))
    if dry_run and available:
        log.info(
            "Available before caps: %s",
            ", ".join(f"{label}={count}" for label, count in sorted(available.items())),
        )

    roi_total = sum(getattr(r, "roi_total", 0) for r in results)
    if roi_total:
        empty = sum(
            1
            for r in results
            for count in getattr(r, "roi_counts", Counter()).values()
            if count == 0
        )
        lost = Counter(
            label for r in results for label, _rid in getattr(r, "lost_rois", [])
        )
        log.info(
            "ROIs: %d selected, %d gave no tiles, %d too small for a tile after "
            "--erode%s",
            roi_total,
            empty,
            sum(lost.values()),
            (" (" + ", ".join(f"{k}={v}" for k, v in sorted(lost.items())) + ")")
            if lost
            else "",
        )

    for status, heading in (
        ("skipped", "Skipped"),
        ("unannotated", "Unannotated (no ROIs)"),
        ("failed", "Failed"),
    ):
        ids = _format_ids(results, status)
        if ids:
            log.info("%s image IDs: %s", heading, ids)

    if image_ids and failed == len(image_ids):
        raise SystemExit(
            f"error: all {len(image_ids)} images failed; see the per-image errors above."
        )


def _run_batch(args: argparse.Namespace) -> None:
    from lavlab.omero_client import iter_image_ids

    params = _build_params(args)
    log.info("Starting %d workers.", args.workers)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(
        args.workers, initializer=_init_worker, initargs=(args, params)
    ) as pool:
        conn = connect_from_args(args)
        try:
            image_ids = list(iter_image_ids(conn, args.group))
        finally:
            conn.close()

        log.info("Found %d images.", len(image_ids))
        # One task per image, collected per image: if a result still can't
        # come back (say, it fails to unpickle), only that image is lost.
        pending = [
            (image_id, pool.apply_async(_process_one, (image_id,)))
            for image_id in image_ids
        ]
        results = [_collect(image_id, task) for image_id, task in pending]

    _summarise_batch(image_ids, results)


def _collect(image_id: int, task):
    """Wait for one image's result; a failure counts that image as failed."""
    from lavlab.tiling import SlideResult

    try:
        return task.get()
    except Exception as exc:
        log.warning(
            "Image %d: worker failed (%s: %s); counting it as failed.",
            image_id,
            type(exc).__name__,
            exc,
        )
        return SlideResult(image_id, "failed", reason=type(exc).__name__)
