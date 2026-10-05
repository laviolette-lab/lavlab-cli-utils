# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""``lavlab geojson`` -- move QuPath GeoJSON annotations in and out of OMERO.

lavlab geojson import slide.geojson --image 1 -u you -s omero
lavlab geojson import ./backups/ --dataset 5 -u you -s omero
lavlab geojson import --map restore.csv -u you -s omero
lavlab geojson export --project 2 --out ./archive/ -u you -s omero
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import logging
import os
import sys
from pathlib import Path

from lavlab.commands._shared import (
    add_creds_args,
    connect_from_args,
    ensure_parent_dir,
    make_temp_path,
)
from lavlab.geojson.geojson_io import (
    ConversionError,
    convert_file,
    dump_geojson,
    summarise_annotations,
    summarise_features,
)

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    geojson_parser = subparsers.add_parser(
        "geojson", help="Move QuPath GeoJSON annotations in and out of OMERO."
    )
    geojson_subparsers = geojson_parser.add_subparsers(
        dest="geojson_action", required=True
    )

    importer = geojson_subparsers.add_parser(
        "import", help="create OMERO ROIs from GeoJSON files"
    )
    importer.add_argument("paths", nargs="*", help=".geojson files or directories")
    target = importer.add_argument_group("which image")
    target.add_argument("--image", type=int, help="image id, for a single file")
    target.add_argument(
        "--dataset", type=int, help="match files to images in this dataset by name"
    )
    target.add_argument("--map", help="CSV of 'geojson_path,image_id' rows")
    style = importer.add_argument_group("appearance")
    style.add_argument(
        "--fill-alpha", type=int, default=90, help="0-255, or 0 for outline only"
    )
    style.add_argument("--stroke-width", type=float, default=2.0)
    add_creds_args(importer)
    importer.add_argument(
        "--dry-run", action="store_true", help="report only; never contacts OMERO"
    )
    importer.set_defaults(handler=run_import)

    exporter = geojson_subparsers.add_parser(
        "export", help="archive OMERO ROIs as GeoJSON"
    )
    source = exporter.add_argument_group("what to export")
    source.add_argument("--image", type=int)
    source.add_argument("--dataset", type=int)
    source.add_argument("--project", type=int)
    source.add_argument("--group", type=int, help="every image in this OMERO group")
    output = exporter.add_argument_group("output")
    output.add_argument(
        "--out", help="directory to write into (required unless --skip-local)"
    )
    output.add_argument(
        "--skip-empty", action="store_true", help="no file for images with no ROIs"
    )
    output.add_argument("--compact", action="store_true", help="minified JSON")
    output.add_argument(
        "--ellipse-points",
        type=int,
        default=64,
        help="vertices used to approximate an ellipse",
    )
    output.add_argument(
        "--keep-bridges",
        action="store_true",
        help="leave keyhole slits in place instead of restoring holes",
    )
    output.add_argument(
        "--datestamp",
        action="store_true",
        help="write into <out>/YYYY-MM-DD/ so runs do not overwrite each other",
    )
    output.add_argument(
        "--overwrite", action="store_true", help="allow replacing an existing file"
    )
    output.add_argument(
        "--upload",
        action="store_true",
        help="also attach the exported GeoJSON to the image in OMERO, under the "
        "lavlab.geojson namespace",
    )
    output.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip images that already have their GeoJSON export attached",
    )
    output.add_argument(
        "--skip-local",
        action="store_true",
        help="keep no local copy -- write to a temporary file, upload it, delete it. "
        "Requires --upload; makes --out unnecessary",
    )
    add_creds_args(exporter)
    exporter.add_argument("--dry-run", action="store_true")
    exporter.set_defaults(handler=run_export)


def _exactly_one(**options) -> list[str]:
    """Return the names of whichever options were supplied.

    :return: supplied option names, formatted as CLI flags
    :rtype: list[str]
    """
    return [f"--{name}" for name, value in options.items() if value is not None]


def _collect_paths(raw_paths: list[str]) -> list[Path]:
    """Expand directories in the argument list into the files inside.

    :param raw_paths: files or directories from the command line
    :type raw_paths: list[str]
    :raises SystemExit: if a path does not exist
    :return: the files to process
    :rtype: list[Path]
    """
    found: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.glob("*.geojson")) + sorted(path.glob("*.json")))
        elif path.exists():
            found.append(path)
        else:
            raise SystemExit(f"error: no such file or directory: {path}")
    return found


def _pairs_from_map(map_path: str) -> list[tuple]:
    """Read an explicit CSV of ``geojson_path,image_id`` rows.

    A header row is optional and detected automatically; ``#`` starts a
    comment.

    :param map_path: the CSV to read
    :type map_path: str
    :raises SystemExit: if a row is malformed or the file has no usable rows
    :return: ``[(path, image_id)]``
    :rtype: list[tuple]
    """
    pairs: list[tuple] = []
    with open(map_path, newline="", encoding="utf-8") as handle:
        for lineno, row in enumerate(csv.reader(handle), start=1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                raise SystemExit(
                    f"error: {map_path}:{lineno}: expected 'path,image_id'"
                )
            path, raw_id = row[0].strip(), row[1].strip()
            if lineno == 1 and not raw_id.lstrip("-").isdigit():
                continue
            if not raw_id.lstrip("-").isdigit():
                raise SystemExit(
                    f"error: {map_path}:{lineno}: {raw_id!r} is not an image id"
                )
            pairs.append((Path(path), int(raw_id)))
    if not pairs:
        raise SystemExit(f"error: {map_path}: no usable rows")
    return pairs


def _report_conversion(path: Path, image_id, result) -> None:
    """Print the per-file summary for an import.

    :param path: the file being imported
    :type path: Path
    :param image_id: destination image id, or ``None`` if unresolved
    :param result: the conversion result to describe
    """
    stats = summarise_annotations(result.annotations)
    target = f" -> image {image_id}" if image_id is not None else ""
    print(f"  {path.name}{target}")
    print(
        f"    {stats['annotations']} feature(s), {stats['shapes']} shape(s), "
        f"{stats['labelled']} labelled, {stats['unlabelled']} unlabelled"
    )
    if stats["classes"]:
        print(f"    classes: {', '.join(stats['classes'])}")
    for warning in result.warnings:
        print(f"    {warning}")


def run_import(args: argparse.Namespace) -> None:
    chosen = _exactly_one(image=args.image, dataset=args.dataset, map=args.map)
    if len(chosen) != 1:
        raise SystemExit(
            "error: give exactly one of --image, --dataset or --map "
            f"(got {', '.join(chosen) or 'none'})"
        )
    if args.map and args.paths:
        raise SystemExit("error: --map already lists the files; do not also pass paths")
    if not args.map and not args.paths:
        raise SystemExit("error: no input files given")

    conn = None
    if args.map:
        pairs = _pairs_from_map(args.map)
        for path, _ in pairs:
            if not path.exists():
                raise SystemExit(f"error: {args.map} refers to a missing file: {path}")
        conn = None if args.dry_run else connect_from_args(args)
    else:
        paths = _collect_paths(args.paths)
        if args.image is not None:
            if len(paths) != 1:
                raise SystemExit(
                    f"error: --image takes one file, but {len(paths)} were given; "
                    "use --dataset or --map for more than one"
                )
            pairs = [(paths[0], args.image)]
            conn = None if args.dry_run else connect_from_args(args)
        elif args.dry_run:
            print(
                f"{len(paths)} file(s) would be matched to images in dataset "
                f"{args.dataset} by filename:\n"
            )
            bad = 0
            for path in paths:
                try:
                    _report_conversion(path, None, convert_file(path))
                except ConversionError as exc:
                    print(f"  {path.name}\n    FAILED: {exc}", file=sys.stderr)
                    bad += 1
            print("\n(dry run -- connect without --dry-run to resolve image ids)")
            if bad:
                raise SystemExit(1)
            return
        else:
            from lavlab.geojson.omero_io import match_by_name

            conn = connect_from_args(args)
            pairs, problems = match_by_name(conn, args.dataset, paths)
            for problem in problems:
                print(f"  SKIP {problem}", file=sys.stderr)

    if not pairs:
        raise SystemExit("error: nothing to import")

    print(f"{len(pairs)} file(s) to import" + (" [DRY RUN]" if args.dry_run else ""))
    print()

    total = 0
    failures: list[tuple] = []
    try:
        for path, image_id in pairs:
            try:
                result = convert_file(path)
                _report_conversion(path, image_id, result)
                if args.dry_run or not result.annotations:
                    continue
                from lavlab.geojson.omero_io import import_annotations

                created = import_annotations(
                    conn,
                    image_id,
                    result.annotations,
                    args.fill_alpha,
                    args.stroke_width,
                ).created
                total += created
                print(f"    created {created} ROI(s)")
            except (ConversionError, LookupError) as exc:
                print(f"    FAILED: {exc}", file=sys.stderr)
                failures.append((path.name, str(exc)))
    finally:
        if conn is not None:
            conn.close()

    print()
    if args.dry_run:
        print("(dry run -- nothing was sent to OMERO)")
    else:
        print(
            f"done: {total} ROI(s) created across {len(pairs) - len(failures)} file(s)"
        )
    if failures:
        print(f"{len(failures)} file(s) failed:", file=sys.stderr)
        for name, error in failures:
            print(f"  {name}: {error}", file=sys.stderr)
        raise SystemExit(1)


def run_export(args: argparse.Namespace) -> None:
    from lavlab.geojson.omero_io import (
        export_image,
        geojson_annotation_name,
        has_uploaded_geojson,
        iter_images,
        upload_geojson,
    )

    chosen = _exactly_one(
        image=args.image, dataset=args.dataset, project=args.project, group=args.group
    )
    if len(chosen) != 1:
        raise SystemExit(
            "error: give exactly one of --image, --dataset, --project or --group "
            f"(got {', '.join(chosen) or 'none'})"
        )
    if args.skip_local and not args.upload:
        raise SystemExit(
            "error: --skip-local without --upload would export a GeoJSON and then "
            "throw it away; add --upload, or drop --skip-local."
        )
    if not args.skip_local and not args.out:
        raise SystemExit("error: --out is required unless --skip-local is given.")

    outdir = Path(args.out) if args.out else None
    if outdir is not None:
        if args.datestamp:
            today = dt.datetime.now(tz=dt.UTC).date().isoformat()
            outdir = outdir / today
        if not args.dry_run:
            outdir.mkdir(parents=True, exist_ok=True)

    conn = connect_from_args(args)
    destination = "OMERO attachments only" if args.skip_local else str(outdir)
    print(f"exporting to {destination}" + (" [DRY RUN]" if args.dry_run else ""))
    print()

    images = written = features = skipped = 0
    try:
        for image in iter_images(
            conn,
            image=args.image,
            dataset=args.dataset,
            project=args.project,
            group=args.group,
        ):
            images += 1
            conn.SERVICE_OPTS.setOmeroGroup(image.getDetails().getGroup().getId())

            if args.skip_existing and has_uploaded_geojson(image):
                print(
                    f"  {image.getName()} (image {image.getId()}) -> "
                    "already attached, skipping"
                )
                skipped += 1
                continue

            result = export_image(
                conn,
                image.getId(),
                ellipse_segments=args.ellipse_points,
                unbridge=not args.keep_bridges,
            )
            if not result.features and args.skip_empty:
                continue

            filename = geojson_annotation_name(image)
            stats = summarise_features(result.features)
            print(f"  {image.getName()} (image {image.getId()}) -> {filename}")
            holes = f", {stats['holed']} with holes" if stats["holed"] else ""
            print(f"    {stats['features']} feature(s){holes}")
            if stats["geometries"]:
                print(
                    "    "
                    + ", ".join(
                        f"{k}: {v}" for k, v in sorted(stats["geometries"].items())
                    )
                )
            if stats["classes"]:
                print(f"    classes: {', '.join(stats['classes'])}")
            for warning in result.warnings:
                print(f"    {warning}")

            if not args.skip_local:
                target = outdir / filename
                if target.exists() and not (args.overwrite or args.dry_run):
                    raise SystemExit(
                        f"error: {target} already exists. An archive that silently "
                        "overwrites itself is not an archive -- use --datestamp "
                        "for dated runs, or --overwrite if that is what you meant."
                    )

            if not args.dry_run:
                text = dump_geojson(result.features, indent=None if args.compact else 1)
                if args.skip_local:
                    local_path = make_temp_path("geojson")
                else:
                    ensure_parent_dir(str(target))
                    local_path = str(target)
                try:
                    Path(local_path).write_text(text, encoding="utf-8")
                    if args.upload:
                        # A failed upload shouldn't lose an export that
                        # otherwise succeeded -- same reasoning as lr/roi.
                        try:
                            upload_geojson(conn, image, local_path)
                        except Exception:
                            log.warning(
                                "Image %d: exported but upload to OMERO failed.",
                                image.getId(),
                                exc_info=True,
                            )
                        else:
                            print(f"    uploaded as {filename}")
                finally:
                    if args.skip_local:
                        with contextlib.suppress(OSError):
                            os.remove(local_path)

            features += len(result.features)
            written += 1
    finally:
        conn.close()

    print()
    skipped_note = f", {skipped} already attached" if skipped else ""
    if args.dry_run:
        print(
            f"(dry run -- {features} feature(s) from {images} image(s)"
            f"{skipped_note}, nothing written)"
        )
    elif args.skip_local:
        print(
            f"done: {features} feature(s) from {images} image(s) uploaded for "
            f"{written} image(s){skipped_note}, nothing stored locally"
        )
    else:
        print(
            f"done: {features} feature(s) from {images} image(s) written to "
            f"{written} file(s) in {outdir}{skipped_note}"
        )
