# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""``lavlab meta roi textvalue`` -- fill in blank ROI shape comments by
matching each shape's stroke color against a palette."""

from __future__ import annotations

import argparse
import logging

from lavlab.commands._shared import (
    add_advanced_group,
    add_command,
    add_creds_args,
    add_subcommands,
    connect_from_args,
    group_of,
)
from lavlab.palettes import load_color_mapping, match_color

# omero.rtypes, lavlab.omero_client and lavlab.roi (numpy, skimage, omero)
# are imported inside the functions that use them, so that building the
# argument parser -- and therefore --help -- stays pure Python.

log = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 10


TEXTVALUE_DESCRIPTION = """\
Label unlabelled ROI shapes by their color.

Every shape with an empty comment (textValue) has its outline color compared
against PALETTE; if one of the palette's colors is close enough, that color's
label is written as the shape's comment. Shapes that already have a comment
are never changed. Try --dry-run first to see what would change.

PALETTE is either 'default' (the lab's standard annotation colors) or a path
to a YAML/JSON file of entries like:
  - rgb: [48, 255, 50]
    text: G3
  - hex: "#FF7A00"
    text: HGPIN"""

TEXTVALUE_EXAMPLES = """\
Examples:
  lavlab meta roi textvalue default 12345 12346 --dry-run
  lavlab meta roi textvalue ./my_palette.yaml -g 3"""


def add_parser(subparsers) -> None:
    meta_parser = add_command(
        subparsers,
        "meta",
        help="Edit metadata in OMERO (e.g. label ROIs by color).",
        usage="%(prog)s COMMAND ...",
        description="Edit metadata stored in OMERO.",
    )
    meta_subparsers = add_subcommands(meta_parser, "meta_resource")

    roi_parser = add_command(
        meta_subparsers,
        "roi",
        help="Edit ROI metadata.",
        usage="%(prog)s COMMAND ...",
        description="Edit metadata on OMERO ROIs.",
    )
    roi_subparsers = add_subcommands(roi_parser, "meta_roi_action")

    textvalue_parser = add_command(
        roi_subparsers,
        "textvalue",
        help="Fill in blank ROI comments by matching each shape's color to a palette.",
        usage="%(prog)s PALETTE (IMAGE_ID ... | -g ID) [options]",
        description=TEXTVALUE_DESCRIPTION,
        epilog=TEXTVALUE_EXAMPLES,
    )
    textvalue_parser.add_argument(
        "text_mapping",
        metavar="PALETTE",
        help="A built-in palette name (e.g. 'default') or a YAML/JSON palette file.",
    )
    textvalue_parser.add_argument(
        "image_ids",
        nargs="*",
        type=int,
        metavar="IMAGE_ID",
        help="OMERO image IDs to process. Leave out and use -g for a whole group.",
    )
    textvalue_parser.add_argument(
        "-g",
        "--group",
        type=int,
        metavar="ID",
        help="Process every image in this OMERO group.",
    )
    textvalue_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which shapes would be labelled, and with what, without "
        "writing anything to OMERO.",
    )

    add_creds_args(textvalue_parser)

    advanced = add_advanced_group(textvalue_parser)
    advanced.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        metavar="N",
        help="How far each of red, green and blue (0-255) may be from a palette "
        f"color and still count as a match (default: {DEFAULT_TOLERANCE}).",
    )
    textvalue_parser.set_defaults(handler=run)


def _shape_has_comment(shape) -> bool:
    tv = shape.getTextValue()
    if tv is None:
        return False
    return bool(tv.getValue().strip())


def _shape_color(shape):
    from lavlab.roi import uint_to_rgba

    stroke = shape.getStrokeColor()
    if stroke is None:
        return None
    r, g, b, _a = uint_to_rgba(stroke.getValue())
    return r, g, b


def _process_image(
    conn, image_id: int, mapping, tolerance: int, dry_run: bool = False
) -> tuple[int, int, int]:
    from omero.rtypes import rstring

    from lavlab.roi import get_rois

    image = conn.getObject("Image", image_id)
    if image is None:
        log.warning("Image %d not found, skipping.", image_id)
        return (0, 0, 0)
    group_of(conn, image)

    updated = skipped_has_comment = skipped_no_match = 0
    for roi in get_rois(image):
        for shape in roi.copyShapes():
            if _shape_has_comment(shape):
                skipped_has_comment += 1
                continue

            rgb = _shape_color(shape)
            if rgb is None:
                skipped_no_match += 1
                continue

            label = match_color(rgb, mapping, tolerance)
            if label is None:
                skipped_no_match += 1
                continue

            shape_id = shape.getId().getValue() if shape.getId() is not None else None
            if dry_run:
                log.info(
                    "Image %d: shape %s would be set to %r (color %s).",
                    image_id,
                    shape_id,
                    label,
                    rgb,
                )
            else:
                shape.setTextValue(rstring(label))
                conn.getUpdateService().saveObject(shape)
                log.info(
                    "Image %d: shape %s set to %r (color %s).",
                    image_id,
                    shape_id,
                    label,
                    rgb,
                )
            updated += 1

    return (updated, skipped_has_comment, skipped_no_match)


def run(args: argparse.Namespace) -> None:
    from lavlab.omero_client import iter_image_ids

    try:
        mapping = load_color_mapping(args.text_mapping)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    if args.image_ids:
        image_ids = args.image_ids
    elif args.group is not None:
        pass
    else:
        raise SystemExit("error: specify one or more image IDs, or --group.")

    conn = connect_from_args(args)
    try:
        if not args.image_ids:
            image_ids = list(iter_image_ids(conn, args.group))

        verb = "would_update" if args.dry_run else "updated"
        total_updated = total_skip_comment = total_skip_no_match = 0
        for image_id in image_ids:
            updated, skip_comment, skip_no_match = _process_image(
                conn, image_id, mapping, args.tolerance, dry_run=args.dry_run
            )
            total_updated += updated
            total_skip_comment += skip_comment
            total_skip_no_match += skip_no_match
            print(
                f"Image {image_id}: {verb}={updated} had_comment={skip_comment} no_match={skip_no_match}"
            )

        summary_verb = "would be updated" if args.dry_run else "updated"
        print(
            f"Done: {len(image_ids)} images, {total_updated} shapes {summary_verb}, "
            f"{total_skip_comment} already commented, {total_skip_no_match} unmatched."
        )
        if args.dry_run:
            print(
                "(dry run -- no changes were written to OMERO; re-run without --dry-run to apply)"
            )
    finally:
        conn.close()
