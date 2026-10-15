# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""``lavlab seg`` -- convert between DICOM SEG and NIfTI segmentation masks.

Local file conversion only; neither direction talks to OMERO.

    lavlab seg dcm2nii mask.dcm reference.nii.gz --out ./out/
    lavlab seg nii2dcm mask.nii.gz ./dicom_series/ --out ./mask.dcm
"""

from __future__ import annotations

import argparse
import logging

from lavlab.commands._shared import add_advanced_group, add_command, add_subcommands

# lavlab.seg pulls in highdicom, nibabel, pydicom and numpy. It is
# imported inside the handlers so that building the argument parser -- and
# therefore --help -- stays pure Python. seg touches neither vips nor OMERO,
# so it should not be able to fail because of a problem in either.

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    seg_parser = add_command(
        subparsers,
        "seg",
        help="Convert segmentation masks between DICOM SEG and NIfTI.",
        usage="%(prog)s COMMAND ...",
        description="Convert segmentation masks between DICOM SEG and NIfTI.\n"
        "Works on local files only; never connects to OMERO.",
    )
    seg_subparsers = add_subcommands(seg_parser, "seg_action")

    dcm2nii = add_command(
        seg_subparsers,
        "dcm2nii",
        help="Split a DICOM SEG file into one NIfTI mask per segment.",
        usage="%(prog)s DICOM_SEG REFERENCE_NIFTI -o DIR",
        description="Split a DICOM SEG file into one NIfTI mask per segment, "
        "aligned to\nREFERENCE_NIFTI (the NIfTI image the segmentation was drawn "
        "on).\nFiles are named <reference name>_<segment number>_<segment "
        "type>.nii.gz.",
        epilog="Example:\n  lavlab seg dcm2nii mask.dcm t2.nii.gz -o ./out/",
    )
    dcm2nii.add_argument(
        "dicom_seg", metavar="DICOM_SEG", help="The DICOM SEG file to split."
    )
    dcm2nii.add_argument(
        "reference_nifti",
        metavar="REFERENCE_NIFTI",
        help="The NIfTI image the segmentation applies to.",
    )
    dcm2nii.add_argument(
        "-o",
        "--out",
        required=True,
        dest="output_dir",
        metavar="DIR",
        help="Folder to write the NIfTI masks into (required).",
    )
    dcm2nii.set_defaults(handler=run_dcm2nii)

    nii2dcm = add_command(
        seg_subparsers,
        "nii2dcm",
        help="Turn a NIfTI label mask into a DICOM SEG file.",
        usage="%(prog)s NIFTI_MASK DICOM_DIR -o FILE [options]",
        description="Turn a NIfTI label mask into a DICOM SEG file that "
        "references the original\nDICOM series (DICOM_DIR) the mask was drawn on.",
        epilog="Example:\n  lavlab seg nii2dcm mask.nii.gz ./dicom_series/ -o mask.dcm",
    )
    nii2dcm.add_argument(
        "nifti_mask", metavar="NIFTI_MASK", help="The NIfTI label mask to convert."
    )
    nii2dcm.add_argument(
        "reference_dicom_dir",
        metavar="DICOM_DIR",
        help="Folder of .dcm files the mask was drawn against.",
    )
    nii2dcm.add_argument(
        "-o",
        "--out",
        required=True,
        dest="output_path",
        metavar="FILE",
        help="Where to write the DICOM SEG file (required).",
    )

    advanced = add_advanced_group(nii2dcm)
    advanced.add_argument(
        "--template",
        metavar="JSON",
        help="JSON template describing the segments for highdicom (default: the "
        "bundled template).",
    )
    advanced.add_argument(
        "--label",
        dest="segment_label",
        metavar="TEXT",
        help="Name for segment 1, replacing the template's.",
    )
    advanced.add_argument(
        "--comment",
        dest="patient_comment",
        metavar="TEXT",
        help="Text to store in the file's PatientComments field.",
    )
    nii2dcm.set_defaults(handler=run_nii2dcm)


def run_dcm2nii(args: argparse.Namespace) -> None:
    from lavlab.seg import dcmseg_to_nifti

    try:
        out_paths = dcmseg_to_nifti(
            args.dicom_seg, args.reference_nifti, args.output_dir
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    print(f"wrote {len(out_paths)} segment(s):")
    for path in out_paths:
        print(f"  {path}")


def run_nii2dcm(args: argparse.Namespace) -> None:
    from lavlab.seg import nifti_to_dcmseg

    try:
        out_path = nifti_to_dcmseg(
            args.nifti_mask,
            args.reference_dicom_dir,
            args.output_path,
            template_path=args.template,
            segment_label=args.segment_label,
            patient_comment=args.patient_comment,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    print(f"wrote {out_path}")
