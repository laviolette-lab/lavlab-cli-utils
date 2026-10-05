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

# lavlab.seg pulls in SimpleITK, highdicom, nibabel, pydicom and numpy. It is
# imported inside the handlers so that building the argument parser -- and
# therefore --help -- stays pure Python. seg touches neither vips nor OMERO,
# so it should not be able to fail because of a problem in either.

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    seg_parser = subparsers.add_parser(
        "seg", help="Convert between DICOM SEG and NIfTI segmentation masks."
    )
    seg_subparsers = seg_parser.add_subparsers(dest="seg_action", required=True)

    dcm2nii = seg_subparsers.add_parser(
        "dcm2nii", help="split a DICOM SEG file into one NIfTI file per segment"
    )
    dcm2nii.add_argument("dicom_seg", help="path to the DICOM SEG file")
    dcm2nii.add_argument(
        "reference_nifti",
        help="path to the reference NIfTI image the segmentation applies to",
    )
    dcm2nii.add_argument(
        "-o",
        "--out",
        required=True,
        dest="output_dir",
        help="directory to write the per-segment NIfTI files into",
    )
    dcm2nii.set_defaults(handler=run_dcm2nii)

    nii2dcm = seg_subparsers.add_parser(
        "nii2dcm", help="write a NIfTI label mask out as a DICOM SEG object"
    )
    nii2dcm.add_argument("nifti_mask", help="path to the NIfTI label mask")
    nii2dcm.add_argument(
        "reference_dicom_dir", help="directory of .dcm files the mask was drawn against"
    )
    nii2dcm.add_argument(
        "-o",
        "--out",
        required=True,
        dest="output_path",
        help="path to write the DICOM SEG object to",
    )
    nii2dcm.add_argument(
        "--template",
        help="segment-attributes JSON template for highdicom (default: bundled template)",
    )
    nii2dcm.add_argument(
        "--label", dest="segment_label", help="overrides the template's segment 1 label"
    )
    nii2dcm.add_argument(
        "--comment",
        dest="patient_comment",
        help="overrides PatientComments on the written object",
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
