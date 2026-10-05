# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Convert between DICOM SEG and NIfTI segmentation masks.

Two independent directions live here:

* :func:`dcmseg_to_nifti` -- split a DICOM SEG file into one NIfTI file per
  segment, aligned to a reference NIfTI image.
* :func:`nifti_to_dcmseg` -- write a single NIfTI label mask back out as a
  DICOM SEG object, using a reference DICOM series for the geometry/
  patient/study metadata a NIfTI file does not carry.

Both are local file conversions -- neither talks to OMERO -- and both raise
a specific exception (``FileNotFoundError``, ``ValueError``) on bad input
rather than letting ``pydicom``/``SimpleITK`` surface a confusing low-level
traceback.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Generator
from importlib import resources
from pathlib import Path

import highdicom
import highdicom.seg as hdseg
import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk  # noqa: N813
from pydicom.sr.coding import Code

log = logging.getLogger(__name__)


def _default_seg_template_path() -> Path:
    return Path(resources.files("lavlab.data").joinpath("default_seg_template.json"))


def _safe_path_component(text: str) -> str:
    """Reduce free text to something safe as one filename component.

    ``seg_name`` comes from a DICOM ``CodeMeaning`` -- coding scheme text
    the file's own author doesn't control the exact spelling of, and some
    schemes include slashes, parentheses, or other characters that would
    otherwise land in the output path as literal separators (silently
    creating an unintended nested directory) rather than as part of the
    filename.

    :param text: text to reduce to a safe path component
    :type text: str
    :return: a filesystem-safe stem
    :rtype: str
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text or "").strip("._-")
    return cleaned or "segment"


def format_output_path(output_dir: str, nii_name: str, seg_name: str) -> str:
    """Format the output path for one segment's NIfTI file.

    :param output_dir: destination directory
    :type output_dir: str
    :param nii_name: stem of the source NIfTI file
    :type nii_name: str
    :param seg_name: the segment's name
    :type seg_name: str
    :return: the formatted output path
    :rtype: str
    """
    return os.path.join(
        output_dir, f"{nii_name}_{_safe_path_component(seg_name)}.nii.gz"
    )


def read_nii(nii_path: str) -> sitk.Image:
    """Read a NIfTI file.

    :param nii_path: path to the NIfTI file
    :type nii_path: str
    :return: the loaded image
    :rtype: sitk.Image
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(nii_path)
    return reader.Execute()


def read_seg(dicom_seg_path: str) -> highdicom.seg.Segmentation:
    """Read a DICOM SEG file.

    :param dicom_seg_path: path to the DICOM SEG file
    :type dicom_seg_path: str
    :return: the DICOM SEG object
    :rtype: highdicom.seg.Segmentation
    """
    return highdicom.seg.segread(dicom_seg_path)


def get_affine_from_sitk(image: sitk.Image) -> np.ndarray:
    """Extract a NIfTI-style affine matrix from a SimpleITK image.

    :param image: the image to read geometry from
    :type image: sitk.Image
    :return: 4x4 affine matrix
    :rtype: np.ndarray
    """
    direction = image.GetDirection()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()

    direction_matrix = np.array(direction).reshape(3, 3)

    affine = np.eye(4)
    for i in range(3):
        for j in range(3):
            affine[i, j] = direction_matrix[i, j] * spacing[j]
    affine[:3, 3] = origin

    return affine


def flip_based_on_affine(seg_data: sitk.Image, ref_image: sitk.Image) -> sitk.Image:
    """Flip axes of ``seg_data`` where its orientation disagrees with ``ref_image``.

    :param seg_data: the image to (maybe) flip
    :type seg_data: sitk.Image
    :param ref_image: the image whose orientation is authoritative
    :type ref_image: sitk.Image
    :return: the correctly oriented image
    :rtype: sitk.Image
    """
    src_x, src_y, _ = nib.aff2axcodes(  # pylint: disable=unbalanced-tuple-unpacking
        get_affine_from_sitk(seg_data)
    )
    dest_x, dest_y, _ = nib.aff2axcodes(  # pylint: disable=unbalanced-tuple-unpacking
        get_affine_from_sitk(ref_image)
    )
    flips = [src_x != dest_x, src_y != dest_y, False]
    return sitk.Flip(seg_data, flips)


def format_nifti(seg_data: sitk.Image, ref_nii: sitk.Image) -> sitk.Image:
    """Orient a DICOM-SEG-derived image to match NIfTI conventions.

    :param seg_data: the segmentation channel to format
    :type seg_data: sitk.Image
    :param ref_nii: the reference NIfTI image
    :type ref_nii: sitk.Image
    :return: the formatted image
    :rtype: sitk.Image
    """
    seg_data = flip_based_on_affine(seg_data, ref_nii)
    # NIfTI uses opposite z indexing from DICOM.
    seg_data = sitk.Flip(seg_data, [False, False, True])
    return seg_data


def write_nifti(nii: sitk.Image, nii_path: str) -> None:
    """Write a NIfTI image to disk.

    :param nii: the image to write
    :type nii: sitk.Image
    :param nii_path: destination path
    :type nii_path: str
    """
    writer = sitk.ImageFileWriter()
    writer.SetFileName(nii_path)
    writer.Execute(nii)


def split_seg_channels(
    seg_data: highdicom.seg.Segmentation,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Split a DICOM SEG object into one array per segment.

    :param seg_data: the DICOM SEG object
    :type seg_data: highdicom.seg.Segmentation
    :return: ``(segment_number, pixel_array)`` pairs, in segment order
    :rtype: Generator[tuple[int, np.ndarray], None, None]
    """
    segment_numbers = list(seg_data.segment_numbers)
    log.info("Number of segments: %d", len(segment_numbers))
    uids = [x[2] for x in seg_data.get_source_image_uids()]
    for segment_number in segment_numbers:
        pixels = seg_data.get_pixels_by_source_instance(
            uids,
            segment_numbers=[segment_number],
            ignore_spatial_locations=True,
            assert_missing_frames_are_empty=True,
        )[..., 0]
        yield segment_number, pixels


def copy_sitk_image_info(src: sitk.Image, dst: sitk.Image) -> sitk.Image:
    """Copy spacing/origin/direction/metadata from one image to another.

    :param src: the image to copy from
    :type src: sitk.Image
    :param dst: the image to copy onto
    :type dst: sitk.Image
    :return: ``dst``, with copied information
    :rtype: sitk.Image
    """
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    for key in src.GetMetaDataKeys():
        dst.SetMetaData(key, src.GetMetaData(key))
    return dst


def dcmseg_to_nifti(dicom_seg_path: str, nii_path: str, output_dir: str) -> list[str]:
    """Split a DICOM SEG file into one NIfTI file per segment.

    :param dicom_seg_path: path to the DICOM SEG file
    :type dicom_seg_path: str
    :param nii_path: path to the reference NIfTI image the segmentation applies to
    :type nii_path: str
    :param output_dir: directory to write the per-segment NIfTI files into
    :type output_dir: str
    :raises FileNotFoundError: if the DICOM SEG or reference NIfTI is missing
    :raises ValueError: if a segment's pixel array does not match the reference image's size
    :return: the written file paths, one per segment
    :rtype: list[str]
    """
    if not os.path.isfile(dicom_seg_path):
        raise FileNotFoundError(f"DICOM SEG file not found: {dicom_seg_path}")
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"reference NIfTI file not found: {nii_path}")
    os.makedirs(output_dir, exist_ok=True)

    dicom_seg = read_seg(dicom_seg_path)  # (z, x, y)
    nii = read_nii(nii_path)  # (x, y, z)
    try:
        sitk_dcm_seg = sitk.ReadImage(dicom_seg_path)
    except RuntimeError:
        log.info(
            "SimpleITK could not read geometry directly from %s "
            "(expected for LABELMAP-type SEGs); using the reference "
            "NIfTI's geometry instead.",
            dicom_seg_path,
        )
        sitk_dcm_seg = nii

    segments_by_number = {
        segment.SegmentNumber: segment for segment in dicom_seg.SegmentSequence
    }

    out_paths = []
    for segment_number, seg_channel in split_seg_channels(dicom_seg):
        nii_out = sitk.GetImageFromArray(seg_channel)
        if nii_out.GetSize() != nii.GetSize():
            raise ValueError(
                f"segmentation size {nii_out.GetSize()} does not match "
                f"NIfTI size {nii.GetSize()}"
            )
        nii_out = copy_sitk_image_info(sitk_dcm_seg, nii_out)
        nii_out = format_nifti(nii_out, nii)
        segment = segments_by_number.get(segment_number)
        if segment is None:
            raise ValueError(
                f"DICOM SEG has no SegmentSequence entry for segment number "
                f"{segment_number}"
            )
        nii_code = segment.SegmentedPropertyTypeCodeSequence[0].CodeMeaning
        nii_basename = os.path.basename(nii_path).split(".")[0]
        channel_output_path = format_output_path(output_dir, nii_basename, nii_code)
        write_nifti(nii_out, channel_output_path)
        out_paths.append(channel_output_path)

    return out_paths


def _load_segment_descriptions(
    template: dict, segment_label_override: str | None
) -> list[hdseg.SegmentDescription]:
    """Build highdicom segment descriptions from the JSON template.

    :param template: the parsed segment-attributes template
    :type template: dict
    :param segment_label_override: replaces segment 1's label if given
    :type segment_label_override: str | None
    :raises ValueError: if the template defines no segments
    :return: one description per segment, numbered from 1
    :rtype: list[hdseg.SegmentDescription]
    """
    segments = template.get("segments") or []
    if not segments:
        raise ValueError("segment-attributes template defines no segments")

    descriptions = []
    for i, segment in enumerate(segments, start=1):
        category = segment["category"]
        seg_type = segment["type"]
        label = segment.get("label", f"Segment {i}")
        if i == 1 and segment_label_override:
            label = segment_label_override
        descriptions.append(
            hdseg.SegmentDescription(
                segment_number=i,
                segment_label=label,
                segmented_property_category=Code(
                    category["value"], category["scheme"], category["meaning"]
                ),
                segmented_property_type=Code(
                    seg_type["value"], seg_type["scheme"], seg_type["meaning"]
                ),
                algorithm_type=segment.get("algorithm_type", "MANUAL"),
            )
        )
    return descriptions


def _read_series_in_slice_order(
    dicom_series_paths: list[Path],
) -> list[pydicom.Dataset]:
    """Read a DICOM series and sort it into ascending slice order.

    Sorted by ``ImagePositionPatient``'s through-plane component when every
    instance has one, falling back to ``InstanceNumber``. The DICOM SEG's
    per-frame pixel array must line up 1:1 with this order.

    :param dicom_series_paths: the ``.dcm`` files making up the series
    :type dicom_series_paths: list[Path]
    :raises ValueError: if instances carry neither ``ImagePositionPatient`` nor ``InstanceNumber``
    :return: the datasets, in slice order
    :rtype: list[pydicom.Dataset]
    """
    datasets = [pydicom.dcmread(str(p)) for p in dicom_series_paths]

    if all("ImagePositionPatient" in ds for ds in datasets):
        datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    elif all("InstanceNumber" in ds for ds in datasets):
        datasets.sort(key=lambda ds: int(ds.InstanceNumber))
    else:
        raise ValueError(
            "reference DICOM series has instances with neither "
            "ImagePositionPatient nor InstanceNumber; cannot determine slice order"
        )
    return datasets


def nifti_to_dcmseg(
    nifti_mask_path: str,
    reference_dicom_dir: str,
    output_path: str,
    template_path: str | None = None,
    segment_label: str | None = None,
    patient_comment: str | None = None,
) -> str:
    """Write a NIfTI label mask out as a DICOM SEG object.

    :param nifti_mask_path: path to the NIfTI label mask
    :type nifti_mask_path: str
    :param reference_dicom_dir: directory of ``.dcm`` files the mask was drawn against
    :type reference_dicom_dir: str
    :param output_path: path to write the DICOM SEG object to
    :type output_path: str
    :param template_path: segment-attributes JSON template; defaults to the bundled template
    :type template_path: str | None
    :param segment_label: overrides the template's label for segment 1
    :type segment_label: str | None
    :param patient_comment: sets ``PatientComments`` on the written object
    :type patient_comment: str | None
    :raises FileNotFoundError: if the mask, reference directory, or template is missing,
        or the reference directory has no ``.dcm`` files
    :raises ValueError: if the template defines no segments, the reference series
        cannot be ordered into slices, or the mask's slice count does not match
        the reference series
    :return: the path written to
    :rtype: str
    """
    mask_path = Path(nifti_mask_path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"NIfTI mask not found: {mask_path}")

    dicom_dir = Path(reference_dicom_dir)
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"reference DICOM directory not found: {dicom_dir}")
    dicom_series_paths = sorted(p for p in dicom_dir.iterdir() if p.suffix == ".dcm")
    if not dicom_series_paths:
        raise FileNotFoundError(f"no .dcm files found in {dicom_dir}")

    resolved_template = (
        Path(template_path) if template_path else _default_seg_template_path()
    )
    if not resolved_template.is_file():
        raise FileNotFoundError(
            f"segment-attributes template not found: {resolved_template}"
        )
    template = json.loads(resolved_template.read_text(encoding="utf-8"))

    segmentation = sitk.ReadImage(str(mask_path))
    segmentation = sitk.Cast(segmentation, sitk.sitkUInt8)
    mask_array = sitk.GetArrayFromImage(segmentation)  # (z, y, x)
    source_images = _read_series_in_slice_order(dicom_series_paths)
    if len(source_images) != mask_array.shape[0]:
        raise ValueError(
            f"reference DICOM series has {len(source_images)} slice(s) but the "
            f"NIfTI mask has {mask_array.shape[0]}; they must match 1:1"
        )

    segment_descriptions = _load_segment_descriptions(template, segment_label)

    seg_obj = hdseg.Segmentation(
        source_images=source_images,
        pixel_array=mask_array,
        segmentation_type=hdseg.SegmentationTypeValues.LABELMAP,
        segment_descriptions=segment_descriptions,
        series_instance_uid=highdicom.UID(),
        series_number=template.get("series_number", 1),
        sop_instance_uid=highdicom.UID(),
        instance_number=template.get("instance_number", 1),
        manufacturer=template.get("manufacturer", "LavLab"),
        manufacturer_model_name=template.get(
            "manufacturer_model_name", "lavlab-cli-utils"
        ),
        software_versions=template.get("software_versions", "0.1.0"),
        device_serial_number=template.get("device_serial_number", "NA"),
        content_label=template.get("content_label", "SEGMENTATION"),
        content_description=template.get("content_description", "Image segmentation"),
    )
    seg_obj.SeriesDescription = template.get("series_description", "Segmentation")
    if template.get("body_part_examined"):
        seg_obj.BodyPartExamined = template["body_part_examined"]
    if patient_comment:
        seg_obj.PatientComments = patient_comment

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg_obj.save_as(str(out_path))
    return str(out_path)


__all__ = [
    "copy_sitk_image_info",
    "dcmseg_to_nifti",
    "flip_based_on_affine",
    "format_nifti",
    "format_output_path",
    "get_affine_from_sitk",
    "nifti_to_dcmseg",
    "read_nii",
    "read_seg",
    "split_seg_channels",
    "write_nifti",
]
