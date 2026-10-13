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
rather than letting ``pydicom`` surface a confusing low-level traceback.
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
from nibabel.spatialimages import SpatialImage
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


def format_output_path(
    output_dir: str,
    nii_name: str,
    seg_name: str,
    segment_number: int | None = None,
) -> str:
    """Format the output path for one segment's NIfTI file.

    :param output_dir: destination directory
    :type output_dir: str
    :param nii_name: stem of the source NIfTI file
    :type nii_name: str
    :param seg_name: the segment's name
    :type seg_name: str
    :param segment_number: optional DICOM segment number to disambiguate names
    :type segment_number: int | None
    :return: the formatted output path
    :rtype: str
    """
    number = f"{segment_number}_" if segment_number is not None else ""
    return os.path.join(
        output_dir, f"{nii_name}_{number}{_safe_path_component(seg_name)}.nii.gz"
    )


def read_nii(nii_path: str) -> SpatialImage:
    """Read a NIfTI file."""
    return nib.load(nii_path)


def read_seg(dicom_seg_path: str) -> highdicom.seg.Segmentation:
    """Read a DICOM SEG file.

    :param dicom_seg_path: path to the DICOM SEG file
    :type dicom_seg_path: str
    :return: the DICOM SEG object
    :rtype: highdicom.seg.Segmentation
    """
    return highdicom.seg.segread(dicom_seg_path)


def get_affine_from_nifti(image: SpatialImage) -> np.ndarray:
    """Return the image's voxel-to-world affine matrix."""
    return np.asarray(image.affine)


def _get_dicom_seg_affine(
    seg_data: pydicom.Dataset, source_uids: list[str] | None = None
) -> np.ndarray | None:
    """Read DICOM SEG geometry in the source-stack voxel order."""
    per_frame = getattr(seg_data, "PerFrameFunctionalGroupsSequence", [])
    shared = getattr(seg_data, "SharedFunctionalGroupsSequence", [])

    def first_item(sequence_name: str):
        for group in [*per_frame[:1], *shared[:1]]:
            sequence = getattr(group, sequence_name, None)
            if sequence:
                return sequence[0]
        return None

    orientation = first_item("PlaneOrientationSequence")
    measures = first_item("PixelMeasuresSequence")
    if orientation is None or measures is None:
        return None

    iop = np.asarray(orientation.ImageOrientationPatient, dtype=float)
    pixel_spacing = np.asarray(measures.PixelSpacing, dtype=float)
    if iop.shape != (6,) or pixel_spacing.shape != (2,):
        return None

    row_direction, column_direction = iop[:3], iop[3:]
    slice_direction = np.cross(row_direction, column_direction)
    slice_spacing = float(
        getattr(measures, "SpacingBetweenSlices", None)
        or getattr(measures, "SliceThickness", 1.0)
    )
    slice_step = slice_direction * slice_spacing
    first_position = None
    if source_uids:
        indexes, positions = [], []
        uid_indexes = {uid: index for index, uid in enumerate(source_uids)}
        for group in per_frame:
            position = getattr(group, "PlanePositionSequence", None)
            derivation = getattr(group, "DerivationImageSequence", None)
            if not position or not derivation:
                continue
            source_images = getattr(derivation[0], "SourceImageSequence", None)
            if not source_images:
                continue
            index = uid_indexes.get(source_images[0].ReferencedSOPInstanceUID)
            if index is not None:
                indexes.append(index)
                positions.append(
                    np.asarray(position[0].ImagePositionPatient, dtype=float)
                )
        if indexes and len(set(indexes)) > 1:
            design = np.column_stack(
                (np.asarray(indexes, dtype=float), np.ones(len(indexes)))
            )
            slice_step, first_position = np.linalg.lstsq(
                design, np.asarray(positions), rcond=None
            )[0]

    if first_position is None:
        position = first_item("PlanePositionSequence")
        if position is None:
            return None
        first_position = np.asarray(position.ImagePositionPatient, dtype=float)
        if source_uids and per_frame:
            derivation = getattr(per_frame[0], "DerivationImageSequence", None)
            source_images = (
                getattr(derivation[0], "SourceImageSequence", None)
                if derivation
                else None
            )
            if source_images:
                source_index = uid_indexes.get(
                    source_images[0].ReferencedSOPInstanceUID
                )
                if source_index is not None:
                    first_position -= slice_step * source_index

    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_direction * pixel_spacing[1]
    affine_lps[:3, 1] = column_direction * pixel_spacing[0]
    affine_lps[:3, 2] = slice_step
    affine_lps[:3, 3] = first_position

    # DICOM patient coordinates are LPS; NIfTI affine world coordinates are RAS.
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ affine_lps


def _flip_affine_axis(affine: np.ndarray, axis: int, length: int) -> np.ndarray:
    """Update an affine for a voxel-axis reversal while preserving world positions."""
    transform = np.eye(4)
    transform[axis, axis] = -1
    transform[axis, 3] = length - 1
    return affine @ transform


def flip_based_on_affine(
    seg_data: np.ndarray, seg_affine: np.ndarray, ref_affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flip voxel axes when SEG and reference orientations differ."""
    source_codes = nib.aff2axcodes(seg_affine)
    reference_codes = nib.aff2axcodes(ref_affine)
    affine = np.asarray(seg_affine).copy()

    for axis in range(3):
        if source_codes[axis] != reference_codes[axis]:
            seg_data = np.flip(seg_data, axis=axis)
            affine = _flip_affine_axis(affine, axis, seg_data.shape[axis])

    return seg_data, affine


def format_nifti(
    seg_data: np.ndarray, seg_affine: np.ndarray, ref_affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Orient DICOM SEG voxels and geometry to match the reference NIfTI."""
    return flip_based_on_affine(seg_data, seg_affine, ref_affine)


def write_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    nii_path: str,
    header: nib.Nifti1Header | None = None,
) -> None:
    """Write a voxel array and affine matrix to a NIfTI file."""
    output_header = header.copy() if header is not None else None
    if output_header is not None:
        output_header.set_data_dtype(data.dtype)
    image = nib.Nifti1Image(data, affine, header=output_header)
    if output_header is not None:
        image.set_qform(affine, code=int(output_header["qform_code"]))
        image.set_sform(image.get_qform(), code=int(output_header["sform_code"]))
    nib.save(image, nii_path)


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

    dicom_seg = read_seg(dicom_seg_path)  # pixel channels are (z, y, x)
    nii = read_nii(nii_path)  # nibabel voxel arrays are (x, y, z)
    source_uids = [item[2] for item in dicom_seg.get_source_image_uids()]
    if len(nii.shape) != 3:
        raise ValueError(f"reference NIfTI must be 3D, got shape {nii.shape}")
    reference_affine = get_affine_from_nifti(nii)
    seg_affine = _get_dicom_seg_affine(dicom_seg, source_uids)
    if seg_affine is None:
        log.info(
            "DICOM SEG geometry is unavailable; using the reference NIfTI's geometry."
        )
        seg_affine = get_affine_from_nifti(nii)

    segments_by_number = {
        segment.SegmentNumber: segment for segment in dicom_seg.SegmentSequence
    }

    out_paths = []
    nii_basename = Path(nii_path).name
    for suffix in (".nii.gz", ".nii"):
        if nii_basename.lower().endswith(suffix):
            nii_basename = nii_basename[: -len(suffix)]
            break

    for segment_number, seg_channel in split_seg_channels(dicom_seg):
        segment = segments_by_number.get(segment_number)
        if segment is None:
            raise ValueError(
                f"DICOM SEG has no SegmentSequence entry for segment number "
                f"{segment_number}"
            )
        nii_data = np.transpose(seg_channel, (2, 1, 0))
        if nii_data.shape != nii.shape[:3]:
            raise ValueError(
                f"segmentation size {nii_data.shape} does not match "
                f"NIfTI size {nii.shape[:3]}"
            )
        nii_data, output_affine = format_nifti(
            nii_data, seg_affine, reference_affine
        )
        nii_code = segment.SegmentedPropertyTypeCodeSequence[0].CodeMeaning
        channel_output_path = format_output_path(
            output_dir, nii_basename, nii_code, segment_number
        )
        write_nifti(nii_data, output_affine, channel_output_path, nii.header)
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

    segmentation = read_nii(str(mask_path))
    if len(segmentation.shape) != 3:
        raise ValueError(f"NIfTI mask must be 3D, got shape {segmentation.shape}")
    mask_xyz = np.asanyarray(segmentation.dataobj).astype(np.uint8, copy=False)
    mask_array = np.transpose(mask_xyz, (2, 1, 0))  # (z, y, x)
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
    "dcmseg_to_nifti",
    "flip_based_on_affine",
    "format_nifti",
    "format_output_path",
    "get_affine_from_nifti",
    "nifti_to_dcmseg",
    "read_nii",
    "read_seg",
    "split_seg_channels",
    "write_nifti",
]
