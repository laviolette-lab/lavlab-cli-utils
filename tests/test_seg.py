# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for lavlab.seg -- skipped if the heavy imaging deps aren't installed
(they're in the `dev` extra, not required just to import the rest of the package)."""

import os

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pydicom = pytest.importorskip("pydicom")

from lavlab.seg import (  # noqa: E402
    dcmseg_to_nifti,
    flip_based_on_affine,
    format_output_path,
    get_affine_from_sitk,
    nifti_to_dcmseg,
)


def _image(size=(4, 4, 2), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    arr = np.zeros(size[::-1], dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def _write_synthetic_series(dicom_dir, rows=8, cols=8, slices=3):
    """Write a minimal-but-valid CT series, one file per slice."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    series_uid = generate_uid()
    study_uid = generate_uid()
    frame_of_reference_uid = generate_uid()
    sop_class = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage

    for z in range(slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = sop_class
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        path = str(dicom_dir / f"slice{z}.dcm")
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.Modality = "CT"
        ds.PatientID = "TEST"
        ds.PatientName = "Test^Patient"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.StudyDate = "20260101"
        ds.StudyTime = "000000"
        ds.StudyID = "1"
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.SeriesNumber = 1
        ds.Rows = rows
        ds.Columns = cols
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImagePositionPatient = [0.0, 0.0, float(z)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.InstanceNumber = z + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = np.zeros((rows, cols), dtype=np.int16).tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(path, enforce_file_format=True)


def test_nifti_to_dcmseg_round_trips_through_dcmseg_to_nifti(tmp_path):
    """A mask written out as DICOM SEG and read back must match exactly."""
    rows, cols, slices = 8, 8, 3
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    _write_synthetic_series(dicom_dir, rows, cols, slices)

    mask = np.zeros((slices, rows, cols), dtype=np.uint8)
    mask[1, 2:5, 2:5] = 1
    mask_path = tmp_path / "mask.nii.gz"
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.SetSpacing((1.0, 1.0, 1.0))
    mask_img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(mask_img, str(mask_path))

    seg_path = tmp_path / "out_seg.dcm"
    nifti_to_dcmseg(
        str(mask_path),
        str(dicom_dir),
        str(seg_path),
        segment_label="TestSeg",
        patient_comment="pytest round trip",
    )

    written = pydicom.dcmread(str(seg_path))
    assert written.Modality == "SEG"
    assert written.PatientComments == "pytest round trip"

    out_dir = tmp_path / "roundtrip"
    out_paths = dcmseg_to_nifti(str(seg_path), str(mask_path), str(out_dir))
    assert len(out_paths) == 1

    recovered = sitk.GetArrayFromImage(sitk.ReadImage(out_paths[0]))
    assert np.array_equal(recovered > 0, mask > 0)


def test_format_output_path():
    path = format_output_path("/out", "case01", "Prostate")
    assert path == "/out/case01_Prostate.nii.gz"


def test_format_output_path_sanitizes_unsafe_segment_name():
    # A DICOM CodeMeaning can contain slashes/parens/etc.; those must not
    # land in the path as literal separators (an unintended nested dir)
    # or otherwise break the write.
    path = format_output_path("/out", "case01", "Gleason 3+4=7 (prostate/left)")
    assert "/" not in os.path.relpath(path, "/out")
    assert path.startswith("/out/case01_")
    assert path.endswith(".nii.gz")


def test_format_output_path_empty_segment_name_falls_back():
    path = format_output_path("/out", "case01", "///")
    assert path == "/out/case01_segment.nii.gz"


def test_get_affine_from_sitk_identity():
    img = _image()
    affine = get_affine_from_sitk(img)
    assert affine.shape == (4, 4)
    assert np.allclose(affine, np.eye(4))


def test_flip_based_on_affine_same_orientation_is_noop():
    img = _image()
    ref = _image()
    flipped = flip_based_on_affine(img, ref)
    assert flipped.GetSize() == img.GetSize()


def test_dcmseg_to_nifti_missing_dicom_seg_raises(tmp_path):
    ref_nii = tmp_path / "ref.nii.gz"
    sitk.WriteImage(_image(), str(ref_nii))

    with pytest.raises(FileNotFoundError):
        dcmseg_to_nifti(
            str(tmp_path / "missing.dcm"), str(ref_nii), str(tmp_path / "out")
        )


def test_dcmseg_to_nifti_missing_reference_nifti_raises(tmp_path):
    fake_seg = tmp_path / "seg.dcm"
    fake_seg.write_bytes(b"not a real dicom seg")

    with pytest.raises(FileNotFoundError):
        dcmseg_to_nifti(
            str(fake_seg), str(tmp_path / "missing.nii.gz"), str(tmp_path / "out")
        )


def test_nifti_to_dcmseg_missing_mask_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        nifti_to_dcmseg(
            str(tmp_path / "missing.nii.gz"),
            str(tmp_path),
            str(tmp_path / "out.dcm"),
        )


def test_nifti_to_dcmseg_missing_reference_dir_raises(tmp_path):
    mask = tmp_path / "mask.nii.gz"
    sitk.WriteImage(_image(), str(mask))

    with pytest.raises(FileNotFoundError):
        nifti_to_dcmseg(
            str(mask),
            str(tmp_path / "no_such_dir"),
            str(tmp_path / "out.dcm"),
        )


def test_nifti_to_dcmseg_empty_reference_dir_raises(tmp_path):
    mask = tmp_path / "mask.nii.gz"
    sitk.WriteImage(_image(), str(mask))
    empty_dir = tmp_path / "dicom"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        nifti_to_dcmseg(str(mask), str(empty_dir), str(tmp_path / "out.dcm"))
