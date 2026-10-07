# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for lavlab.seg -- skipped if pydicom isn't installed."""

import os
import subprocess

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pydicom = pytest.importorskip("pydicom")

from lavlab.seg import (  # noqa: E402
    dcmseg_to_nifti,
    flip_based_on_affine,
    format_output_path,
    get_affine_from_nifti,
    nifti_to_dcmseg,
)


def _image(size=(4, 4, 2), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    arr = np.zeros(size, dtype=np.uint8)
    affine = np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])
    affine[:3, 3] = [-origin[0], -origin[1], origin[2]]
    return nib.Nifti1Image(arr, affine)


def _write_nii_from_zyx(path, data):
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(data.transpose(2, 1, 0), affine), path)


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
    _write_nii_from_zyx(mask_path, mask)

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

    recovered_xyz = np.asanyarray(nib.load(out_paths[0]).dataobj)
    recovered = recovered_xyz.transpose(2, 1, 0)
    assert np.array_equal(recovered > 0, mask > 0)


def test_onefile_binary_seg_conversions_when_configured(tmp_path):
    binary = os.environ.get("LAVLAB_BINARY")
    if not binary:
        pytest.skip("LAVLAB_BINARY is not set")

    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    _write_synthetic_series(dicom_dir)
    mask = np.zeros((3, 8, 8), dtype=np.uint8)
    mask[1, 2:4, 3:5] = 1
    mask_path = tmp_path / "mask.nii.gz"
    _write_nii_from_zyx(mask_path, mask)
    seg_path = tmp_path / "seg.dcm"

    subprocess.run(
        [
            binary,
            "seg",
            "nii2dcm",
            str(mask_path),
            str(dicom_dir),
            "--out",
            str(seg_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert pydicom.dcmread(seg_path).Modality == "SEG"

    output_dir = tmp_path / "nifti"
    subprocess.run(
        [
            binary,
            "seg",
            "dcm2nii",
            str(seg_path),
            str(mask_path),
            "--out",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    outputs = list(output_dir.glob("*.nii.gz"))
    assert len(outputs) == 1
    assert nib.load(outputs[0]).shape == (8, 8, 3)


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


def test_get_affine_from_nifti_returns_image_affine():
    img = _image()
    affine = get_affine_from_nifti(img)
    assert affine.shape == (4, 4)
    assert np.array_equal(affine, img.affine)


def test_flip_based_on_affine_same_orientation_is_noop():
    img = _image()
    data = np.arange(4 * 4 * 2).reshape((4, 4, 2))
    flipped, affine = flip_based_on_affine(data, img.affine, img.affine)
    assert np.array_equal(flipped, data)
    assert np.array_equal(affine, img.affine)


def test_flip_based_on_affine_reverses_mismatched_axis_and_updates_affine():
    ref = _image()
    source_affine = ref.affine.copy()
    source_affine[0, 0] *= -1
    data = np.arange(4 * 4 * 2).reshape((4, 4, 2))

    flipped, affine = flip_based_on_affine(data, source_affine, ref.affine)

    assert np.array_equal(flipped, np.flip(data, axis=0))
    assert nib.aff2axcodes(affine) == nib.aff2axcodes(ref.affine)


def test_dcmseg_to_nifti_missing_dicom_seg_raises(tmp_path):
    ref_nii = tmp_path / "ref.nii.gz"
    nib.save(_image(), ref_nii)

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
    nib.save(_image(), mask)

    with pytest.raises(FileNotFoundError):
        nifti_to_dcmseg(
            str(mask),
            str(tmp_path / "no_such_dir"),
            str(tmp_path / "out.dcm"),
        )


def test_nifti_to_dcmseg_empty_reference_dir_raises(tmp_path):
    mask = tmp_path / "mask.nii.gz"
    nib.save(_image(), mask)
    empty_dir = tmp_path / "dicom"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        nifti_to_dcmseg(str(mask), str(empty_dir), str(tmp_path / "out.dcm"))
