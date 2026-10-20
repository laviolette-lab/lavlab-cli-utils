# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Batch reporting and exit-code rules for ``lavlab lr batch``.

The failure mode these guard against: a scheduled run that processes
thousands of images entirely over the slow network tier -- or fails every
single one -- and still exits 0, so the CronJob goes green and nobody
notices.
"""

from __future__ import annotations

import argparse

import pytest

from lavlab.cli import build_parser
from lavlab.commands import lr as lr_cmd
from lavlab.commands.lr import _keep_existing, _summarise_batch, _validate_args


def _ok(image_id, tier="local"):
    return (image_id, f"/out/{image_id}.jp2", tier)


def _skipped(image_id):
    return (image_id, f"/out/{image_id}.jp2", None)


def _tier_line(caplog):
    lines = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    matches = [line for line in lines if line.startswith("Served by tier:")]
    return matches[0] if matches else None


# --- tier breakdown -------------------------------------------------------


def test_tier_breakdown_logged_at_info(caplog):
    ids = [1, 2, 3, 4]
    results = [
        _ok(1, "local"),
        _ok(2, "local"),
        _ok(3, "network"),
        _ok(4, "annotation"),
    ]

    with caplog.at_level("INFO"):
        _summarise_batch(ids, results, None)

    assert _tier_line(caplog) == "Served by tier: annotation=1, local=2, network=1"


def test_tier_breakdown_makes_a_silent_all_network_run_visible(caplog):
    """The case that motivated this: everything "succeeded", slowly."""
    ids = list(range(50))
    results = [_ok(i, "network") for i in ids]

    with caplog.at_level("INFO"):
        _summarise_batch(ids, results, None)

    assert _tier_line(caplog) == "Served by tier: network=50"


def test_skipped_images_counted_separately_from_tiers(caplog):
    ids = [1, 2, 3]
    results = [_ok(1, "local"), _skipped(2), _skipped(3)]

    with caplog.at_level("INFO"):
        _summarise_batch(ids, results, None)

    # Skips did no work, so they must not be attributed to a tier.
    assert _tier_line(caplog) == "Served by tier: local=1, skipped=2"


def test_summary_counts_failures(caplog):
    ids = [1, 2, 3, 4]
    results = [_ok(1), None, _ok(3), None]

    with caplog.at_level("INFO"):
        _summarise_batch(ids, results, None)

    msgs = [r.getMessage() for r in caplog.records]
    assert "Batch complete: 2/4 images processed, 2 failed." in msgs


# --- exit code ------------------------------------------------------------


def test_all_failed_exits_non_zero():
    ids = [1, 2, 3]
    with pytest.raises(SystemExit) as excinfo:
        _summarise_batch(ids, [None, None, None], None)
    assert "all 3 images failed" in str(excinfo.value)


def test_partial_failure_exits_zero_by_default():
    """One bad slide in a batch must not fail a scheduled run."""
    ids = list(range(100))
    results = [None] + [_ok(i) for i in ids[1:]]
    _summarise_batch(ids, results, None)  # must not raise


def test_max_failed_threshold_exceeded_exits_non_zero():
    ids = [1, 2, 3, 4]
    results = [None, None, None, _ok(4)]
    with pytest.raises(SystemExit) as excinfo:
        _summarise_batch(ids, results, 2)
    assert "3 of 4 images failed" in str(excinfo.value)
    assert "--max-failed 2" in str(excinfo.value)


def test_max_failed_threshold_met_exactly_exits_zero():
    ids = [1, 2, 3]
    _summarise_batch(ids, [None, _ok(2), _ok(3)], 1)  # 1 failed, threshold 1


def test_max_failed_zero_fails_on_any_error():
    ids = [1, 2]
    with pytest.raises(SystemExit):
        _summarise_batch(ids, [None, _ok(2)], 0)


def test_empty_batch_exits_zero():
    """No images found is not the same as every image failing."""
    _summarise_batch([], [], None)


def test_fully_successful_batch_exits_zero():
    ids = [1, 2]
    _summarise_batch(ids, [_ok(1), _ok(2)], 0)


# --- argument validation --------------------------------------------------


def _args(**kw):
    base = dict(
        target="batch",
        skip_existing=False,
        regenerate=False,
        skip_local=False,
        skip_upload=False,
        output=None,
        max_failed=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_max_failed_rejected_outside_batch_mode():
    with pytest.raises(SystemExit) as excinfo:
        _validate_args(_args(target="2607", max_failed=5))
    assert "batch-mode" in str(excinfo.value)


def test_max_failed_rejects_negative():
    with pytest.raises(SystemExit) as excinfo:
        _validate_args(_args(max_failed=-1))
    assert "zero or greater" in str(excinfo.value)


def test_max_failed_zero_is_accepted_in_batch_mode():
    _validate_args(_args(max_failed=0))


# --- existing output files ------------------------------------------------


@pytest.mark.parametrize(
    ("override", "regenerate", "kept"),
    [
        (False, False, True),
        (True, False, False),
        # Regenerating and then skipping the write would do nothing at all.
        (False, True, False),
    ],
)
def test_existing_output_kept_unless_override_or_regenerate(
    tmp_path, override, regenerate, kept
):
    existing = tmp_path / "LR10_slide.jp2"
    existing.write_bytes(b"old")

    args = _args(override=override, regenerate=regenerate)

    assert _keep_existing(args, str(existing)) is kept


def test_missing_output_is_never_kept(tmp_path):
    assert not _keep_existing(_args(override=False), str(tmp_path / "absent.jp2"))


@pytest.mark.parametrize(
    "argv",
    [
        ["lr", "12345", "--regenerate", "--override"],
        ["--override", "lr", "12345", "--regenerate"],
    ],
    ids=["after the command", "before the command"],
)
def test_override_accepted_on_either_side_of_the_command(argv):
    assert build_parser().parse_args(argv).override is True


@pytest.mark.parametrize("command", [["lr", "1"], ["roi", "1", "--all"]])
def test_override_defaults_off(command):
    assert build_parser().parse_args(command).override is False


# --- existing output files, end to end -----------------------------------
#
# The bug these guard: 'lr ID --regenerate' with the output already on disk
# skipped the image and told the user to pass an --override flag that the
# lr subcommand didn't accept.


class _FakeImage:
    def __init__(self, image_id):
        self._id = image_id

    def getId(self):
        return self._id

    def getName(self):
        return "slide.ome.tiff"


class _FakeOpts:
    def setOmeroGroup(self, group):
        self.group = group


class _FakeConn:
    def __init__(self):
        self.SERVICE_OPTS = _FakeOpts()

    def isConnected(self):
        return True

    def getObject(self, kind, image_id):
        return _FakeImage(image_id)

    def close(self):
        pass


@pytest.fixture
def fake_lr(monkeypatch, tmp_path):
    """Stand in for OMERO; record which images actually got rendered."""
    rendered = []
    monkeypatch.setattr(lr_cmd, "connect_from_args", lambda args: _FakeConn())
    monkeypatch.setattr(lr_cmd, "group_of", lambda conn, image: 3)
    monkeypatch.setattr(lr_cmd, "load_fs_map_from_args", lambda args: {})
    monkeypatch.setattr(
        lr_cmd,
        "_render_and_write",
        lambda conn, image, args, path: (
            rendered.append((image.getId(), path)) or "network"
        ),
    )
    existing = tmp_path / "LR10_slide.jp2"
    existing.write_bytes(b"old")
    return rendered, existing


def _run_args(**kw):
    return _args(**{"downsample": 10, "format": "jp2", "fs_map": None, **kw})


@pytest.mark.parametrize(
    ("override", "regenerate", "rebuilt"),
    [(False, False, False), (True, False, True), (False, True, True)],
    ids=["plain run skips", "--override replaces", "--regenerate replaces"],
)
def test_single_image_with_output_on_disk(
    fake_lr, capsys, override, regenerate, rebuilt
):
    rendered, existing = fake_lr
    args = _run_args(
        target="7", output=str(existing), override=override, regenerate=regenerate
    )

    lr_cmd._run_single(args, 7)

    assert rendered == ([(7, str(existing))] if rebuilt else [])
    if not rebuilt:
        assert "use --override to replace" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("override", "regenerate", "rebuilt"),
    [(False, False, False), (True, False, True), (False, True, True)],
    ids=["plain run skips", "--override replaces", "--regenerate replaces"],
)
def test_batch_image_with_output_on_disk(
    fake_lr, monkeypatch, override, regenerate, rebuilt
):
    rendered, existing = fake_lr
    args = _run_args(
        output=str(existing.parent), override=override, regenerate=regenerate
    )
    monkeypatch.setattr(
        lr_cmd,
        "_WORKER_STATE",
        {"conn": _FakeConn(), "args": args, "fs_map": {}},
    )

    result = lr_cmd._process_one(7)

    assert rendered == ([(7, str(existing))] if rebuilt else [])
    # A kept file still counts as done, but with no tier: nothing was fetched.
    assert result == (7, str(existing), "network" if rebuilt else None)
