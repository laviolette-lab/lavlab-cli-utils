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

from lavlab.commands.lr import _summarise_batch, _validate_args


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
