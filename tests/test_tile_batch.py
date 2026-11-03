"""Batch resilience for ``lavlab tile batch``: one bad slide must not end the run.

Two regressions: an ``omero.ResourceError`` (corrupt or missing pixel
buffer) was misread as a connection failure and set off a reconnect storm,
and any exception escaping a worker was re-raised by the pool and killed the
whole batch. The end-to-end tests here run a real (forked) worker pool
against a fake OMERO.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections import Counter

import pytest

pytest.importorskip("omero")
pytest.importorskip("numpy")

import Ice
import omero.clients

import lavlab.commands.tile as tile_cmd
import lavlab.omero_client as omero_client
from lavlab.cli import build_parser
from lavlab.omero_client import (
    describe_error,
    is_conn_error,
    is_resource_error,
    retry_delay,
)
from lavlab.tiling import SlideResult

# What OMERO actually sends: the message, plus a server stack trace that
# mentions sessions -- which is what fooled the old message-sniffing check.
SERVER_TRACE = (
    "ome.conditions.ResourceError: Error instantiating pixel buffer\n"
    "    at ome.io.nio.PixelsService.getPixelBuffer(PixelsService.java:609)\n"
    "    at ome.services.sessions.SessionManagerImpl.doWork(SessionManagerImpl.java)"
)


def _resource_error():
    return omero.ResourceError(
        serverStackTrace=SERVER_TRACE,
        serverExceptionClass="ome.conditions.ResourceError",
        message="Error instantiating pixel buffer: /OMERO/Pixels/2054_pyramid",
    )


class _FakeImage:
    def __init__(self, image_id):
        self._id = image_id

    def getId(self):
        return self._id


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


def _batch_args(workers=2):
    return build_parser().parse_args(
        [
            "tile",
            "batch",
            "-o",
            "/tmp/unused",
            "--roi",
            "--all",
            "--workers",
            str(workers),
        ]
    )


@pytest.fixture
def fake_batch(monkeypatch):
    """Wire _run_batch onto a fake OMERO; returns the shared login counter."""
    # Shared with the forked workers, so logins made in them are counted.
    logins = multiprocessing.get_context("fork").Value("i", 0)

    def fake_connect(args):
        with logins.get_lock():
            logins.value += 1
        return _FakeConn()

    monkeypatch.setattr(tile_cmd, "connect_from_args", fake_connect)
    monkeypatch.setattr(tile_cmd, "group_of", lambda conn, image: None)
    monkeypatch.setattr(tile_cmd, "_MAX_STARTUP_STAGGER", 0.0)
    monkeypatch.setattr(
        omero_client, "iter_image_ids", lambda conn, group: [1, 2054, 3, 4]
    )
    return logins


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def _tile_one_failing_on(image_id_to_fail, error):
    def fake_tile_one(conn, image, args, params):
        if image.getId() == image_id_to_fail:
            raise error
        return SlideResult(
            image.getId(), "done", tier="network", label_counts=Counter(G3=2)
        )

    return fake_tile_one


# ---------------------------------------------------------------------------
# the two regressions, end to end through a real worker pool
# ---------------------------------------------------------------------------


def test_a_resource_error_fails_one_slide_and_the_batch_completes(
    fake_batch, monkeypatch, caplog
):
    tile_one = _tile_one_failing_on(2054, _resource_error())

    def tile_once_both_workers_are_up(conn, image, args, params):
        # The fake slides finish in microseconds: without this, one worker
        # can tile all four before the other has run its initializer, and
        # the pool exits with that worker's login never made -- the count
        # below would then depend on fork timing, not on reconnect logic.
        deadline = time.monotonic() + 10
        while fake_batch.value < 1 + 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        return tile_one(conn, image, args, params)

    monkeypatch.setattr(tile_cmd, "_tile_one", tile_once_both_workers_are_up)

    with caplog.at_level("INFO"):
        tile_cmd._run_batch(_batch_args(workers=2))

    messages = _messages(caplog)
    assert (
        "Batch complete: 3/4 slides tiled, 0 skipped, 0 unannotated, 1 failed."
        in messages
    )
    assert "Failed image IDs: 2054" in messages
    assert "Tiles by label: G3=6" in messages
    # one login for the parent's image listing + one per worker -- a
    # ResourceError must not trigger a single reconnect
    assert fake_batch.value == 1 + 2


def test_an_arbitrary_worker_exception_fails_one_slide_and_the_batch_completes(
    fake_batch, monkeypatch, caplog
):
    monkeypatch.setattr(
        tile_cmd, "_tile_one", _tile_one_failing_on(3, ZeroDivisionError("boom"))
    )

    with caplog.at_level("INFO"):
        tile_cmd._run_batch(_batch_args(workers=2))

    messages = _messages(caplog)
    assert (
        "Batch complete: 3/4 slides tiled, 0 skipped, 0 unannotated, 1 failed."
        in messages
    )
    assert "Failed image IDs: 3" in messages


def test_a_worker_that_cannot_log_in_at_start_does_not_hang_the_pool(
    fake_batch, monkeypatch, caplog
):
    """A raising pool initializer makes Pool respawn workers forever."""
    failed_once = multiprocessing.get_context("fork").Value("i", 0)

    def flaky_connect(args):
        in_worker = multiprocessing.current_process().name != "MainProcess"
        with failed_once.get_lock():
            fail = in_worker and failed_once.value == 0
            if fail:
                failed_once.value = 1
        if fail:
            raise RuntimeError("Failed to connect to OMERO after 5 attempts")
        return _FakeConn()

    monkeypatch.setattr(tile_cmd, "connect_from_args", flaky_connect)
    monkeypatch.setattr(tile_cmd, "_tile_one", _tile_one_failing_on(-1, None))

    with caplog.at_level("INFO"):
        tile_cmd._run_batch(_batch_args(workers=2))

    assert failed_once.value == 1  # a worker's start-up login really did fail
    assert (
        "Batch complete: 4/4 slides tiled, 0 skipped, 0 unannotated, 0 failed."
        in _messages(caplog)
    )


# ---------------------------------------------------------------------------
# the worker, in process
# ---------------------------------------------------------------------------


_IMAGE_GROUPS = {1: 3, 2: 5}


class _GroupedImage(_FakeImage):
    def __init__(self, image_id, group_id):
        super().__init__(image_id)
        self.details = type(
            "Details",
            (),
            {"group": type("G", (), {"id": type("I", (), {"val": group_id})})},
        )


class _GroupScopedConn(_FakeConn):
    """Sees an image only from its own group or the all-groups scope (-1),
    as OMERO's query service does."""

    def getObject(self, kind, image_id):
        scope = getattr(self.SERVICE_OPTS, "group", -1)
        group_id = _IMAGE_GROUPS[image_id]
        if str(scope) not in ("-1", str(group_id)):
            return None
        return _GroupedImage(image_id, group_id)


def test_a_worker_finds_images_in_every_group(monkeypatch):
    """Tiling an image scopes the session to its group; the next lookup,
    for an image in another group, must still find it."""
    conn = _GroupScopedConn()
    monkeypatch.setattr(
        tile_cmd,
        "_WORKER_STATE",
        {"conn": conn, "args": _batch_args(), "params": None},
    )
    monkeypatch.setattr(tile_cmd, "_tile_one", _tile_one_failing_on(-1, None))

    results = [tile_cmd._process_one(image_id) for image_id in (1, 2, 1)]

    assert [r.status for r in results] == ["done", "done", "done"]
    assert conn.SERVICE_OPTS.group == "3"  # left in the last image's group


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(tile_cmd, "group_of", lambda conn, image: None)
    monkeypatch.setattr(
        tile_cmd,
        "_WORKER_STATE",
        {"conn": _FakeConn(), "args": _batch_args(), "params": None},
    )
    return tile_cmd._WORKER_STATE


def test_a_resource_error_is_not_retried(worker, monkeypatch, caplog):
    tile_calls = []

    def fake_tile_one(conn, image, args, params):
        tile_calls.append(image.getId())
        raise _resource_error()

    def no_reconnect(args):
        raise AssertionError("must not reconnect for a ResourceError")

    monkeypatch.setattr(tile_cmd, "_tile_one", fake_tile_one)
    monkeypatch.setattr(tile_cmd, "connect_from_args", no_reconnect)

    with caplog.at_level("WARNING"):
        result = tile_cmd._process_one(2054)

    assert result.status == "failed"
    assert result.reason == (
        "ResourceError: Error instantiating pixel buffer: /OMERO/Pixels/2054_pyramid"
    )
    assert tile_calls == [2054]
    (warning,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert "Image 2054" in warning.getMessage()
    assert "Error instantiating pixel buffer" in warning.getMessage()
    assert "serverStackTrace" not in warning.getMessage()


def test_a_missing_tifffile_extra_fails_one_slide_with_the_install_command(
    worker, monkeypatch, caplog
):
    """--tier local on a source only tifffile can read, without the extra."""
    from lavlab.tiling import TifffileUnavailableError

    message = "install it with: pip install 'lavlab-cli-utils[tifffile]'"

    def fake_tile_one(conn, image, args, params):
        raise TifffileUnavailableError(message)

    monkeypatch.setattr(tile_cmd, "_tile_one", fake_tile_one)

    with caplog.at_level("WARNING"):
        result = tile_cmd._process_one(885)

    assert result.status == "failed"
    assert result.reason == f"TifffileUnavailableError: {message}"
    (warning,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warning.getMessage() == f"Image 885: failed -- {result.reason}"
    assert warning.exc_info is None  # one line, not a traceback per slide


def test_a_genuine_connection_loss_reconnects_and_closes_the_old_session(
    worker, monkeypatch
):
    closed = []

    class DyingConn(_FakeConn):
        def close(self):
            closed.append(self)

    first = DyingConn()
    worker["conn"] = first
    attempts = []

    def fake_tile_one(conn, image, args, params):
        attempts.append(conn)
        if conn is first:
            raise Ice.ConnectionLostException()
        return SlideResult(image.getId(), "done", tier="network")

    monkeypatch.setattr(tile_cmd, "_tile_one", fake_tile_one)
    monkeypatch.setattr(tile_cmd, "connect_from_args", lambda args: _FakeConn())

    result = tile_cmd._process_one(7)

    assert result.status == "done"
    assert len(attempts) == 2 and attempts[1] is not first
    assert closed == [first]  # the abandoned session was released


def test_a_failed_reconnect_fails_the_slide_instead_of_escaping(worker, monkeypatch):
    def fake_tile_one(conn, image, args, params):
        raise Ice.ConnectionLostException()

    def refused(args):
        raise RuntimeError("Failed to connect to OMERO after 5 attempts")

    monkeypatch.setattr(tile_cmd, "_tile_one", fake_tile_one)
    monkeypatch.setattr(tile_cmd, "connect_from_args", refused)

    result = tile_cmd._process_one(9)

    assert result.status == "failed"
    assert "Failed to connect" in result.reason


def test_process_one_never_raises_even_for_system_exit(worker, monkeypatch):
    def fake_tile_one(conn, image, args, params):
        raise SystemExit("error: something called sys.exit")

    monkeypatch.setattr(tile_cmd, "_tile_one", fake_tile_one)

    result = tile_cmd._process_one(5)

    assert result.status == "failed"


def test_init_worker_never_raises(monkeypatch):
    def refused(args):
        raise RuntimeError("refused")

    monkeypatch.setattr(tile_cmd, "connect_from_args", refused)
    monkeypatch.setattr(tile_cmd, "_MAX_STARTUP_STAGGER", 0.0)

    tile_cmd._init_worker(_batch_args(), params=None)

    assert tile_cmd._WORKER_STATE["conn"] is None


# ---------------------------------------------------------------------------
# classification and backoff
# ---------------------------------------------------------------------------


def test_a_resource_error_is_not_a_connection_error_despite_its_stack_trace():
    error = _resource_error()

    assert "session" in str(error).lower()  # what used to fool the check
    assert is_resource_error(error)
    assert not is_conn_error(error)


@pytest.mark.parametrize(
    "error",
    [
        Ice.ConnectionLostException(),
        Ice.ConnectionRefusedException(),
        Ice.TimeoutException(),
        Ice.ProtocolException(),
        Ice.CommunicatorDestroyedException(),
        Ice.ObjectNotExistException(),
        ConnectionResetError(),
        TimeoutError(),
    ],
    ids=type,
)
def test_transport_failures_are_connection_errors(error):
    assert is_conn_error(error)


def test_an_expired_session_is_a_connection_error():
    assert is_conn_error(omero.RemovedSessionException(message="session gone"))


@pytest.mark.parametrize(
    "error",
    [
        omero.ApiUsageException(message="bad call"),
        omero.SecurityViolation(message="not yours"),
        Ice.UnknownLocalException(unknown="server-side bug"),
        ValueError("connection timeout session"),  # message text means nothing
        RuntimeError("Failed to connect"),
    ],
    ids=type,
)
def test_request_failures_are_not_connection_errors(error):
    assert not is_conn_error(error)


def test_describe_error_drops_the_server_stack_trace():
    assert describe_error(_resource_error()) == (
        "ResourceError: Error instantiating pixel buffer: /OMERO/Pixels/2054_pyramid"
    )
    assert describe_error(ValueError("first line\nsecond")) == "ValueError: first line"


def test_rate_limited_logins_back_off_longer_and_jittered():
    rate_limited = [retry_delay(1, Ice.ProtocolException(), 1.0) for _ in range(200)]
    ordinary = [retry_delay(1, None, 1.0) for _ in range(200)]

    assert all(d >= 2.0 for d in rate_limited)
    assert min(ordinary) < 2.0  # 1 s + up to 1 s jitter
    assert max(rate_limited) - min(rate_limited) > 4.0  # spread, not lockstep
    # capped, however many attempts
    assert all(retry_delay(20, Ice.ProtocolException(), 1.0) <= 60.0 for _ in range(50))


def test_connect_waits_longer_when_the_session_service_refuses(monkeypatch):
    from lavlab.config import OmeroCreds

    class Refusing:
        def __init__(self, *args, **kwargs):
            raise Ice.ProtocolException()

    delays = []
    monkeypatch.setattr(omero_client, "BlitzGateway", Refusing)
    monkeypatch.setattr(omero_client.time, "sleep", delays.append)

    with pytest.raises(RuntimeError, match="after 4 attempts"):
        omero_client.connect(
            OmeroCreds(user="u", password="p", host="h", port=4064), retries=4
        )

    assert len(delays) == 3
    assert all(d >= 2.0 for d in delays)


# ---------------------------------------------------------------------------
# start-up: the label map is read once, before any slide
# ---------------------------------------------------------------------------


def _batch_args_in(out, *extra, workers=2):
    return build_parser().parse_args(
        [
            "tile",
            "batch",
            "-o",
            str(out),
            "--roi",
            "--all",
            "--workers",
            str(workers),
            *extra,
        ]
    )


def _no_connection(args):
    raise AssertionError("connected to OMERO before checking the label map")


@pytest.mark.parametrize(
    "content, match",
    [(None, "FileNotFoundError|does not exist"), ("[a, b", "YAMLError|Scanner")],
)
def test_an_unreadable_label_map_stops_the_run_before_connecting(
    tmp_path, monkeypatch, content, match
):
    labels = tmp_path / "labels.yaml"
    if content is not None:
        labels.write_text(content)
    monkeypatch.setattr(tile_cmd, "connect_from_args", _no_connection)
    args = _batch_args_in(tmp_path / "out", "--labels", str(labels))

    with pytest.raises(SystemExit, match="cannot read the label map") as excinfo:
        tile_cmd.run(args)

    assert str(labels) in str(excinfo.value)
    assert "Nothing was tiled" in str(excinfo.value)
    assert not (tmp_path / "out").exists()


def test_a_label_map_that_vanishes_mid_run_fails_no_slide(
    fake_batch, monkeypatch, tmp_path, caplog
):
    """A dropped mount after start-up: every slide still has its labels."""
    labels = tmp_path / "labels.yaml"
    labels.write_text("Cancer: [g3, g4]\n")

    def tile_one(conn, image, args, params):
        # The mount drops as soon as the first slide starts.
        if labels.exists():
            labels.unlink()
        if params.label_map != {"Cancer": ["g3", "g4"]}:
            raise RuntimeError("label map was not carried into the worker")
        return SlideResult(image.getId(), "done", label_counts=Counter(Cancer=1))

    monkeypatch.setattr(tile_cmd, "_tile_one", tile_one)

    with caplog.at_level("INFO"):
        tile_cmd.run(_batch_args_in(tmp_path / "out", "--labels", str(labels)))

    assert not labels.exists()
    assert (
        "Batch complete: 4/4 slides tiled, 0 skipped, 0 unannotated, 0 failed."
        in _messages(caplog)
    )


# ---------------------------------------------------------------------------
# every failure: one log line, a row in failures.csv, a reason in the summary
# ---------------------------------------------------------------------------


def _failure_files(out):
    """This out dir's failures files, oldest run first.

    By (stamp, attempt), not by name: ``-2`` sorts before ``.csv``.
    """

    def run_order(path):
        _prefix, stamp, *attempt = path.stem.split("-")
        return stamp, int(attempt[0]) if attempt else 1

    return sorted(out.glob("failures-*.csv"), key=run_order)


def _read_failures(path):
    import csv

    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def test_every_failure_is_written_to_failures_csv_and_summarised(
    fake_batch, monkeypatch, tmp_path, caplog
):
    from lavlab.tiling import TilingError

    failures = {
        1: TilingError("label map '/mnt/labels.yaml' does not exist."),
        3: TilingError("label map '/mnt/labels.yaml' does not exist."),
        2054: _resource_error(),
    }

    def tile_one(conn, image, args, params):
        if image.getId() in failures:
            raise failures[image.getId()]
        return SlideResult(image.getId(), "done", label_counts=Counter(G3=2))

    monkeypatch.setattr(tile_cmd, "_tile_one", tile_one)
    out = tmp_path / "out"

    with caplog.at_level("INFO"):
        tile_cmd._run_batch(_batch_args_in(out))

    # (the per-slide "Image <id>: failed -- ..." lines are logged in the
    # forked workers, out of caplog's reach; see the _process_one tests)
    label_reason = "TilingError: label map '/mnt/labels.yaml' does not exist."
    resource_reason = (
        "ResourceError: Error instantiating pixel buffer: /OMERO/Pixels/2054_pyramid"
    )
    (failures,) = _failure_files(out)
    assert _read_failures(failures) == [
        ["image_id", "reason"],
        ["1", label_reason],
        ["2054", resource_reason],
        ["3", label_reason],
    ]
    messages = _messages(caplog)
    assert "Failed by type: TilingError=2, ResourceError=1" in messages
    assert f"      2 x {label_reason}" in messages
    assert f"      1 x {resource_reason}" in messages


def test_each_run_keeps_its_own_failures_file(fake_batch, monkeypatch, tmp_path):
    """A --skip-existing resume must not erase the earlier run's failures."""
    out = tmp_path / "out"
    monkeypatch.setattr(
        tile_cmd, "_tile_one", _tile_one_failing_on(3, ZeroDivisionError("boom"))
    )
    tile_cmd._run_batch(_batch_args_in(out))
    monkeypatch.setattr(tile_cmd, "_tile_one", _tile_one_failing_on(-1, None))
    tile_cmd._run_batch(_batch_args_in(out, "--skip-existing"))

    first, second = _failure_files(out)
    assert _read_failures(first) == [
        ["image_id", "reason"],
        ["3", "ZeroDivisionError: boom"],
    ]
    assert _read_failures(second) == [["image_id", "reason"]]  # a clean run


def test_runs_in_the_same_second_never_share_a_failures_file(tmp_path):
    results = [SlideResult(1, "failed", reason="ValueError: x")]

    paths = [
        tile_cmd._write_failures(str(tmp_path), results, "20261103T153929Z")
        for _ in range(3)
    ]

    assert [os.path.basename(p) for p in paths] == [
        "failures-20261103T153929Z.csv",
        "failures-20261103T153929Z-2.csv",
        "failures-20261103T153929Z-3.csv",
    ]


def test_a_dry_run_writes_no_failures_csv(fake_batch, monkeypatch, tmp_path):
    monkeypatch.setattr(
        tile_cmd, "_tile_one", _tile_one_failing_on(3, ZeroDivisionError("boom"))
    )
    out = tmp_path / "out"

    tile_cmd._run_batch(_batch_args_in(out, "--dry-run"))

    assert not out.exists()


def test_the_summary_lists_the_top_reasons_and_counts_the_rest(caplog):
    results = [
        SlideResult(i, "failed", reason=f"ResourceError: no pixels for {i}")
        for i in range(8)
    ] + [SlideResult(100, "done")]

    with caplog.at_level("INFO"):
        tile_cmd._summarise_batch(list(range(9)), results)

    messages = _messages(caplog)
    assert "Failed by type: ResourceError=8" in messages
    assert sum(" x ResourceError: no pixels for " in m for m in messages) == 5
    assert "  ... and 3 other reasons." in messages
    assert any("missing or corrupt on the server" in m for m in messages)


def test_a_batch_where_everything_failed_names_the_cause(caplog):
    reason = "TilingError: label map '/mnt/labels.yaml' does not exist."
    results = [SlideResult(i, "failed", reason=reason) for i in range(3)]

    with pytest.raises(SystemExit, match="all 3 images failed") as excinfo:
        tile_cmd._summarise_batch([0, 1, 2], results)

    assert f"(3x) was: {reason}" in str(excinfo.value)


def test_a_tiling_error_is_one_log_line_with_its_type(worker, monkeypatch, caplog):
    from lavlab.tiling import TilingError

    error = TilingError("label map '/mnt/labels.yaml' does not exist.")
    monkeypatch.setattr(tile_cmd, "_tile_one", _tile_one_failing_on(885, error))

    with caplog.at_level("WARNING"):
        result = tile_cmd._process_one(885)

    (warning,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warning.getMessage() == (
        "Image 885: failed -- TilingError: label map '/mnt/labels.yaml' does not exist."
    )
    assert warning.exc_info is None
    assert result.reason == "TilingError: label map '/mnt/labels.yaml' does not exist."


def test_an_unexpected_failure_keeps_its_traceback(worker, monkeypatch, caplog):
    monkeypatch.setattr(
        tile_cmd, "_tile_one", _tile_one_failing_on(5, ZeroDivisionError("boom"))
    )

    with caplog.at_level("WARNING"):
        result = tile_cmd._process_one(5)

    assert result.reason == "ZeroDivisionError: boom"
    (warning,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warning.getMessage() == "Image 5: failed -- ZeroDivisionError: boom"
    assert warning.exc_info is not None


# ---------------------------------------------------------------------------
# retries: only for the connection, with jittered exponential backoff
# ---------------------------------------------------------------------------


def test_reconnects_back_off_exponentially_with_jitter(worker, monkeypatch):
    delays = []
    calls = []

    def fake_tile_one(conn, image, args, params):
        calls.append(conn)
        if len(calls) < 3:
            raise Ice.ConnectionLostException()
        return SlideResult(image.getId(), "done")

    monkeypatch.setattr(tile_cmd, "_tile_one", fake_tile_one)
    monkeypatch.setattr(tile_cmd, "connect_from_args", lambda args: _FakeConn())
    monkeypatch.setattr(tile_cmd.time, "sleep", delays.append)

    result = tile_cmd._process_one(7)

    assert result.status == "done"
    assert len(delays) == 2
    # retry_delay: base * 2**(attempt-1) plus up to a second of jitter
    assert 1.0 <= delays[0] < 2.0 <= delays[1] < 3.0


def test_a_resource_error_never_waits_or_reconnects(worker, monkeypatch):
    delays = []
    monkeypatch.setattr(
        tile_cmd, "_tile_one", _tile_one_failing_on(2054, _resource_error())
    )
    monkeypatch.setattr(tile_cmd.time, "sleep", delays.append)
    monkeypatch.setattr(tile_cmd, "connect_from_args", _no_connection)

    result = tile_cmd._process_one(2054)

    assert result.status == "failed"
    assert delays == []


# ---------------------------------------------------------------------------
# --manifest on a filesystem that cannot lock (SMB)
# ---------------------------------------------------------------------------


def _manifest_rows(image_id, count=2):
    from lavlab.tiling import MANIFEST_COLUMNS

    return [
        {
            **dict.fromkeys(MANIFEST_COLUMNS, "0"),
            "tile_path": f"N1/s{image_id}/G3/t{i}.png",
            "omero_image_id": str(image_id),
            "label": "G3",
            "tier": "network",
        }
        for i in range(count)
    ]


def _tile_one_writing_manifest_rows(conn, image, args, params):
    """Append rows the way tile_slide does, through _tile_one's contract."""
    from lavlab import tiling

    tiling.append_manifest(
        args.manifest,
        _manifest_rows(image.getId()),
        args.out,
        lock=not getattr(args, "manifest_is_shard", False),
    )
    return SlideResult(image.getId(), "done", label_counts=Counter(G3=2))


def test_batch_manifest_works_where_flock_always_fails(
    fake_batch, monkeypatch, tmp_path, caplog
):
    """SMB: flock raises BlockingIOError [Errno 35]. Workers write shards."""
    import csv
    import errno
    import fcntl

    from lavlab import tiling

    def flock(fh, operation):
        raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(fcntl, "flock", flock)
    monkeypatch.setattr(tiling, "_LOCK_ATTEMPTS", 1)
    monkeypatch.setattr(tile_cmd, "_tile_one", _tile_one_writing_manifest_rows)
    out = tmp_path / "out"
    manifest = tmp_path / "dataset.csv"

    with caplog.at_level("INFO"):
        tile_cmd._run_batch(_batch_args_in(out, "--manifest", str(manifest)))

    with open(manifest, newline="") as fh:
        merged = list(csv.DictReader(fh))
    assert sorted(r["omero_image_id"] for r in merged) == sorted(
        ["1", "1", "2054", "2054", "3", "3", "4", "4"]
    )
    # paths are relative to the manifest, as append_manifest always wrote them
    assert {r["tile_path"].split("/")[0] for r in merged} == {"out"}
    assert list(tmp_path.glob(".dataset.csv.*.part")) == []  # shards merged away
    messages = _messages(caplog)
    assert (
        "Batch complete: 4/4 slides tiled, 0 skipped, 0 unannotated, 0 failed."
        in messages
    )
    # only the parent's single merge needed the lock, and it carried on without
    assert sum("Could not lock --manifest" in m for m in messages) == 1
    assert (
        "Manifest: 8 row(s) appended from 2 worker shard(s); 0 not appended "
        "(0 identical to existing rows, 0 conflicting, 0 malformed)"
    ) in messages


def test_a_batch_rerun_into_the_same_manifest_adds_no_duplicates(
    fake_batch, monkeypatch, tmp_path, caplog
):
    import csv

    monkeypatch.setattr(tile_cmd, "_tile_one", _tile_one_writing_manifest_rows)
    manifest = tmp_path / "dataset.csv"
    args = _batch_args_in(tmp_path / "out", "--manifest", str(manifest))

    tile_cmd._run_batch(args)
    with caplog.at_level("INFO"):
        tile_cmd._run_batch(args)

    with open(manifest, newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 8
    # the skipped duplicates are counted, at WARNING, in the log and summary
    summary = [r for r in caplog.records if r.getMessage().startswith("Manifest: ")]
    assert [r.levelname for r in summary] == ["WARNING"]
    assert summary[0].getMessage() == (
        "Manifest: 0 row(s) appended from 2 worker shard(s); 8 not appended "
        "(8 identical to existing rows, 0 conflicting, 0 malformed)"
    )
    assert any(
        "8 of 8 shard row(s) not appended -- 8 identical" in r.getMessage()
        and r.levelname == "WARNING"
        for r in caplog.records
    )


def test_a_manifest_with_other_columns_stops_the_batch_before_any_slide(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "dataset.csv"
    manifest.write_text("tile_path,label\nx.png,G3\n")
    monkeypatch.setattr(tile_cmd, "connect_from_args", _no_connection)

    with pytest.raises(SystemExit, match="different columns"):
        tile_cmd._run_batch(
            _batch_args_in(tmp_path / "out", "--manifest", str(manifest))
        )


def test_tile_one_appends_to_a_shard_without_locking(monkeypatch):
    from lavlab import tiling

    seen = {}
    monkeypatch.setattr(tiling, "tile_slide", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(tile_cmd, "connect_from_args", lambda args: _FakeConn())
    monkeypatch.setattr(tile_cmd, "_MAX_STARTUP_STAGGER", 0.0)
    monkeypatch.setattr(tile_cmd, "_WORKER_STATE", {})
    args = _batch_args_in("/tmp/out", "--manifest", "/tmp/dataset.csv")
    tile_cmd._tile_one(None, None, args, None)
    assert seen["manifest_lock"] is True

    tile_cmd._init_worker(args, None, manifest_run="run1")
    shard_args = tile_cmd._WORKER_STATE["args"]
    tile_cmd._tile_one(None, None, shard_args, None)

    assert seen["manifest_lock"] is False
    assert seen["manifest_path"] == tiling.manifest_shard_path(
        "/tmp/dataset.csv", "run1", os.getpid()
    )
    assert args.manifest == "/tmp/dataset.csv"  # the parent's args untouched
