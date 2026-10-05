"""Batch resilience for ``lavlab tile batch``: one bad slide must not end the run.

Two regressions: an ``omero.ResourceError`` (corrupt or missing pixel
buffer) was misread as a connection failure and set off a reconnect storm,
and any exception escaping a worker was re-raised by the pool and killed the
whole batch. The end-to-end tests here run a real (forked) worker pool
against a fake OMERO.
"""

from __future__ import annotations

import multiprocessing
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


class _FakeConn:
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
    monkeypatch.setattr(
        tile_cmd, "_tile_one", _tile_one_failing_on(2054, _resource_error())
    )

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
