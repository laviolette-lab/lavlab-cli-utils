# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for lavlab.omero_asyncio -- pure asyncio logic, no Ice/OMERO required.

Wraps a fake object exposing Ice's begin_f/end_f/f convention and confirms
AsyncService/AsyncSession convert it correctly. This also stands as a
regression check that the module imports with zero Ice/omero dependency.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from lavlab.omero_asyncio import AsyncService, AsyncSession, run_sync


class _FakeAsyncResult:
    def isSent(self):
        return True

    def isCompleted(self):
        return True


class _FakeService:
    """Mimics an Ice proxy: async methods come in begin_f/end_f pairs
    alongside a synchronous f, plus some methods with no async counterpart."""

    def begin_add(self, a, b, _response=None, _ex=None):
        _response(a + b)
        return _FakeAsyncResult()

    def end_add(self, async_result):
        raise AssertionError("end_add should never be called by AsyncService")

    def add(self, a, b):
        return a + b

    def begin_fail(self, _response=None, _ex=None):
        _ex(ValueError("boom"))
        return _FakeAsyncResult()

    def end_fail(self, async_result):
        raise AssertionError("end_fail should never be called by AsyncService")

    def fail(self):
        raise ValueError("boom")

    def plain(self):
        return "sync-only"


class _FakeSession(_FakeService):
    """Mimics a ServiceFactory: has a getXxxService()/createXxxStore() whose
    *result* should also come back wrapped in AsyncService."""

    def begin_getWidgetService(self, _response=None, _ex=None):
        _response(_FakeService())
        return _FakeAsyncResult()

    def end_getWidgetService(self, async_result):
        raise AssertionError(
            "end_getWidgetService should never be called by AsyncService"
        )

    def getWidgetService(self):
        return _FakeService()


def test_async_service_wraps_begin_end_pair_into_single_awaitable():
    async def run():
        svc = AsyncService(_FakeService())
        result = await svc.add(2, 3)
        assert result == 5

    asyncio.run(run())


def test_async_service_propagates_exceptions():
    async def run():
        svc = AsyncService(_FakeService())
        with pytest.raises(ValueError, match="boom"):
            await svc.fail()

    asyncio.run(run())


def test_async_service_leaves_plain_sync_methods_untouched():
    async def run():
        svc = AsyncService(_FakeService())
        assert svc.plain() == "sync-only"

    asyncio.run(run())


def test_async_session_wraps_get_service_results_too():
    async def run():
        session = AsyncSession(_FakeSession())
        widget_svc = await session.getWidgetService()
        assert isinstance(widget_svc, AsyncService)
        result = await widget_svc.add(10, 20)
        assert result == 30

    asyncio.run(run())


# ---------------------------------------------------------------------------
# run_sync: synchronous callers, with or without an event loop already running
# ---------------------------------------------------------------------------


async def _thread_and_sum(a, b):
    await asyncio.sleep(0)
    return threading.current_thread().name, a + b


def test_run_sync_without_a_running_loop_is_asyncio_run():
    thread, total = run_sync(_thread_and_sum(2, 3))

    assert total == 5
    assert thread == threading.current_thread().name


def test_run_sync_works_inside_a_running_loop():
    """A Jupyter kernel's main thread always runs an event loop, where
    asyncio.run() refuses to start; lavlab's network reads must still work."""

    async def kernel_cell():
        # Synchronous lavlab code, called from inside a running loop.
        return run_sync(_thread_and_sum(2, 3))

    thread, total = asyncio.run(kernel_cell())

    assert total == 5
    assert thread != threading.current_thread().name  # its own loop and thread


def test_run_sync_inside_a_running_loop_drives_ice_style_services():
    async def fetch():
        return await AsyncService(_FakeService()).add(2, 3)

    async def kernel_cell():
        return run_sync(fetch())

    assert asyncio.run(kernel_cell()) == 5


def test_run_sync_propagates_exceptions_inside_a_running_loop():
    async def fail():
        raise ValueError("boom")

    async def kernel_cell():
        return run_sync(fail())

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(kernel_cell())
