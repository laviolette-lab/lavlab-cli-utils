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

import pytest

from lavlab.omero_asyncio import AsyncService, AsyncSession


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
