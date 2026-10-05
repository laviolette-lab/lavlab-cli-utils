# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Asyncio adapter for OMERO's synchronous Ice service proxies."""

from __future__ import annotations

import asyncio
import logging
from functools import partial, update_wrapper

log = logging.getLogger(__name__)


def _firstline_truncate(s) -> str:
    lines = f"{s}\n".splitlines()
    if len(lines[0]) > 80 or len(lines) > 1:
        s = lines[0][:79] + "…"
    return s


async def ice_async(loop: asyncio.AbstractEventLoop, func, *args, **kwargs):
    """Wrap an asynchronous Ice service method so it can be used with asyncio.

    :param loop: The event loop.
    :param func: The Ice service method (a ``begin_f``).
    :param args: Positional arguments for the Ice service method.
    :param kwargs: Keyword arguments for the Ice service method.
    """

    future = loop.create_future()

    def exception_cb(ex):
        if log.isEnabledFor(logging.DEBUG):
            log.debug("exception_cb: %s", _firstline_truncate(ex))
        loop.call_soon_threadsafe(future.set_exception, ex)

    def response_cb(result=None, *outparams):
        if log.isEnabledFor(logging.DEBUG):
            log.debug("response_cb: %s", _firstline_truncate(result))
        loop.call_soon_threadsafe(future.set_result, result)

    a = func(*args, **kwargs, _response=response_cb, _ex=exception_cb)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "_exec_ice_async(%s) sent:%s completed:%s",
            func.__name__,
            a.isSent(),
            a.isCompleted(),
        )

    result = await future
    return result


class AsyncService:
    """Convert an OMERO Ice service to an async service."""

    def __init__(
        self, svc: object, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        """
        :param svc: The OMERO Ice service.
        :param loop: The async event loop (optional; defaults to the
            currently running loop).
        """
        if not loop:
            loop = asyncio.get_running_loop()
        methods = {
            m for m in dir(svc) if callable(getattr(svc, m)) and not m.startswith("_")
        }

        async_methods = {m for m in methods if m.startswith("begin_")}
        for async_m in async_methods:
            sync_m = async_m[6:]
            methods.discard(sync_m)
            methods.discard("begin_" + sync_m)
            methods.discard("end_" + sync_m)
            setattr(
                self,
                sync_m,
                update_wrapper(
                    partial(ice_async, loop, getattr(svc, async_m)),
                    getattr(svc, sync_m),
                ),
            )
        for sync_m in methods:
            setattr(
                self,
                sync_m,
                update_wrapper(partial(getattr(svc, sync_m)), getattr(svc, sync_m)),
            )


async def _getServiceWrapper(getsvc_m, loop):  # noqa: N802
    svc = await getsvc_m()
    return AsyncService(svc, loop)


class AsyncSession(AsyncService):
    """Wrap a session (e.g. a BlitzGateway connection's ``c.sf``) so all
    services it hands out are async."""

    def __init__(self, session, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """
        :param session: The OMERO session (e.g. ``conn.c.sf``).
        :param loop: The async event loop (optional).
        """
        super().__init__(session, loop)
        getsvc_methods = {
            m
            for m in dir(self)
            if (
                callable(getattr(self, m))
                and (m.startswith("get") and m.endswith("Service"))
            )
            or (m.startswith("create") and m.endswith("Store"))
        }

        for getsvc_m in getsvc_methods:
            setattr(
                self,
                getsvc_m,
                update_wrapper(
                    partial(_getServiceWrapper, getattr(self, getsvc_m), loop),
                    getattr(session, getsvc_m),
                ),
            )
