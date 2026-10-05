# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""OMERO connection helpers.

Per the design: object lookups use the "dummy" group (-1) so objects are
visible across all groups the user belongs to; once a specific object is
being operated on, the connection is switched into that object's own group.
"""

from __future__ import annotations

import logging
import random
import time

import omero.sys
from omero.gateway import BlitzGateway

from lavlab.config import OmeroCreds

log = logging.getLogger(__name__)

ALL_GROUPS = -1


RATE_LIMIT_MAX_DELAY = 60.0


def is_rate_limited(exc: BaseException | None) -> bool:
    """Return True if *exc* looks like the session service pushing back.

    Many workers logging in at once can get the session service to refuse
    with an Ice ``ProtocolException``. That is not a dead server, it is a
    busy one -- the answer is to wait longer, not to retry quickly.
    """
    import Ice

    return isinstance(exc, Ice.ProtocolException)


def retry_delay(attempt: int, exc: BaseException | None, base_delay: float) -> float:
    """Seconds to wait before connection attempt ``attempt + 1``.

    Ordinary failures back off exponentially from *base_delay*. A
    rate-limited refusal (:func:`is_rate_limited`) instead draws from a much
    wider, capped window ("full jitter"), so several workers refused at the
    same moment spread out instead of retrying in lockstep.
    """
    if is_rate_limited(exc):
        ceiling = min(RATE_LIMIT_MAX_DELAY, 5.0 * base_delay * 2**attempt)
        return random.uniform(2.0 * base_delay, max(ceiling, 2.0 * base_delay))
    return base_delay * (2 ** (attempt - 1)) + random.random()


def connect(
    creds: OmeroCreds, retries: int = 5, base_delay: float = 1.0
) -> BlitzGateway:
    """Connect to OMERO, retrying transient failures with exponential backoff.

    Rate-limited refusals back off longer and with more jitter; see
    :func:`retry_delay`.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            conn = BlitzGateway(
                creds.user,
                creds.password,
                host=creds.host,
                port=creds.port,
                secure=True,
            )
            if conn.connect():
                conn.SERVICE_OPTS.setOmeroGroup(ALL_GROUPS)
                return conn
            log.info("Failed to create OMERO session. Attempt %d/%d", attempt, retries)
            try:
                conn.close()
            except Exception:
                pass
        except Exception as exc:
            last_exc = exc
            log.info(
                "Failed to create OMERO session. Attempt %d/%d: %s",
                attempt,
                retries,
                describe_error(exc),
            )

        if attempt < retries:
            delay = retry_delay(attempt, last_exc, base_delay)
            if is_rate_limited(last_exc):
                log.info(
                    "OMERO session service refused the login (busy); waiting %.0f s.",
                    delay,
                )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to connect to OMERO after {retries} attempts"
    ) from last_exc


def switch_to_object_group(conn: BlitzGateway, obj) -> None:
    """Switch the connection's security context to the given object's own group."""
    group_id = obj.details.group.id.val
    conn.SERVICE_OPTS.setOmeroGroup(str(group_id))


def is_conn_error(exc: BaseException) -> bool:
    """Return True if *exc* is a connection failure that reconnecting can fix.

    Decided by exception *type*, never by message text: an OMERO server error
    carries the server's whole stack trace in ``str(exc)``, which routinely
    mentions sessions and timeouts whatever actually went wrong -- matching
    on that turned a corrupt pixel buffer into a reconnect storm.

    * ``omero.SessionException`` (session expired or removed): yes.
    * any other ``omero.ServerError`` -- ``ResourceError``,
      ``ApiUsageException``, ``SecurityViolation``, ...: no. The server
      answered; it is the request (usually: this image) that failed.
    * Ice transport failures (``Ice.LocalException``: connection lost or
      refused, timeouts, protocol errors, destroyed communicator, stale
      proxies): yes -- except ``Ice.UnknownException``, which is a
      server-side failure relayed over a healthy connection.
    * the builtin ``ConnectionError`` / ``TimeoutError``: yes.
    """
    import Ice

    if isinstance(exc, omero.SessionException):
        return True
    if isinstance(exc, omero.ServerError):
        return False
    if isinstance(exc, Ice.UnknownException):
        return False
    if isinstance(exc, Ice.LocalException):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError))


def is_resource_error(exc: BaseException) -> bool:
    """Return True if OMERO could not open the resource (e.g. pixels) asked for.

    ``omero.ResourceError`` on ``setPixelsId`` ("Error instantiating pixel
    buffer") means the image's file is missing or corrupt server-side: a
    property of that one image, which no reconnect will change.
    """
    return isinstance(exc, omero.ResourceError)


def describe_error(exc: BaseException) -> str:
    """A one-line description of *exc*, without OMERO's server stack trace.

    :param exc: any exception
    :return: ``"<Type>: <message>"``
    """
    message = (
        getattr(exc, "message", None) if isinstance(exc, omero.ServerError) else None
    )
    if not message:
        text = str(exc).strip()
        message = text.splitlines()[0] if text else ""
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def get_source_file_path(conn: BlitzGateway, image_id: int) -> str | None:
    """Return the absolute server-side path of the primary file for an image.

    Uses the fileset -> usedFiles -> originalFile relationship so the path is
    always what OMERO recorded on import.
    """
    qs = conn.getQueryService()
    params = omero.sys.ParametersI()
    params.addId(image_id)
    files = qs.findAllByQuery(
        "select f from Image i "
        "join i.fileset fs "
        "join fs.usedFiles fe "
        "join fe.originalFile f "
        "where i.id = :id",
        params,
        conn.SERVICE_OPTS,
    )
    if not files:
        return None

    # Prefer the largest file -- that's the primary image file in a multi-file set.
    files.sort(key=lambda f: f.size.val if f.size is not None else 0, reverse=True)
    f = files[0]
    return "/OMERO/ManagedRepository/" + f.path.val + f.name.val


def iter_image_ids(conn: BlitzGateway, group_id: int | None = None):
    """Yield image IDs to batch-process.

    If group_id is given, only that group's images are listed (and the
    connection stays scoped to that group). Otherwise all images visible
    across every group the user belongs to are listed (dummy group -1);
    callers must switch_to_object_group() before operating on each one.
    """
    if group_id is not None:
        conn.SERVICE_OPTS.setOmeroGroup(str(group_id))
    else:
        conn.SERVICE_OPTS.setOmeroGroup(ALL_GROUPS)

    for image in conn.getObjects("Image"):
        yield image.getId()
