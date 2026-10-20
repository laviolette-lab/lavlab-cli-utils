# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""A MONAI whole-slide reader whose "files" are OMERO images.

:class:`OmeroWSIReader` is a ``monai.data.BaseWSIReader``: hand it to
``monai.data.PatchWSIDataset`` (or anything else that takes a WSI reader)
and patches are read from OMERO on demand. Pixels come through the same two
tiers ``lavlab tile`` uses -- :class:`lavlab.tiling.LocalRegionReader` on a
mounted managed repository, :class:`lavlab.tiling.NetworkRegionReader`
otherwise -- with the same fall-forward and JPEG-2000 rules, and are resized
with :func:`lavlab.tiling.resize_tile`, so a patch read here is the tile
``lavlab tile`` would have written.

Connections are per process. A ``DataLoader`` forks or spawns its workers
from a parent that may already hold a session; an Ice session does not
survive being copied into another process, so each process logs in on its
first read, never touches one it inherited, and closes its own on exit.
"""

from __future__ import annotations

import functools
import logging
import multiprocessing.util
import os
import threading

import numpy as np

from lavlab.monai._optional import EXTRA_HINT, is_missing_extra
from lavlab.monai.datalist import claim_vips, image_uri, parse_image_id
from lavlab.tiling import (
    MISSING_PIXEL_SIZE_MESSAGE,
    LevelPlan,
    LocalRegionReader,
    MissingPixelSizeError,
    NetworkRegionReader,
    TilingError,
    _local_source_errors,
    choose_tier,
    image_pixel_sizes,
    local_level_dims,
    network_level_dims,
    resize_tile,
    warn_local_fallback,
)

try:
    from monai.data import BaseWSIReader
except ImportError as exc:  # pragma: no cover - depends on the install
    if is_missing_extra(exc):
        raise ImportError(EXTRA_HINT) from exc
    raise

log = logging.getLogger(__name__)

# Guards every per-process lazy initialisation below. A lock held by another
# thread at fork time stays held, forever, in the child, so the child gets a
# fresh one.
_PROCESS_LOCK = threading.RLock()


def _reset_process_lock() -> None:
    global _PROCESS_LOCK
    _PROCESS_LOCK = threading.RLock()


os.register_at_fork(after_in_child=_reset_process_lock)


def _connect(creds):
    from lavlab.config import resolve_creds
    from lavlab.omero_client import connect

    return connect(creds if creds is not None else resolve_creds())


def _close_owned(conn, owner_pid: int) -> None:
    """Close *conn*, but only in the process that opened it.

    Closing a gateway ends its server session, so a copy inherited by a
    forked child must never be closed there -- it would log the parent out.
    """
    if os.getpid() != owner_pid:
        return
    try:
        conn.close()
    except Exception:
        log.debug("closing an OMERO connection failed", exc_info=True)


def _get_image(conn, image_id: int):
    """Look an image up across every group the user is in.

    The session may still be scoped to the previous image's group, and a
    lookup scoped to one group cannot see an image in another.
    """
    from lavlab.omero_client import ALL_GROUPS

    conn.SERVICE_OPTS.setOmeroGroup(ALL_GROUPS)
    image = conn.getObject("Image", image_id)
    if image is None:
        raise ValueError(
            f"OMERO image {image_id} not found (or not visible to this user)."
        )
    return image


def _objective_power(image) -> float | None:
    try:
        objective = image.getObjectiveSettings().getObjective()
        value = objective.getNominalMagnification()
    except Exception:
        return None
    if value is None:
        return None
    if hasattr(value, "getValue"):
        value = value.getValue()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


class OmeroWSI:
    """An open OMERO image, as returned by :meth:`OmeroWSIReader.read`.

    Opaque to MONAI. Holds what the reader needs about the image -- its
    size, pixel size and pyramid, and the tier its pixels come from -- as
    plain values, so it pickles into ``DataLoader`` workers. The live part
    (gateway image object, open region readers) is rebuilt lazily in each
    process and never pickled.
    """

    def __init__(
        self,
        image_id: int,
        *,
        width0: int,
        height0: int,
        pixel_size_x: float | None,
        pixel_size_y: float | None,
        level_dims: list[tuple[int, int]],
        tier: str,
        src_path: str | None,
        objective_power: float | None = None,
    ):
        """
        :param image_id: the OMERO image ID
        :type image_id: int
        :param width0: full-resolution width
        :type width0: int
        :param height0: full-resolution height
        :type height0: int
        :param pixel_size_x: physical pixel size in um, or None
        :type pixel_size_x: float | None
        :param pixel_size_y: ditto, Y
        :type pixel_size_y: float | None
        :param level_dims: ``(width, height)`` per level, 0 = full resolution
        :type level_dims: list[tuple[int, int]]
        :param tier: ``"local"`` or ``"network"``
        :type tier: str
        :param src_path: the managed-repository source path, if known
        :type src_path: str | None
        :param objective_power: nominal magnification, if recorded
        :type objective_power: float | None
        """
        self.image_id = image_id
        self.width0 = width0
        self.height0 = height0
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.level_dims = level_dims
        self.tier = tier
        self.src_path = src_path
        self.objective_power = objective_power
        self._live: dict | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_live"] = None
        return state

    def __repr__(self) -> str:
        return (
            f"OmeroWSI({image_uri(self.image_id)}, {self.width0}x{self.height0}, "
            f"{len(self.level_dims)} levels, tier={self.tier})"
        )


class OmeroWSIReader(BaseWSIReader):
    """Read patches from OMERO images through ``lavlab tile``'s pixel tiers.

    ``read`` takes an ``omero://<id>`` string (the form
    :func:`lavlab.monai.build_datalist` writes) or a bare integer image ID.
    Everything else follows ``monai.data.BaseWSIReader``: ``location`` is
    ``(row, col)`` in level-0 pixels, ``size`` is ``(height, width)`` at
    ``level``, and ``get_size`` / ``get_mpp`` answer ``(height, width)`` /
    ``(y, x)``.

    lavlab tiles sit at a target mpp, usually between two pyramid levels:
    the region is read from the next finer level and shrunk. Pass
    *resize_to* to have every patch resized to that edge with
    :func:`lavlab.tiling.resize_tile`, exactly as ``lavlab tile`` does;
    without it a patch is returned at its level's resolution, as every other
    MONAI reader does.

    Each process opens its own OMERO session on first use -- see the module
    docstring -- from *conn_factory* if given, else from *creds*, else from
    the ``OMERO_USER`` / ``OMERO_PASSWORD`` / ``OMERO_HOST`` / ``OMERO_PORT``
    environment variables.

    :param creds: OMERO credentials; pickled into worker processes
    :type creds: lavlab.config.OmeroCreds | None
    :param conn_factory: zero-argument callable returning a connected
        ``BlitzGateway``, used instead of *creds*; must pickle if workers
        are spawned rather than forked
    :param force_local: read JPEG-2000 sources locally anyway
        (``lavlab tile --force-local``)
    :type force_local: bool
    :param store_count: parallel raw-pixels stores per network read
        (``lavlab tile --connections``)
    :type store_count: int | None
    :param resize_to: output patch edge in pixels, or None for none
    :type resize_to: int | None
    :param kwargs: ``monai.data.BaseWSIReader`` arguments (``level``,
        ``channel_dim``, ``dtype``, ``mode``, ...)
    """

    # OMERO images are not files: verify_suffix is overridden instead. Typed
    # as BaseWSIReader declares it.
    supported_suffixes: list[str] = []  # noqa: RUF012
    backend = "omero"

    def __init__(
        self,
        *,
        creds=None,
        conn_factory=None,
        force_local: bool = False,
        store_count: int | None = None,
        resize_to: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.creds = creds
        self.conn_factory = conn_factory
        self.force_local = force_local
        self.store_count = store_count
        self.resize_to = resize_to
        self._session: dict = {"pid": None, "conn": None, "group": None}
        self._tls = threading.local()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_session"] = {"pid": None, "conn": None, "group": None}
        state.pop("_tls", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._tls = threading.local()

    # -- connection ---------------------------------------------------------

    def connection(self):
        """Return this process's OMERO gateway, logging in if it has none.

        A gateway inherited from a parent process is dropped, not closed,
        and a dead one is replaced. The new one is closed when the reader
        is garbage-collected or the process exits, including a
        ``DataLoader`` worker.

        :return: a connected ``BlitzGateway``
        """
        with _PROCESS_LOCK:
            pid = os.getpid()
            session = self._session
            if session["pid"] != pid:
                session.update(pid=pid, conn=None, group=None)
            conn = session["conn"]
            if conn is not None:
                try:
                    alive = conn.isConnected()
                except Exception:
                    alive = False
                if alive:
                    return conn
                _close_owned(conn, pid)
            factory = self.conn_factory or functools.partial(_connect, self.creds)
            conn = factory()
            session.update(conn=conn, group=None)
            multiprocessing.util.Finalize(
                self, _close_owned, args=(conn, pid), exitpriority=10
            )
            return conn

    def close(self) -> None:
        """Close this process's OMERO session, if it opened one."""
        with _PROCESS_LOCK:
            session = self._session
            if session["pid"] == os.getpid() and session["conn"] is not None:
                _close_owned(session["conn"], session["pid"])
            session.update(conn=None, group=None)

    def _drop_connection(self) -> None:
        with _PROCESS_LOCK:
            session = self._session
            if session["pid"] == os.getpid() and session["conn"] is not None:
                _close_owned(session["conn"], session["pid"])
            session.update(conn=None, group=None)

    def _live(self, wsi: OmeroWSI) -> dict:
        """Return the per-process live state of *wsi*, (re)building it as needed."""
        with _PROCESS_LOCK:
            conn = self.connection()
            live = wsi._live
            if live is None or live["pid"] != os.getpid() or live["conn"] is not conn:
                image = _get_image(conn, wsi.image_id)
                live = {
                    "pid": os.getpid(),
                    "conn": conn,
                    "image": image,
                    "readers": {},
                }
                wsi._live = live
            return live

    def _enter_group(self, conn, image) -> None:
        """Switch the session into *image*'s group before a stateful service."""
        from lavlab.omero_tiles import _switch_group_before_stateful_service

        group_id = image.details.group.id.val
        if self._session.get("group") != group_id:
            _switch_group_before_stateful_service(conn, image)
            self._session["group"] = group_id

    # -- BaseWSIReader ------------------------------------------------------

    def verify_suffix(self, filename) -> bool:
        """Return whether every item of *filename* names an OMERO image.

        :param filename: an image name or a sequence of them
        :return: whether :meth:`read` would accept them all
        :rtype: bool
        """
        items = filename if isinstance(filename, (list, tuple)) else [filename]
        try:
            for item in items:
                parse_image_id(item)
        except ValueError:
            return False
        return True

    def read(self, data, **kwargs):
        """Open one OMERO image, or several.

        Picks the tier with :func:`lavlab.tiling.choose_tier` (so JPEG-2000
        sources go over the network unless *force_local*), and reads the
        pyramid from the local source, falling forward to OMERO's own
        description of it with a WARNING if the source can't be read.

        :param data: ``omero://<id>`` or an int, or a list/tuple of those
        :param kwargs: unused; accepted for the ``ImageReader`` interface
        :return: an :class:`OmeroWSI`, or a list of them for a sequence
        :raises ValueError: for an unknown or unreadable image name
        """
        items = data if isinstance(data, (list, tuple)) else [data]
        handles = [self._open(parse_image_id(item)) for item in items]
        return handles if len(handles) > 1 else handles[0]

    def _open(self, image_id: int) -> OmeroWSI:
        from lavlab.omero_client import switch_to_object_group

        with _PROCESS_LOCK:
            conn = self.connection()
            image = _get_image(conn, image_id)
            switch_to_object_group(conn, image)
            tier, src_path = choose_tier(conn, image_id, force_local=self.force_local)
            level_dims: list[tuple[int, int]] = []
            if tier == "local" and src_path is not None:
                try:
                    level_dims = local_level_dims(src_path)
                except _local_source_errors() as exc:
                    warn_local_fallback(image_id, src_path, exc)
                    tier = "network"
            if tier == "network":
                level_dims = network_level_dims(conn, image)
            pixel_x, pixel_y = image_pixel_sizes(image)
            wsi = OmeroWSI(
                image_id,
                width0=int(image.getSizeX()),
                height0=int(image.getSizeY()),
                pixel_size_x=pixel_x,
                pixel_size_y=pixel_y,
                level_dims=level_dims,
                tier=tier,
                src_path=src_path,
                objective_power=_objective_power(image),
            )
            wsi._live = {
                "pid": os.getpid(),
                "conn": conn,
                "image": image,
                "readers": {},
            }
            return wsi

    def get_size(self, wsi: OmeroWSI, level: int) -> tuple[int, int]:
        """Return ``(height, width)`` of one pyramid level.

        :param wsi: an open image
        :type wsi: OmeroWSI
        :param level: the level, 0 = full resolution
        :type level: int
        :return: ``(height, width)``
        :rtype: tuple[int, int]
        """
        width, height = wsi.level_dims[level]
        return int(height), int(width)

    def get_level_count(self, wsi: OmeroWSI) -> int:
        """Return how many pyramid levels the image has.

        :param wsi: an open image
        :type wsi: OmeroWSI
        :return: the level count
        :rtype: int
        """
        return len(wsi.level_dims)

    def get_downsample_ratio(self, wsi: OmeroWSI, level: int) -> float:
        """Return the level's downsample from full resolution.

        Full-resolution width over the level's width -- the ratio
        :func:`lavlab.tiling.select_level` uses -- so a region read at
        ``location`` lands on the same level pixels ``lavlab tile`` reads.

        :param wsi: an open image
        :type wsi: OmeroWSI
        :param level: the level
        :type level: int
        :return: level-0 pixels per level pixel
        :rtype: float
        """
        return wsi.width0 / wsi.level_dims[level][0]

    def get_file_path(self, wsi: OmeroWSI) -> str:
        """Return the image's ``omero://<id>`` name.

        :param wsi: an open image
        :type wsi: OmeroWSI
        :return: e.g. ``omero://1962``
        :rtype: str
        """
        return image_uri(wsi.image_id)

    def get_mpp(self, wsi: OmeroWSI, level: int) -> tuple[float, float]:
        """Return ``(y, x)`` micrometres per pixel at a level.

        :param wsi: an open image
        :type wsi: OmeroWSI
        :param level: the level
        :type level: int
        :return: ``(mpp_y, mpp_x)``
        :rtype: tuple[float, float]
        :raises lavlab.tiling.MissingPixelSizeError: if OMERO has no
            physical pixel size for the image -- the error ``lavlab tile
            --mpp`` gives for the same image
        """
        if not wsi.pixel_size_x:
            raise MissingPixelSizeError(
                f"image {wsi.image_id}: {MISSING_PIXEL_SIZE_MESSAGE}"
            )
        ratio = self.get_downsample_ratio(wsi, level)
        pixel_y = wsi.pixel_size_y or wsi.pixel_size_x
        return pixel_y * ratio, wsi.pixel_size_x * ratio

    def get_power(self, wsi: OmeroWSI, level: int) -> float:
        """Return the objective power at a level, from OMERO's objective settings.

        :param wsi: an open image
        :type wsi: OmeroWSI
        :param level: the level
        :type level: int
        :return: nominal magnification divided by the level's downsample
        :rtype: float
        :raises ValueError: if the image records no objective magnification
        """
        if not wsi.objective_power:
            raise ValueError(
                f"image {wsi.image_id} has no objective magnification recorded in "
                "OMERO; use `level` or `mpp` instead."
            )
        return wsi.objective_power / self.get_downsample_ratio(wsi, level)

    def _get_patch(
        self,
        wsi: OmeroWSI,
        location: tuple[int, int],
        size: tuple[int, int],
        level: int,
        dtype,
        mode: str,
    ) -> np.ndarray:
        """Read one patch through ``lavlab tile``'s region readers.

        :param wsi: an open image
        :type wsi: OmeroWSI
        :param location: ``(row, col)`` of the top-left corner, level 0
        :type location: tuple[int, int]
        :param size: ``(height, width)`` at *level*; lavlab regions are square
        :type size: tuple[int, int]
        :param level: the pyramid level
        :type level: int
        :param dtype: output dtype
        :param mode: ``"RGB"`` or ``"RGBA"``
        :type mode: str
        :return: the patch, channels at ``channel_dim``
        :rtype: numpy.ndarray
        :raises ValueError: for a non-square *size*
        """
        if int(size[0]) != int(size[1]):
            raise ValueError(
                f"OmeroWSIReader reads square regions, as lavlab tile cuts them; "
                f"got size {tuple(size)}."
            )
        claim_vips()
        read_size = int(size[0])
        y0, x0 = (int(v) for v in location)
        plan = LevelPlan(
            index=int(level),
            downsample=self.get_downsample_ratio(wsi, level),
            read_size=read_size,
            scale=1.0,
        )
        region = self._read_region(wsi, plan, x0, y0)
        if self.resize_to:
            region = resize_tile(region, int(self.resize_to))

        bands = region.shape[2]
        if mode == "RGBA":
            if bands != 4:
                raise ValueError(
                    f"image {wsi.image_id} has {bands} channels; 'RGBA' mode needs 4."
                )
        elif bands in (3, 4):
            region = region[:, :, :3]
        patch = np.moveaxis(region, -1, self.channel_dim)
        return np.ascontiguousarray(patch, dtype=dtype)

    def _read_region(
        self, wsi: OmeroWSI, plan: LevelPlan, x0: int, y0: int
    ) -> np.ndarray:
        from lavlab.omero_client import describe_error, is_conn_error

        w0 = h0 = int(round(plan.read_size * plan.downsample))
        for attempt in (1, 2):
            try:
                if wsi.tier == "local":
                    try:
                        region = self._read_local(wsi, plan, x0, y0, w0, h0)
                        self._tls.tier = "local"
                        return region
                    except _local_source_errors() as exc:
                        self._fall_forward(wsi, exc)
                region = self._read_network(wsi, plan, x0, y0, w0, h0)
                self._tls.tier = "network"
                return region
            except Exception as exc:
                if attempt == 1 and is_conn_error(exc):
                    log.warning(
                        "Image %d: OMERO connection lost (%s); reconnecting.",
                        wsi.image_id,
                        describe_error(exc),
                    )
                    self._drop_connection()
                    wsi._live = None
                    continue
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _read_local(self, wsi, plan, x0, y0, w0, h0) -> np.ndarray:
        key = ("local", plan.index, plan.read_size)
        with _PROCESS_LOCK:
            live = wsi._live
            if live is None or live["pid"] != os.getpid():
                live = {"pid": os.getpid(), "conn": None, "image": None, "readers": {}}
                wsi._live = live
            reader = live["readers"].get(key)
            if reader is None:
                reader = LocalRegionReader(
                    wsi.src_path, plan, wsi.level_dims[plan.index]
                )
                live["readers"][key] = reader
        return reader.read(x0, y0, w0, h0)

    def _fall_forward(self, wsi: OmeroWSI, exc: BaseException) -> None:
        """Switch *wsi* to tier 3 after its local source failed mid-read."""
        with _PROCESS_LOCK:
            if wsi.tier != "local":
                return
            warn_local_fallback(wsi.image_id, wsi.src_path, exc)
            live = self._live(wsi)
            network_dims = network_level_dims(live["conn"], live["image"])
            if [tuple(d) for d in network_dims] != [tuple(d) for d in wsi.level_dims]:
                raise TilingError(
                    f"image {wsi.image_id}: the local source's pyramid "
                    f"{wsi.level_dims} differs from OMERO's {network_dims}, so "
                    "patch levels and sizes chosen against one cannot be read "
                    "from the other."
                ) from exc
            wsi.tier = "network"

    def _read_network(self, wsi, plan, x0, y0, w0, h0) -> np.ndarray:
        # A session has one group context and NetworkRegionReader keeps a
        # per-read tile cache, so network reads in one process take turns;
        # DataLoader workers each have their own session and run in parallel.
        key = ("network", plan.index, plan.read_size)
        with _PROCESS_LOCK:
            live = self._live(wsi)
            conn, image = live["conn"], live["image"]
            self._enter_group(conn, image)
            reader = live["readers"].get(key)
            if reader is None:
                if self.store_count:
                    reader = NetworkRegionReader(
                        conn, image, plan, store_count=self.store_count
                    )
                else:
                    reader = NetworkRegionReader(conn, image, plan)
                live["readers"][key] = reader
            # Prefetching just this region also drops the previous patch's
            # tiles, so memory stays flat over an epoch of random reads.
            reader.prefetch([(x0, y0, w0, h0)])
            return reader.read(x0, y0, w0, h0)

    def _get_metadata(self, wsi, patch, location, size, level) -> dict:
        """Add the pixel ``tier`` that served the patch to MONAI's metadata."""
        metadata = super()._get_metadata(
            wsi=wsi, patch=patch, location=location, size=size, level=level
        )
        metadata["tier"] = getattr(self._tls, "tier", wsi.tier)
        return metadata
