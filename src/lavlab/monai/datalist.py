# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Turn OMERO image IDs into the sample dicts MONAI's WSI datasets consume.

The selection itself -- grid, tissue mask, ROI coverage, ``erode``,
exclusions, the benign rule, seeded caps -- is
:func:`lavlab.tiling.select_tiles`, the same function ``lavlab tile`` runs.
This module only translates its answer into MONAI's vocabulary, so a
datalist and an extraction made with the same settings name the same tiles.

Nothing here imports MONAI or torch: a datalist can be built, inspected and
saved without the ``monai`` extra installed.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from collections.abc import Iterable, Sequence

from lavlab.commands.tile import DEFAULT_BACKGROUND_LABEL
from lavlab.naming import get_stem
from lavlab.tiling import (
    SlideSelection,
    TileParams,
    TilingError,
    build_alias_lookup,
    load_label_map,
    map_text_value,
    select_tiles,
    subject_dir_name,
    tile_center,
)

log = logging.getLogger(__name__)

#: How an OMERO image is named wherever MONAI expects a file path.
IMAGE_SCHEME = "omero://"

#: The keys every sample carries besides the image(s), in both
#: :class:`lavlab.monai.OmeroPatchDataset` and
#: :class:`lavlab.monai.LavlabTileDataset`. Mirrors the manifest's
#: provenance columns; ``roi_id`` / ``shape_id`` use -1 and ``mpp`` NaN where
#: the manifest leaves a blank, so a batch collates.
SAMPLE_KEYS = (
    "label",
    "label_name",
    "omero_image_id",
    "case_id",
    "slide_id",
    "roi_id",
    "shape_id",
    "mpp",
    "tile_size",
    "x",
    "y",
    "x0",
    "y0",
    "w0",
    "h0",
    "level",
    "coverage",
    "tissue_frac",
    "tier",
)


# The process that first ran libvips through lavlab.monai. libvips does not
# survive fork once it has run -- a forked child's first libvips call hangs,
# on Linux as on macOS -- and building a datalist already runs it (the
# analysis thumbnail), as does reading a patch. Deliberately not reset in a
# forked child: inheriting it is exactly the condition to detect.
_VIPS_PID: int | None = None

FORKED_AFTER_VIPS = (
    "this process was forked from process {parent}, which had already used "
    "libvips (building a lavlab datalist or reading patches does); libvips "
    "cannot be used after fork and would hang here. Create the DataLoader with "
    "multiprocessing_context='spawn' (or 'forkserver')."
)


def claim_vips() -> None:
    """Record that this process runs libvips; fail fast in a fork of one that did.

    :raises RuntimeError: in a process forked from one that already used
        libvips through ``lavlab.monai``, instead of hanging
    """
    global _VIPS_PID
    pid = os.getpid()
    if _VIPS_PID is None:
        _VIPS_PID = pid
    elif _VIPS_PID != pid:
        raise RuntimeError(FORKED_AFTER_VIPS.format(parent=_VIPS_PID))


def image_uri(image_id: int) -> str:
    """Return the ``omero://<id>`` name of an OMERO image.

    :param image_id: the OMERO image ID
    :type image_id: int
    :return: e.g. ``omero://1962``
    :rtype: str
    """
    return f"{IMAGE_SCHEME}{int(image_id)}"


def parse_image_id(data) -> int:
    """Return the image ID named by an ``omero://<id>`` string or an int.

    ``omero://<id>`` is the canonical form -- it is what :func:`build_datalist`
    writes into ``image`` and what ``OmeroWSIReader.get_file_path`` reports.
    A bare integer is accepted as a convenience. Anything else, including a
    bare numeric *string*, is rejected so a file path handed to the wrong
    reader fails loudly.

    :param data: ``omero://1962`` or ``1962``
    :return: the image ID
    :rtype: int
    :raises ValueError: if *data* names no OMERO image
    """
    if isinstance(data, bool):
        raise ValueError(f"not an OMERO image: {data!r}")
    if isinstance(data, str):
        if data.startswith(IMAGE_SCHEME):
            text = data[len(IMAGE_SCHEME) :].strip("/")
            if text.isdigit():
                return int(text)
        raise ValueError(
            f"not an OMERO image: {data!r}; expected '{IMAGE_SCHEME}<image id>' "
            "or an integer image ID."
        )
    try:
        image_id = int(data)
    except (TypeError, ValueError):
        raise ValueError(f"not an OMERO image: {data!r}") from None
    if image_id != data:
        raise ValueError(f"not an OMERO image: {data!r}")
    return image_id


def tile_params(
    *,
    labels: Sequence[str] | None = None,
    all_labels: bool = False,
    size: int | None = None,
    mpp: float | None = None,
    scales: Sequence[float] | None = None,
    downsample: float | None = None,
    overlap: int = 0,
    erode: float | str | None = None,
    min_coverage: float | None = None,
    tissue_threshold: float | None = None,
    exclude_text: Sequence[str] | None = None,
    labels_yaml: str | None = None,
    background_label: str | None = DEFAULT_BACKGROUND_LABEL,
    background_margin: float | None = None,
    max_tiles: int | None = None,
    max_tiles_per_label: int | None = None,
    max_background_tiles: int | None = None,
    seed: int = 0,
) -> TileParams:
    """Return the :class:`lavlab.tiling.TileParams` ``lavlab tile --roi`` would use.

    Each argument is the ``lavlab tile`` flag of the same meaning, and is run
    through that command's own validation and defaults
    (``lavlab.commands.tile._validate_args`` / ``_build_params``), so a
    default changed for the CLI changes here too and a combination the CLI
    rejects is rejected here.

    ============================  =====================================
    argument                      ``lavlab tile`` flag
    ============================  =====================================
    ``labels``                    ``-t`` (repeated)
    ``all_labels``                ``--all``
    ``size``                      ``--size``
    ``mpp`` / ``scales``          ``--mpp`` / ``--scales``
    ``downsample``                ``--downsample``
    ``overlap``                   ``--overlap``
    ``erode``                     ``--erode`` (um, 0, or ``"auto"``)
    ``min_coverage``              ``--min-coverage``
    ``tissue_threshold``          ``--tissue-threshold``
    ``exclude_text``              ``--exclude-text`` (repeated)
    ``labels_yaml``               ``--labels``
    ``background_label``          ``--background-label``; None is
                                  ``--no-background``
    ``background_margin``         ``--background-margin``
    ``max_tiles``                 ``--max-tiles``
    ``max_tiles_per_label``       ``--max-tiles-per-label``
    ``max_background_tiles``      ``--max-background-tiles``
    ``seed``                      ``--seed``
    ============================  =====================================

    :return: the settings
    :rtype: TileParams
    :raises ValueError: for a combination ``lavlab tile`` would refuse
    """
    from lavlab.commands import tile as tile_cmd

    if size is None:
        size = tile_cmd.DEFAULT_SIZE
    parsed_scales = None
    if scales is not None:
        try:
            parsed_scales = tile_cmd._parse_scales(",".join(str(s) for s in scales))
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"scales: {exc}") from None
    parsed_erode = None
    if erode is not None:
        try:
            parsed_erode = tile_cmd._parse_erode(str(erode))
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"erode: {exc}") from None
    if sum(v is not None for v in (mpp, scales, downsample)) > 1:
        raise ValueError("pass at most one of mpp, scales and downsample.")

    args = argparse.Namespace(
        whole=False,
        roi=True,
        all=bool(all_labels),
        text_filter=[str(t).lower() for t in (labels or [])],
        size=int(size),
        mpp=mpp,
        scales=parsed_scales,
        downsample=downsample,
        overlap=int(overlap),
        stride=None,
        tissue_thresh=(
            tile_cmd.DEFAULT_TISSUE_THRESH
            if tissue_threshold is None
            else float(tissue_threshold)
        ),
        min_coverage=(
            tile_cmd.DEFAULT_MIN_COVERAGE
            if min_coverage is None
            else float(min_coverage)
        ),
        erode=parsed_erode,
        exclude_text=(
            None if exclude_text is None else [str(t).lower() for t in exclude_text]
        ),
        labels=labels_yaml,
        background_label=background_label,
        background_margin=background_margin,
        no_background=background_label is None,
        max_tiles=max_tiles,
        max_tiles_per_label=max_tiles_per_label,
        max_background_tiles=max_background_tiles,
        seed=int(seed),
        format=None,
        quality=None,
        coords_only=False,
        connections=None,
        workers=1,
    )
    try:
        tile_cmd._validate_args(args)
    except SystemExit as exc:
        message = str(exc.code).removeprefix("error: ")
        raise ValueError(_python_names(message)) from None
    return tile_cmd._build_params(args)


#: How :func:`tile_params` spells each ``lavlab tile`` flag its errors can
#: name. The errors come from the CLI's own validation, so they name flags;
#: a Python caller passed keyword arguments.
_PYTHON_NAMES = {
    "--roi": "tile selection",
    "--all": "all_labels=True",
    "--text-filter": "labels=[...]",
    "--size": "size",
    "--mpp": "mpp",
    "--scales": "scales",
    "--downsample": "downsample",
    "--overlap": "overlap",
    "--erode": "erode",
    "--min-coverage": "min_coverage",
    "--tissue-thresh": "tissue_threshold",
    "--exclude-text": "exclude_text",
    "--labels": "labels_yaml",
    "--background-label": "background_label",
    "--no-background": "background_label=None",
    "--background-margin": "background_margin",
    "--max-tiles": "max_tiles",
    "--max-tiles-per-label": "max_tiles_per_label",
    "--max-background-tiles": "max_background_tiles",
    "--seed": "seed",
}


def _python_names(message: str) -> str:
    """Rewrite the ``lavlab tile`` flags in *message* as :func:`tile_params` arguments."""
    return re.sub(
        r"--[a-z][a-z-]*",
        lambda m: _PYTHON_NAMES.get(m.group(0), m.group(0)),
        message,
    )


def default_classes(params: TileParams) -> list[str] | None:
    """Return the class order a run's labels imply, or None if it can't be known.

    With explicit ``labels`` the classes are those labels' folder names, in
    the order given, then the background label -- the same answer for every
    datalist built with those settings, so indices agree across sections and
    folds. With ``all_labels`` the classes depend on what the slides carry;
    None tells :func:`build_datalist` to use the sorted names it finds, and
    callers that build several datalists should pass ``classes`` instead.

    :param params: the run's settings
    :type params: TileParams
    :return: class names in index order, or None
    :rtype: list[str] | None
    """
    if params.include_all:
        return None
    lookup = build_alias_lookup(load_label_map(params.labels))
    names: list[str] = []
    for text in params.text_filter:
        folder = map_text_value(text, lookup, include_all=True)
        if folder is not None and folder not in names:
            names.append(folder)
    if params.background_label and params.background_label not in names:
        names.append(params.background_label)
    return names


def _blank_to(value, missing):
    return missing if value is None else value


def selection_samples(
    selection: SlideSelection,
    params: TileParams,
    *,
    case_id: str,
    slide_id: str,
    classes: Sequence[str],
) -> list[dict]:
    """Translate one slide's :class:`lavlab.tiling.SlideSelection` into samples.

    One sample per kept tile location. Its ``image`` / ``location`` /
    ``size`` / ``level`` are MONAI's WSI keys for the finest scale: the
    region of ``size`` pixels at pyramid ``level`` whose top-left corner is
    ``location`` -- ``(row, col)`` in level-0 pixels, i.e. ``(y0, x0)``. A
    ``--scales`` run adds ``extra_scales``, the same four keys (plus ``mpp``
    and the level-0 box) for each coarser, concentric crop, finest first;
    the datasets return those as ``image_1``, ``image_2``, ...

    The remaining keys are :data:`SAMPLE_KEYS`, valued exactly as the
    manifest row ``lavlab tile`` writes for the same tile.

    :param selection: the slide's selection, with ``status == "ok"``
    :type selection: SlideSelection
    :param params: the settings that produced it
    :type params: TileParams
    :param case_id: grouping key, as written to the manifest's ``case_id``
    :type case_id: str
    :param slide_id: the slide stem, as written to ``slide_id``
    :type slide_id: str
    :param classes: class names in index order
    :type classes: Sequence[str]
    :return: the samples, row-major
    :rtype: list[dict]
    :raises ValueError: if a tile's label is not in *classes*
    """
    uri = image_uri(selection.image_id)
    per_scale = [
        (
            selection.records_at(index),
            selection.plans[index],
            selection.output_mpp(index),
        )
        for index in range(len(selection.scale_ds))
    ]

    samples = []
    for position, record in enumerate(selection.records):
        if record.label not in classes:
            raise ValueError(
                f"image {selection.image_id}: label '{record.label}' is not in "
                f"classes {list(classes)}; add it, or narrow the labels selected."
            )
        geometry = []
        for scaled, plan, mpp in per_scale:
            tile = scaled[position].tile
            geometry.append(
                {
                    "location": [tile.y0, tile.x0],
                    "size": [plan.read_size, plan.read_size],
                    "level": plan.index,
                    "mpp": math.nan if mpp is None else round(mpp, 6),
                    "x0": tile.x0,
                    "y0": tile.y0,
                    "w0": tile.w0,
                    "h0": tile.h0,
                }
            )
        finest = geometry[0]
        x, y = tile_center(record.tile)
        roi_id = selection.roi_id(record)
        sample = {
            "image": uri,
            "location": finest["location"],
            "size": finest["size"],
            "level": finest["level"],
            "label": classes.index(record.label),
            "label_name": record.label,
            "omero_image_id": selection.image_id,
            "case_id": case_id,
            "slide_id": slide_id,
            "roi_id": _blank_to(roi_id, -1),
            "shape_id": _blank_to(record.shape_id, -1),
            "mpp": finest["mpp"],
            "tile_size": params.size,
            "x": x,
            "y": y,
            "x0": finest["x0"],
            "y0": finest["y0"],
            "w0": finest["w0"],
            "h0": finest["h0"],
            "coverage": round(record.coverage, 6),
            "tissue_frac": round(record.tissue_frac, 6),
            "tier": selection.tier,
        }
        if len(geometry) > 1:
            sample["scales"] = [g["mpp"] for g in geometry]
            sample["extra_scales"] = geometry[1:]
        samples.append(sample)
    return samples


def _connect(creds):
    """Open a gateway from *creds*, or from the ``OMERO_*`` environment variables.

    :raises lavlab.config.ConfigError: if a credential is in neither, worded
        for a Python caller rather than the CLI's command line
    """
    from lavlab.config import ConfigError, resolve_creds
    from lavlab.omero_client import connect

    if creds is None:
        try:
            creds = resolve_creds()
        except ConfigError as exc:
            raise ConfigError(
                str(exc).replace(
                    "pass it on the command line",
                    "pass creds=lavlab.config.OmeroCreds(user, password, host, port)",
                )
            ) from None
    return connect(creds)


def build_datalist(
    image_ids: int | Iterable[int] | None = None,
    *,
    group: int | None = None,
    conn=None,
    creds=None,
    labels: Sequence[str] | None = None,
    all_labels: bool = False,
    classes: Sequence[str] | None = None,
    size: int | None = None,
    mpp: float | None = None,
    scales: Sequence[float] | None = None,
    downsample: float | None = None,
    overlap: int = 0,
    erode: float | str | None = None,
    min_coverage: float | None = None,
    tissue_threshold: float | None = None,
    exclude_text: Sequence[str] | None = None,
    labels_yaml: str | None = None,
    background_label: str | None = DEFAULT_BACKGROUND_LABEL,
    background_margin: float | None = None,
    max_tiles: int | None = None,
    max_tiles_per_label: int | None = None,
    max_background_tiles: int | None = None,
    seed: int = 0,
    force_local: bool = False,
    skip_failed: bool = False,
) -> list[dict]:
    """Build a MONAI WSI datalist from OMERO, choosing tiles as ``lavlab tile`` does.

    Every slide goes through :func:`lavlab.tiling.select_tiles` with the
    settings :func:`tile_params` derives from the keyword arguments, so the
    result is the same set of tile locations and labels that
    ``lavlab tile <id> --roi ...`` with the matching flags would cut (and
    that ``--dry-run`` counts). Only analysis-resolution pixels are read.

    A slide with no ROIs is skipped with the CLI's warning, not tiled as
    benign; one smaller than a tile contributes nothing.

    :param image_ids: one image ID or several; not with *group*
    :type image_ids: int | Iterable[int] | None
    :param group: every image in this OMERO group, like ``lavlab tile batch -g``
    :type group: int | None
    :param conn: a connected gateway to use; by default one is opened from
        *creds* (or the ``OMERO_*`` environment variables) and closed again
    :param creds: credentials for that connection
    :type creds: lavlab.config.OmeroCreds | None
    :param classes: class names in label-index order; by default the
        selected labels in the order given, then the background label (see
        :func:`default_classes`)
    :type classes: Sequence[str] | None
    :param force_local: read JPEG-2000 sources locally anyway (``--force-local``)
    :type force_local: bool
    :param skip_failed: log and skip a slide that cannot be tiled, as
        ``lavlab tile batch`` does, instead of raising
    :type skip_failed: bool
    :return: one dict per tile location; see :func:`selection_samples`
    :rtype: list[dict]
    :raises ValueError: for settings ``lavlab tile`` would refuse
    :raises lavlab.tiling.MissingPixelSizeError: with *mpp* on an image that
        has no pixel size recorded (unless *skip_failed*)
    """
    params = tile_params(
        labels=labels,
        all_labels=all_labels,
        size=size,
        mpp=mpp,
        scales=scales,
        downsample=downsample,
        overlap=overlap,
        erode=erode,
        min_coverage=min_coverage,
        tissue_threshold=tissue_threshold,
        exclude_text=exclude_text,
        labels_yaml=labels_yaml,
        background_label=background_label,
        background_margin=background_margin,
        max_tiles=max_tiles,
        max_tiles_per_label=max_tiles_per_label,
        max_background_tiles=max_background_tiles,
        seed=seed,
    )
    datalist, _classes = datalist_for_params(
        params,
        image_ids,
        group=group,
        conn=conn,
        creds=creds,
        classes=classes,
        force_local=force_local,
        skip_failed=skip_failed,
    )
    return datalist


def datalist_for_params(
    params: TileParams,
    image_ids: int | Iterable[int] | None = None,
    *,
    group: int | None = None,
    conn=None,
    creds=None,
    classes: Sequence[str] | None = None,
    force_local: bool = False,
    skip_failed: bool = False,
) -> tuple[list[dict], list[str]]:
    """:func:`build_datalist` from ready-made settings, also returning the classes.

    :param params: the settings, e.g. from :func:`tile_params`
    :type params: TileParams
    :return: ``(datalist, class names in label-index order)``
    :rtype: tuple[list[dict], list[str]]
    """
    from lavlab.omero_client import (
        ALL_GROUPS,
        iter_image_ids,
        switch_to_object_group,
    )

    if (image_ids is None) == (group is None):
        raise ValueError("pass exactly one of image_ids and group.")
    fixed_classes = list(classes) if classes is not None else default_classes(params)

    own_conn = conn is None
    if own_conn:
        conn = _connect(creds)
    try:
        if group is not None:
            ids = list(iter_image_ids(conn, group))
        elif isinstance(image_ids, int):
            ids = [image_ids]
        else:
            ids = [int(i) for i in image_ids or ()]

        selected: list[tuple[SlideSelection, str, str]] = []
        for image_id in ids:
            # Unscoped: the previous image may have left another group selected.
            conn.SERVICE_OPTS.setOmeroGroup(ALL_GROUPS)
            image = conn.getObject("Image", image_id)
            if image is None:
                if not skip_failed:
                    raise ValueError(f"OMERO image {image_id} not found.")
                log.warning("Image %d not found, skipping.", image_id)
                continue
            switch_to_object_group(conn, image)
            claim_vips()
            try:
                selection = select_tiles(conn, image, params, force_local=force_local)
            except TilingError as exc:
                if not skip_failed:
                    raise
                log.warning("Image %d: %s", image_id, exc)
                continue
            if selection.status != "ok":
                continue
            name = image.getName()
            selected.append((selection, subject_dir_name(name), get_stem(name)))
    finally:
        if own_conn:
            conn.close()

    if fixed_classes is None:
        fixed_classes = sorted({r.label for s, _c, _sl in selected for r in s.records})

    datalist: list[dict] = []
    for selection, case_id, slide_id in selected:
        datalist.extend(
            selection_samples(
                selection,
                params,
                case_id=case_id,
                slide_id=slide_id,
                classes=fixed_classes,
            )
        )
    return datalist, fixed_classes
