# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""MONAI datasets over lavlab tiles: read live from OMERO, or from disk.

Both follow the ``monai.apps`` dataset convention (``MedNISTDataset``,
``DecathlonDataset``): a ``CacheDataset`` with ``section``, ``seed``,
``val_frac`` / ``test_frac``, the cache arguments, and a
``_split_datalist`` that ``monai.apps.CrossValidation`` can override. Both
yield the same sample dict -- ``image`` (a channel-first uint8
``MetaTensor``), an integer ``label`` and the provenance keys in
:data:`lavlab.monai.datalist.SAMPLE_KEYS` -- so a model trained on one can
be evaluated on the other.

Sections are drawn over cases, never tiles. Tiles from one patient on both
sides of a split inflate every metric, so ``_split_datalist`` is handed one
entry per ``case_id`` (``{"case_id": ..., "items": [...]}``) and its answer
is flattened back into tiles; an override such as ``CrossValidation``'s
therefore partitions cases too.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import sys
from collections.abc import Callable, Sequence

import numpy as np

from lavlab.commands.tile import DEFAULT_BACKGROUND_LABEL
from lavlab.monai._optional import EXTRA_HINT, is_missing_extra
from lavlab.monai.datalist import (
    SAMPLE_KEYS,
    datalist_for_params,
    tile_params,
)
from lavlab.monai.reader import OmeroWSIReader
from lavlab.tiling import MANIFEST_COLUMNS

try:
    from monai.data import CacheDataset, MetaTensor, PatchWSIDataset
    from monai.transforms import Compose, LoadImaged, Randomizable, Transform
except ImportError as exc:  # pragma: no cover - depends on the install
    if is_missing_extra(exc):
        raise ImportError(EXTRA_HINT) from exc
    raise

log = logging.getLogger(__name__)

SECTIONS = ("training", "validation", "test")

# What lavlab.tiling.scaled_tile_filename names a --scales tile.
_SCALED_NAME = re.compile(r"_x(-?\d+)_y(-?\d+)_mpp([0-9.eE+-]+)\.[A-Za-z0-9]+$")


def group_by_case(items: Sequence[dict]) -> list[dict]:
    """Group samples by ``case_id``, one entry per case.

    :param items: samples, each with a ``case_id``
    :type items: Sequence[dict]
    :return: ``[{"case_id": ..., "items": [...]}, ...]`` sorted by case,
        items in their original order
    :rtype: list[dict]
    """
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(str(item["case_id"]), []).append(item)
    return [{"case_id": case, "items": groups[case]} for case in sorted(groups)]


def _chain(loader: Transform, transform) -> Compose:
    """Put *loader* in front of the user's transform(s), flattened.

    Flattened so ``CacheDataset`` sees the loader and any deterministic
    transforms after it as one cacheable prefix.
    """
    steps: list = [loader]
    if isinstance(transform, Sequence) and not isinstance(transform, str):
        steps.extend(transform)
    elif transform:
        steps.append(transform)
    return Compose(steps).flatten()


class _CaseSplitDataset(Randomizable, CacheDataset):
    """Section selection and caching shared by both lavlab datasets."""

    #: Class names, in label-index order.
    class_names: list[str]

    def _init_split(
        self,
        items: list[dict],
        *,
        section: str,
        transform: Compose,
        seed: int,
        val_frac: float,
        test_frac: float,
        cache_num: int,
        cache_rate: float,
        num_workers: int | None,
        progress: bool,
        copy_cache: bool,
        as_contiguous: bool,
        runtime_cache: bool,
    ) -> None:
        if section not in SECTIONS:
            raise ValueError(
                f"Unsupported section: {section}, available options are "
                f"{list(SECTIONS)}."
            )
        if not 0.0 <= val_frac <= 1.0 or not 0.0 <= test_frac <= 1.0:
            raise ValueError("val_frac and test_frac are fractions between 0 and 1.")
        if val_frac + test_frac > 1.0:
            raise ValueError("val_frac + test_frac must not exceed 1.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)

        groups = group_by_case(items)
        chosen = self._split_datalist(groups)
        if any(not isinstance(g, dict) or "items" not in g for g in chosen):
            raise TypeError(
                "_split_datalist must return case groups ({'case_id', 'items'}), "
                "not individual samples."
            )
        self.case_ids = [str(g["case_id"]) for g in chosen]
        data = [item for group in chosen for item in group["items"]]
        if not data:
            log.warning(
                "Section '%s' is empty: %d case(s) in total, val_frac=%g, "
                "test_frac=%g.",
                section,
                len(groups),
                val_frac,
                test_frac,
            )

        CacheDataset.__init__(
            self,
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            runtime_cache=runtime_cache,
        )

    def randomize(self, data: np.ndarray) -> None:
        self.R.shuffle(data)

    def get_num_classes(self) -> int:
        """Return the number of classes.

        :return: ``len(self.class_names)``
        :rtype: int
        """
        return len(self.class_names)

    def _split_datalist(self, datalist: list[dict]) -> list[dict]:
        """Return this section's cases.

        Cases are shuffled with ``seed``, then the first ``test_frac`` of them
        (rounded down, as ``MedNISTDataset`` does) are ``test``, the next
        ``val_frac`` ``validation`` and the rest ``training``.

        :param datalist: one ``{"case_id", "items"}`` entry per case
        :type datalist: list[dict]
        :return: the entries in this section
        :rtype: list[dict]
        """
        length = len(datalist)
        indices = np.arange(length)
        self.randomize(indices)
        test_length = int(length * self.test_frac)
        val_length = int(length * self.val_frac)
        if self.section == "test":
            section_indices = indices[:test_length]
        elif self.section == "validation":
            section_indices = indices[test_length : test_length + val_length]
        else:
            section_indices = indices[test_length + val_length :]
        return [datalist[i] for i in sorted(section_indices)]


# ---------------------------------------------------------------------------
# A: live from OMERO
# ---------------------------------------------------------------------------


class _OmeroPatchSource(PatchWSIDataset):
    """``PatchWSIDataset`` over a lavlab datalist, returning lavlab samples.

    ``load`` is ``PatchWSIDataset._transform`` for one datalist entry,
    without the transform: every scale's patch is read through MONAI's own
    ``_get_data`` (so its WSI-object cache and reader dispatch apply), and
    the provenance keys are copied across.
    """

    def __init__(self, data: Sequence[dict], reader: OmeroWSIReader):
        super().__init__(
            data=data,
            patch_size=None,
            patch_level=None,
            include_label=False,
            center_location=False,
            reader=reader,
        )

    def load(self, sample: dict) -> dict:
        output = {key: sample[key] for key in SAMPLE_KEYS}
        provenance = {key: sample[key] for key in SAMPLE_KEYS if key != "tier"}
        geometries = [sample, *sample.get("extra_scales", ())]
        for index, geometry in enumerate(geometries):
            image, metadata = self._get_data({"image": sample["image"], **geometry})
            metadata.update(provenance)
            if index == 0:
                output["image"] = MetaTensor(image, meta=metadata)
                output["tier"] = metadata["tier"]
            else:
                metadata["mpp"] = geometry["mpp"]
                output[f"image_{index}"] = MetaTensor(image, meta=metadata)
        if "scales" in sample:
            output["scales"] = list(sample["scales"])
        return output

    def _transform(self, index: int):
        output = self.load(self.data[index])
        return self.transform(output) if self.transform else output


class _ReadOmeroPatch(Transform):
    """The cacheable first step of :class:`OmeroPatchDataset`'s transform."""

    def __init__(self, source: _OmeroPatchSource):
        self.source = source

    def __call__(self, data: dict) -> dict:
        return self.source.load(data)


class OmeroPatchDataset(_CaseSplitDataset):
    """Patches read live from OMERO, chosen exactly as ``lavlab tile`` chooses tiles.

    Builds a datalist with :func:`lavlab.monai.build_datalist` (or takes one
    via *datalist*), keeps this *section*'s cases, and reads each patch on
    demand through :class:`lavlab.monai.OmeroWSIReader`, resized to *size*
    with ``lavlab tile``'s own kernel. Built for exploring and for small or
    fast-changing selections: every epoch pays OMERO's latency on every
    patch. For real training runs, extract once with ``lavlab tile`` and
    use :class:`LavlabTileDataset`.

    Tile selection arguments (*labels* ... *tile_seed*) are
    :func:`lavlab.monai.tile_params`'s, i.e. ``lavlab tile``'s flags;
    *tile_seed* is ``--seed``, while *seed* (as in every ``monai.apps``
    dataset) shuffles the split.

    Each sample: ``image`` (``[C, size, size]`` uint8 ``MetaTensor``,
    with the pixel tier that served it in ``image.meta["tier"]``), the
    keys in :data:`lavlab.monai.datalist.SAMPLE_KEYS`, and for a *scales*
    run ``image_1``, ``image_2``, ... (coarser concentric crops) and
    ``scales``.

    The underlying ``monai.data.PatchWSIDataset`` is :attr:`patch_dataset`.

    :param image_ids: OMERO image IDs; not with *group* or *datalist*
    :type image_ids: int | Sequence[int] | None
    :param group: every image in this OMERO group
    :type group: int | None
    :param datalist: a prebuilt :func:`lavlab.monai.build_datalist` result,
        to skip re-analysing the slides (e.g. once per section or fold)
    :type datalist: Sequence[dict] | None
    :param classes: class names in label-index order (see
        :func:`lavlab.monai.datalist.default_classes`)
    :type classes: Sequence[str] | None
    :param section: ``training``, ``validation`` or ``test``
    :type section: str
    :param transform: transforms applied after the patch is read
    :param seed: seed for the case-level split
    :type seed: int
    :param val_frac: fraction of cases in ``validation``
    :type val_frac: float
    :param test_frac: fraction of cases in ``test``
    :type test_frac: float
    :param cache_rate: fraction of patches cached in memory; 0 (the
        default) reads every patch live, every epoch
    :type cache_rate: float
    :param creds: OMERO credentials (default: ``OMERO_*`` environment)
    :type creds: lavlab.config.OmeroCreds | None
    :param conn: a gateway for building the datalist only; patch reads
        always use per-process sessions
    :param conn_factory: passed to :class:`OmeroWSIReader`
    :param force_local: ``lavlab tile --force-local``
    :type force_local: bool
    :param store_count: ``lavlab tile --connections``
    :type store_count: int | None
    :param skip_failed: skip slides that cannot be tiled
    :type skip_failed: bool
    """

    def __init__(
        self,
        image_ids: int | Sequence[int] | None = None,
        *,
        group: int | None = None,
        datalist: Sequence[dict] | None = None,
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
        tile_seed: int = 0,
        section: str = "training",
        transform: Sequence[Callable] | Callable = (),
        seed: int = 0,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        cache_num: int = sys.maxsize,
        cache_rate: float = 0.0,
        num_workers: int | None = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        runtime_cache: bool = False,
        creds=None,
        conn=None,
        conn_factory=None,
        force_local: bool = False,
        store_count: int | None = None,
        skip_failed: bool = False,
    ) -> None:
        if datalist is not None:
            if image_ids is not None or group is not None:
                raise ValueError("pass datalist, or image_ids/group -- not both.")
            items = [dict(item) for item in datalist]
            sizes = {int(item["tile_size"]) for item in items}
            if len(sizes) > 1:
                raise ValueError(f"datalist mixes tile sizes {sorted(sizes)}.")
            out_size = sizes.pop() if sizes else size
            class_names = (
                list(classes) if classes is not None else _classes_from_items(items)
            )
        else:
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
                seed=tile_seed,
            )
            items, class_names = datalist_for_params(
                params,
                image_ids,
                group=group,
                conn=conn,
                creds=creds,
                classes=classes,
                force_local=force_local,
                skip_failed=skip_failed,
            )
            out_size = params.size
        self.class_names = class_names

        reader = OmeroWSIReader(
            creds=creds,
            conn_factory=conn_factory,
            force_local=force_local,
            store_count=store_count,
            resize_to=out_size,
        )
        self.patch_dataset = _OmeroPatchSource([], reader)
        self._init_split(
            items,
            section=section,
            transform=_chain(_ReadOmeroPatch(self.patch_dataset), transform),
            seed=seed,
            val_frac=val_frac,
            test_frac=test_frac,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            runtime_cache=runtime_cache,
        )
        self.patch_dataset.data = self.data


def _classes_from_items(items: Sequence[dict]) -> list[str]:
    """Recover class names, by index, from samples that carry both."""
    names: dict[int, str] = {}
    for item in items:
        index, name = int(item["label"]), str(item["label_name"])
        if names.setdefault(index, name) != name:
            raise ValueError(
                f"label {index} is both '{names[index]}' and '{name}' in the datalist."
            )
    if names and sorted(names) != list(range(len(names))):
        raise ValueError(
            "datalist does not use every label index (a class with no tiles?); "
            "pass classes= to name them."
        )
    return [names[i] for i in sorted(names)]


# ---------------------------------------------------------------------------
# B: tiles already extracted by `lavlab tile`
# ---------------------------------------------------------------------------


def _int_or(text: str, missing: int) -> int:
    return missing if text in ("", None) else int(text)


def _float_or_nan(text: str) -> float:
    return math.nan if text in ("", None) else float(text)


def _row_sample(row: dict, path: str) -> dict:
    return {
        "image": path,
        "label_name": row["label"],
        "omero_image_id": int(row["omero_image_id"]),
        "case_id": row["case_id"],
        "slide_id": row["slide_id"],
        "roi_id": _int_or(row["roi_id"], -1),
        "shape_id": _int_or(row["shape_id"], -1),
        "mpp": _float_or_nan(row["mpp"]),
        "tile_size": int(row["size"]),
        "x": int(row["x"]),
        "y": int(row["y"]),
        "x0": int(row["x0"]),
        "y0": int(row["y0"]),
        "w0": int(row["w0"]),
        "h0": int(row["h0"]),
        "level": int(row["level"]),
        "coverage": float(row["coverage"]),
        "tissue_frac": float(row["tissue_frac"]),
        "tier": row["tier"],
    }


def _has_scale(mpp: float, scales: Sequence[float] | None) -> bool:
    if scales is None:
        return True
    return any(math.isclose(mpp, s, rel_tol=1e-6) for s in scales)


def read_manifest_samples(
    manifest: str,
    *,
    root: str | None = None,
    classes: Sequence[str] | None = None,
    scales: Sequence[float] | None = None,
) -> tuple[list[dict], list[str]]:
    """Read a ``lavlab tile`` manifest into lavlab samples.

    A ``--scales`` run writes one row per crop; rows of one tile location
    (recognised by their ``_x<x>_y<y>_mpp<mpp>`` file name) become one
    sample, finest crop as ``image``, the others as ``image_1``, ... A
    combined ``--manifest`` may hold several runs; *scales* picks which
    crops to use, and is required when the runs disagree.

    :param manifest: a per-slide ``manifest.csv`` or a ``--manifest`` CSV
    :type manifest: str
    :param root: directory ``tile_path`` is relative to; default the CSV's
        own folder, which is right for a ``--manifest`` CSV. A per-slide
        ``manifest.csv`` is relative to the ``-o`` directory instead.
    :type root: str | None
    :param classes: class names in label-index order; rows of any other
        label are left out. Default: every label, sorted.
    :type classes: Sequence[str] | None
    :param scales: keep only crops at these mpps
    :type scales: Sequence[float] | None
    :return: ``(samples, class names)``
    :rtype: tuple[list[dict], list[str]]
    :raises ValueError: for a file that is not a lavlab manifest, or one
        mixing runs that *scales* does not disambiguate
    :raises FileNotFoundError: if tile paths don't resolve against *root*
    """
    manifest = os.path.abspath(manifest)
    base = os.path.abspath(root) if root is not None else os.path.dirname(manifest)
    with open(manifest, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"'{manifest}' is not a lavlab tile manifest (missing {missing})."
            )
        rows = list(reader)

    coords_only = sum(1 for row in rows if not row["tile_path"])
    if coords_only:
        log.warning(
            "%s: %d row(s) have no tile image (--coords-only); leaving them out.",
            manifest,
            coords_only,
        )
    rows = [row for row in rows if row["tile_path"]]
    if classes is not None:
        wanted = set(classes)
        dropped = sum(1 for row in rows if row["label"] not in wanted)
        if dropped:
            log.info("%s: %d row(s) of other classes left out.", manifest, dropped)
        rows = [row for row in rows if row["label"] in wanted]
    class_names = (
        list(classes) if classes is not None else sorted({r["label"] for r in rows})
    )
    if rows:
        first = os.path.join(base, rows[0]["tile_path"])
        if not os.path.isfile(first):
            raise FileNotFoundError(
                f"tile '{rows[0]['tile_path']}' from '{manifest}' is not under "
                f"'{base}'. tile_path is relative to a --manifest CSV's folder, but "
                "to the -o directory for a per-slide manifest.csv; pass root=."
            )

    single: list[dict] = []
    multi: dict[tuple, dict[float, dict]] = {}
    duplicates = 0
    for row in rows:
        sample = _row_sample(
            row, os.path.normpath(os.path.join(base, row["tile_path"]))
        )
        sample["label"] = class_names.index(row["label"])
        name = _SCALED_NAME.search(row["tile_path"])
        if name is None or (int(name.group(1)), int(name.group(2))) != (
            sample["x"],
            sample["y"],
        ):
            if _has_scale(sample["mpp"], scales):
                single.append(sample)
            continue
        if not _has_scale(sample["mpp"], scales):
            continue
        key = (
            sample["omero_image_id"],
            sample["x"],
            sample["y"],
            sample["label_name"],
            sample["tile_size"],
        )
        crops = multi.setdefault(key, {})
        if sample["mpp"] in crops:
            duplicates += 1
            continue
        crops[sample["mpp"]] = sample
    if duplicates:
        log.warning(
            "%s: %d row(s) repeat a tile location and scale; using the first.",
            manifest,
            duplicates,
        )

    if multi and single:
        if scales is None:
            raise ValueError(
                f"'{manifest}' mixes single-scale and --scales runs; pass scales= "
                "to choose which crops make a sample."
            )
        if len(scales) > 1:
            log.warning(
                "%s: %d single-scale row(s) left out; they have no context crops.",
                manifest,
                len(single),
            )
            single = []
    mpp_sets = {tuple(sorted(crops)) for crops in multi.values()}
    if scales is None and len(mpp_sets) > 1:
        raise ValueError(
            f"'{manifest}' holds --scales runs with different scales "
            f"{sorted(mpp_sets)}; pass scales= to choose."
        )

    samples = list(single)
    incomplete = 0
    for crops in multi.values():
        mpps = sorted(crops)
        if scales is not None and len(mpps) != len(scales):
            incomplete += 1
            continue
        if len(mpps) == 1:
            samples.append(crops[mpps[0]])
            continue
        sample = dict(crops[mpps[0]])
        for index, mpp in enumerate(mpps[1:], 1):
            sample[f"image_{index}"] = crops[mpp]["image"]
        sample["scales"] = mpps
        samples.append(sample)
    if incomplete:
        log.warning(
            "%s: %d tile location(s) lack a crop at one of scales %s; left out.",
            manifest,
            incomplete,
            list(scales or []),
        )
    return samples, class_names


class LavlabTileDataset(_CaseSplitDataset):
    """Tiles already extracted by ``lavlab tile``, read from disk.

    The path to train on: ``lavlab tile ... --manifest dataset.csv`` once,
    then this dataset over ``dataset.csv`` for every epoch. Paths in the
    manifest are resolved against its own folder, not the working
    directory, so the CSV and its tiles can move together.

    Each sample matches :class:`OmeroPatchDataset`'s: ``image`` (a
    ``[C, H, W]`` uint8 ``MetaTensor``), ``image_1``, ... and ``scales``
    for a ``--scales`` extraction, and the keys in
    :data:`lavlab.monai.datalist.SAMPLE_KEYS`.

    :param manifest: a ``--manifest`` CSV or a per-slide ``manifest.csv``
    :type manifest: str
    :param section: ``training``, ``validation`` or ``test``
    :type section: str
    :param transform: transforms applied after the tile is loaded
    :param classes: class names in label-index order; other labels are left
        out (default: every label in the manifest, sorted)
    :type classes: Sequence[str] | None
    :param scales: which ``--scales`` crops to use (see
        :func:`read_manifest_samples`)
    :type scales: Sequence[float] | None
    :param root: directory ``tile_path`` is relative to (default: the
        manifest's folder)
    :type root: str | None
    :param seed: seed for the case-level split
    :type seed: int
    :param val_frac: fraction of cases in ``validation``
    :type val_frac: float
    :param test_frac: fraction of cases in ``test``
    :type test_frac: float
    :param cache_rate: fraction of tiles cached in memory (default 0)
    :type cache_rate: float
    """

    def __init__(
        self,
        manifest: str,
        section: str = "training",
        transform: Sequence[Callable] | Callable = (),
        *,
        classes: Sequence[str] | None = None,
        scales: Sequence[float] | None = None,
        root: str | None = None,
        seed: int = 0,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        cache_num: int = sys.maxsize,
        cache_rate: float = 0.0,
        num_workers: int | None = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        runtime_cache: bool = False,
    ) -> None:
        items, self.class_names = read_manifest_samples(
            manifest, root=root, classes=classes, scales=scales
        )
        extra = max(
            (len(item.get("scales", ())) - 1 for item in items),
            default=0,
        )
        keys = ["image", *(f"image_{k}" for k in range(1, extra + 1))]
        loader = LoadImaged(
            keys=keys,
            # (H, W, C) as stored, like the WSI reader's patches; MONAI's
            # default reverse_indexing would transpose them to (W, H).
            reader="PILReader",
            reverse_indexing=False,
            image_only=True,
            ensure_channel_first=True,
            dtype=np.uint8,
        )
        self._init_split(
            items,
            section=section,
            transform=_chain(loader, transform),
            seed=seed,
            val_frac=val_frac,
            test_frac=test_frac,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            runtime_cache=runtime_cache,
        )
