# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Output filename/path resolution.

Naming convention: ``LR${DOWNSAMPLE}_${FILENAME}(_${SUFFIX}).${EXT}``, where
``FILENAME`` is the image name split on '.' and truncated to the first
segment (so ``foo.ome.tiff`` -> ``foo``).
"""

from __future__ import annotations

import glob
import logging
import os

from lavlab.config import ConfigError, FsMapGroup

log = logging.getLogger(__name__)


def get_stem(filename: str) -> str:
    """Return the filename with all extensions stripped (splits on the first '.')."""
    return filename.split(".")[0]


def build_filename(
    downsample: int, filename: str, suffix: str | None = None, ext: str = "jp2"
) -> str:
    stem = get_stem(filename)
    name = f"LR{downsample}_{stem}"
    if suffix:
        name += f"_{suffix}"
    return f"{name}.{ext}"


def resolve_fs_map_dir(
    fs_map: dict[str, FsMapGroup], group_id, filename: str
) -> str | None:
    """Return the directory an image should be written to per fs_map.

    Returns None if there's no fs_map entry for this group/filename. Raises
    ConfigError if a matching entry's base_dir isn't present on disk (i.e.
    the storage isn't mounted) -- this is the only fs_map failure that should
    abort the whole run.
    """
    group = fs_map.get(str(group_id))
    if group is None:
        return None

    for entry in group.maps:
        match = entry.match.match(filename)
        if not match:
            continue
        if not os.path.isdir(entry.base_dir):
            raise ConfigError(
                f"fs_map base_dir '{entry.base_dir}' for group {group_id} "
                "does not exist -- is the storage mounted?"
            )

        base_dir = entry.base_dir
        if entry.subject_glob:
            value = match.group(entry.subject_glob)
            candidates = sorted(glob.glob(os.path.join(entry.base_dir, f"*{value}")))
            if not candidates:
                log.warning(
                    "fs_map: no directory under '%s' matching '*%s' for '%s'; "
                    "trying the next entry.",
                    entry.base_dir,
                    value,
                    filename,
                )
                continue
            base_dir = candidates[0]

        formatted = entry.formatted_dir
        for name, value in match.groupdict().items():
            formatted = formatted.replace(f"${{{name}}}", value or "")
        return os.path.join(base_dir, formatted)

    return None


def resolve_output_path(
    output_arg: str | None,
    fs_map: dict[str, FsMapGroup] | None,
    group_id,
    filename: str,
    downsample: int,
    suffix: str | None = None,
    ext: str = "jp2",
    batch: bool = False,
) -> str | None:
    """Resolve the final output file path for one image.

    - If ``output_arg`` is given, use it directly (joined with the generated
      filename when it's a directory, or always in batch mode where -o is a
      directory by definition).
    - Otherwise resolve via fs_map. A missing regex match or a specific
      formatted directory that doesn't exist on disk is only ever a warning:
      single-image mode falls back to the current directory, batch mode
      returns None so the caller can skip that one image and continue.
    - An fs_map base_dir that doesn't exist raises ConfigError (propagated),
      since that means the whole storage target is unavailable.
    """
    name = build_filename(downsample, filename, suffix, ext)

    if output_arg:
        if batch or os.path.isdir(output_arg):
            return os.path.join(output_arg, name)
        return output_arg

    directory = None
    if fs_map:
        directory = resolve_fs_map_dir(fs_map, group_id, filename)

    if directory is None or not os.path.isdir(directory):
        if directory is not None:
            log.warning(
                "fs_map resolved directory '%s' does not exist; %s",
                directory,
                "skipping image" if batch else "writing to current directory instead",
            )
        return None if batch else os.path.join(os.getcwd(), name)

    return os.path.join(directory, name)
