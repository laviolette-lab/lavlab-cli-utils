# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Credential resolution and fs_map (filesystem mapping) configuration.

fs_map is an optional YAML file describing, per-OMERO-group, where output
images should be written on disk when the user doesn't pass an explicit
``-o`` path. See ``lavlab/data/default_fs_map.yaml`` for the format and the
lab's default mapping.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised for unrecoverable configuration problems.

    Examples: missing OMERO credentials, or an fs_map ``base_dir`` that isn't
    mounted/present on disk (as opposed to a single missing subdirectory,
    which is only ever a warning).
    """


@dataclass(frozen=True)
class OmeroCreds:
    user: str
    password: str
    host: str
    port: int


def _resolve_field(cli_value: str | None, env_name: str, field_name: str) -> str:
    if cli_value:
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    raise ConfigError(
        f"OMERO {field_name} was not provided: pass it on the command line "
        f"or set the {env_name} environment variable."
    )


def resolve_creds(
    user: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> OmeroCreds:
    """Resolve OMERO connection credentials.

    CLI-provided values always win. Otherwise falls back to the
    OMERO_USER / OMERO_PASSWORD / OMERO_HOST / OMERO_PORT environment
    variables. Raises ConfigError if a field is available from neither source.
    """
    resolved_user = _resolve_field(user, "OMERO_USER", "user")
    resolved_password = _resolve_field(password, "OMERO_PASSWORD", "password")
    resolved_host = _resolve_field(host, "OMERO_HOST", "host")
    resolved_port = _resolve_field(
        str(port) if port is not None else None, "OMERO_PORT", "port"
    )
    return OmeroCreds(
        resolved_user, resolved_password, resolved_host, int(resolved_port)
    )


@dataclass(frozen=True)
class FsMapEntry:
    """One fs_map regex -> directory rule.

    ``subject_glob``, if set, names a regex group whose captured value is
    matched as a *suffix* against directories directly under ``base_dir``
    (``glob(base_dir + "/*" + value)``); the matched directory replaces
    ``base_dir`` for this resolution. The lab's on-disk subject directories
    don't consistently share the OMERO image name's prefix (subject "101"
    lives in a directory named "1101"), so a literal ``${subject}``
    substitution can't find them.
    """

    match: re.Pattern[str]
    base_dir: str
    formatted_dir: str
    subject_glob: str | None = None


@dataclass(frozen=True)
class FsMapGroup:
    group_id: str
    name: str
    maps: list[FsMapEntry] = field(default_factory=list)


def _default_fs_map_path() -> Path:
    return Path(resources.files("lavlab.data").joinpath("default_fs_map.yaml"))


def load_fs_map(path: str | None) -> dict[str, FsMapGroup]:
    """Load an fs_map YAML file, falling back to the bundled lab default.

    Returns a mapping of group ID (as a string) -> FsMapGroup. Returns an
    empty mapping if no fs_map is available at all.
    """
    map_path: Path
    if path is not None:
        map_path = Path(path)
        if not map_path.is_file():
            raise ConfigError(f"fs_map file '{path}' does not exist.")
    else:
        map_path = _default_fs_map_path()
        if not map_path.is_file():
            return {}

    with open(map_path) as f:
        raw = yaml.safe_load(f) or {}

    groups: dict[str, FsMapGroup] = {}
    for group_id, group_data in raw.items():
        maps = [
            FsMapEntry(
                match=re.compile(m["match"]),
                base_dir=m["base_dir"],
                formatted_dir=m["formatted_dir"],
                subject_glob=m.get("subject_glob"),
            )
            for m in group_data.get("maps", [])
        ]
        groups[str(group_id)] = FsMapGroup(
            group_id=str(group_id), name=group_data.get("name", ""), maps=maps
        )
    return groups
