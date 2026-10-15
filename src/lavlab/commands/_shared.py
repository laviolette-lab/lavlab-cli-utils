# SPDX-FileCopyrightText: 2026-present LavLab <domurphy@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Shared helpers for the command modules: help layout, OMERO credential
flags, and the connection/output plumbing ``lr`` and ``roi`` have in common."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile

from lavlab.config import ConfigError, load_fs_map, resolve_creds

# lavlab.omero_client pulls in omero-py (and through it Ice). It is imported
# inside the functions that need a connection so that building the argument
# parser -- and therefore --help -- stays pure Python.

log = logging.getLogger(__name__)


def make_temp_path(fmt: str) -> str:
    """Create an empty temp file with a *fmt* extension and return its path.

    Used by ``--skip-local``: OMERO's upload API reads from a real path
    rather than raw bytes, so "don't keep a local copy" still needs a file
    to exist for the duration of the upload. The caller is responsible for
    deleting it afterward.

    :param fmt: file extension, without the dot
    :type fmt: str
    :return: path to a new empty file
    :rtype: str
    """
    fd, path = tempfile.mkstemp(suffix=f".{fmt}")
    os.close(fd)
    return path


SECTION_RULE = "=" * 22

#: Shared description for TARGET on the commands that take an image or 'batch'.
BATCH_TARGET_HELP = (
    "An OMERO image ID, or the word 'batch' to process every image you can "
    "see (only one group's with -g)."
)


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Help layout shared by every lavlab command.

    Everyday flags come first under "Options"; any further section
    (OMERO connection, advanced options) is set off by a rule so it reads
    as something most runs can skip. Descriptions and epilogs keep their
    line breaks, so they can carry short lists and examples.
    """

    _UNRULED_SECTIONS = frozenset({"Arguments", "Options", "Commands"})

    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(
            usage, actions, groups, "Usage: " if prefix is None else prefix
        )

    def start_section(self, heading):
        if heading not in self._UNRULED_SECTIONS:
            self.add_text(SECTION_RULE)
        super().start_section(heading)

    def _format_action(self, action):
        # List subcommands directly under "Commands", without the extra
        # "COMMAND" header line argparse puts above them.
        if isinstance(action, argparse._SubParsersAction):
            return "".join(self._format_action(sub) for sub in action._get_subactions())
        return super()._format_action(action)


def style_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Rename argparse's default sections to match :class:`HelpFormatter`."""
    parser._positionals.title = "Arguments"
    parser._optionals.title = "Options"
    return parser


def add_command(subparsers, name: str, **kwargs) -> argparse.ArgumentParser:
    """Add a subcommand parser that uses the shared help layout."""
    return style_parser(
        subparsers.add_parser(name, formatter_class=HelpFormatter, **kwargs)
    )


def add_subcommands(parser: argparse.ArgumentParser, dest: str):
    """Add a required, "Commands"-titled subcommand list to *parser*.

    ``prog`` is passed explicitly because argparse otherwise builds each
    subcommand's name from the parent's whole usage line.
    """
    return parser.add_subparsers(
        dest=dest, required=True, title="Commands", metavar="COMMAND", prog=parser.prog
    )


def add_advanced_group(parser: argparse.ArgumentParser, topic: str | None = None):
    title = "Advanced Options" + (f" ({topic})" if topic else "")
    return parser.add_argument_group(title)


def add_creds_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "OMERO Connection",
        "Each flag falls back to an environment variable, so you can set them "
        "once in your shell\n(e.g. export OMERO_USER=you) instead of typing "
        "them on every run.",
    )
    group.add_argument("-u", "--user", help="OMERO username (or set OMERO_USER).")
    group.add_argument(
        "-w", "--password", help="OMERO password (or set OMERO_PASSWORD)."
    )
    group.add_argument("-s", "--host", help="OMERO server address (or set OMERO_HOST).")
    group.add_argument(
        "-p", "--port", type=int, help="OMERO server port (or set OMERO_PORT)."
    )


def add_downsample_arg(group) -> None:
    group.add_argument(
        "--downsample",
        type=int,
        default=10,
        metavar="N",
        help="Shrink factor: 10 makes the output 1/10 of the full-resolution "
        "width and height (default: 10).",
    )


def add_fs_map_arg(group) -> None:
    group.add_argument(
        "--fs-map",
        dest="fs_map",
        metavar="PATH",
        help="YAML file mapping OMERO groups to output folders, used when -o is "
        "not given (default: the lab's bundled map).",
    )


def connect_from_args(args: argparse.Namespace):
    from lavlab.omero_client import connect

    try:
        creds = resolve_creds(args.user, args.password, args.host, args.port)
    except ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None
    return connect(creds)


def load_fs_map_from_args(args: argparse.Namespace):
    try:
        return load_fs_map(args.fs_map)
    except ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None


def parse_target(target: str) -> int | str:
    """Return the image id as an int, or 'batch' verbatim."""
    if target == "batch":
        return "batch"
    try:
        return int(target)
    except ValueError:
        raise SystemExit(
            f"error: expected an image ID or 'batch', got '{target}'"
        ) from None


def group_of(conn, obj) -> int:
    from lavlab.omero_client import switch_to_object_group

    switch_to_object_group(conn, obj)
    return obj.details.group.id.val


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
