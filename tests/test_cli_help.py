"""Help layout shared by every ``lavlab`` command (see commands/_shared.py)."""

import argparse
import re

import pytest

from lavlab.cli import build_parser
from lavlab.commands._shared import SECTION_RULE

LEAF_COMMANDS = [
    ["lr"],
    ["roi"],
    ["tile"],
    ["meta", "roi", "textvalue"],
    ["geojson", "import"],
    ["geojson", "export"],
    ["seg", "dcm2nii"],
    ["seg", "nii2dcm"],
]
GROUP_COMMANDS = [[], ["meta"], ["meta", "roi"], ["geojson"], ["seg"]]
OMERO_COMMANDS = [cmd for cmd in LEAF_COMMANDS if cmd[0] != "seg"]


def _parser_for(path):
    parser = build_parser()
    for name in path:
        subparsers = next(
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        )
        parser = subparsers.choices[name]
    return parser


def _help(path):
    return _parser_for(path).format_help()


@pytest.mark.parametrize("path", LEAF_COMMANDS + GROUP_COMMANDS[1:], ids=" ".join)
def test_usage_names_only_the_command_itself(path):
    first_line = _help(path).splitlines()[0]

    assert first_line.startswith(f"Usage: {' '.join(['lavlab', *path])} ")
    # argparse's default would splice the parent's whole usage line in.
    assert "[--override]" not in first_line


@pytest.mark.parametrize("path", LEAF_COMMANDS, ids=" ".join)
def test_options_come_before_ruled_sections(path):
    text = _help(path)

    assert "\nOptions:\n" in text
    assert "positional arguments:" not in text
    assert "\noptions:\n" not in text
    if SECTION_RULE in text:
        assert text.index("\nOptions:\n") < text.index(SECTION_RULE)


@pytest.mark.parametrize("path", OMERO_COMMANDS, ids=" ".join)
def test_connection_flags_have_their_own_section(path):
    text = _help(path)
    connection = text.index(f"{SECTION_RULE}\n\nOMERO Connection:")
    section = text[connection:].split(SECTION_RULE)[1]

    for flag in ("--user", "--password", "--host", "--port"):
        assert re.search(rf"^\s+-\w [A-Z]+, {flag}\b", section, re.MULTILINE)
        assert flag not in text[:connection]


@pytest.mark.parametrize("path", OMERO_COMMANDS, ids=" ".join)
def test_advanced_options_come_after_connection(path):
    text = _help(path)

    assert text.index("OMERO Connection:") < text.index("Advanced Options")


def test_seg_never_mentions_omero_connection():
    for path in (["seg", "dcm2nii"], ["seg", "nii2dcm"]):
        assert "OMERO Connection" not in _help(path)


@pytest.mark.parametrize("path", GROUP_COMMANDS, ids=lambda p: " ".join(p) or "root")
def test_command_lists_have_no_metavar_header(path):
    text = _help(path)
    commands = text.split("\nCommands:\n")[1]

    assert not commands.lstrip().startswith("COMMAND")


def test_advanced_flags_are_grouped_away_from_everyday_ones():
    text = _help(["lr"])
    options, _, rest = text.partition(SECTION_RULE)

    def listed(flag, section):
        # Where the flag is defined, not where another flag's help mentions it.
        return re.search(rf"^\s+(-\w [A-Z]+, )?{flag}\b", section, re.MULTILINE)

    for everyday in ("--output", "--group", "--downsample", "--format"):
        assert listed(everyday, options)
    for advanced in ("--regenerate", "--skip-upload", "--skip-local", "--fs-map"):
        assert not listed(advanced, options)
        assert listed(advanced, rest)


@pytest.mark.parametrize(
    "argv",
    [
        ["lr", "12345", "--regenerate", "--fs-map", "map.yaml", "-u", "me"],
        ["roi", "batch", "-t", "G3", "--upload", "--suffix", "_x"],
        ["meta", "roi", "textvalue", "default", "1", "2", "--tolerance", "5"],
        ["geojson", "import", "a.geojson", "--image", "1", "--fill-alpha", "0"],
        ["geojson", "export", "--project", "2", "--out", "d", "--keep-bridges"],
        ["seg", "nii2dcm", "m.nii.gz", "dcm/", "-o", "o.dcm", "--label", "L"],
        ["tile", "1", "-o", "d", "--roi", "--all", "--erode", "auto", "--seed", "3"],
    ],
    ids=lambda argv: " ".join(argv[:2]),
)
def test_moving_flags_between_sections_kept_their_meaning(argv):
    """Every flag parses into the same attribute it always did."""
    args = build_parser().parse_args(argv)

    assert callable(args.handler)
    expected = {
        "lr": {"target": "12345", "regenerate": True, "fs_map": "map.yaml"},
        "roi": {"text_filter": ["g3"], "upload": True, "suffix": "_x"},
        "meta": {"text_mapping": "default", "image_ids": [1, 2], "tolerance": 5},
        "geojson": {"paths": ["a.geojson"], "image": 1, "fill_alpha": 0}
        if argv[1] == "import"
        else {"project": 2, "out": "d", "keep_bridges": True},
        "seg": {"segment_label": "L", "output_path": "o.dcm"},
        "tile": {"erode": "auto", "seed": 3, "roi": True},
    }[argv[0]]
    for name, value in expected.items():
        assert getattr(args, name) == value
