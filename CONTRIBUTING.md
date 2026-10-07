# Contributing to lavlab-cli-utils


## Setting up a dev environment

```sh
git clone <repo>
cd omero-cli-utils
pip install -e ".[dev]"
```

`omero-py` needs the Glencoe/ZeroC Ice wheel for your platform installed
*before* this, from the lab's approved internal URL -- it isn't on PyPI and
`pip install -e ".[dev]"` can't fetch it for you. See
`build-requirements.txt`'s comment, or use the `Dockerfile` (see the
README's Docker section) for a pre-built environment that already has it.

Run from source with `python -m lavlab ...` -- the compiled binary
(`lavlab ...`, no `python -m`) only exists after a Nuitka build (see the
README's "Building the compiled wheel" section), so don't expect the plain
`lavlab` command to work in a dev checkout unless you've built it.

## Running the tests

```sh
pytest
```

`tests/test_geometry.py` and `tests/test_geojson_io.py` have no
dependencies beyond the standard library and always run.
`tests/test_seg.py` uses `pytest.importorskip` for `nibabel`/`pydicom` -- it's
skipped, not failed, if the imaging stack isn't installed, so you can still
run the suite without the full `dev` extra if you're only touching GeoJSON code.

If you touch `src/lavlab/seg.py`, run the tests with the real imaging stack
installed and pay attention to
`test_nifti_to_dcmseg_round_trips_through_dcmseg_to_nifti` specifically --
this module has already had real bugs that only a real round-trip test
caught, not unit tests of individual functions. If you change anything in
the DICOM SEG read/write path, re-verify the round trip still holds, don't
just check that the function you touched doesn't raise.

## The command-group pattern

Every CLI command group follows the same shape. If you're adding a new
one, copy `src/lavlab/commands/roi_cmd.py` or `src/lavlab/commands/geojson.py` as
a template rather than inventing a new structure:

1. **A logic module with no argparse in it** (`src/lavlab/roi.py`,
   `src/lavlab/geojson/geojson_io.py`, `src/lavlab/seg.py`, ...) -- plain functions
   that take plain arguments and either return a result or raise a
   specific exception. This is what makes the logic testable and usable
   from a notebook without going through the CLI at all.
2. **A command module** (`src/lavlab/commands/<name>.py`) with:
   - `add_parser(subparsers) -> None` that registers the subcommand(s) and
     calls `parser.set_defaults(handler=run)` (or `run_<subaction>` for
     multiple nested subcommands, like `geojson.py`'s `import`/`export`).
   - A `run(args)` function that pulls values off `args`, calls into the
     logic module, and prints results. `run` should not itself contain
     conversion/business logic -- that belongs in step 1's module.
   - Errors the user can plausibly cause (bad input, missing file, OMERO
     object not found) become `raise SystemExit("error: ...")` right in
     `run`/its helpers, not an uncaught exception. Let genuinely
     unexpected exceptions propagate uncaught -- don't swallow errors you
     don't understand.
3. **Register it** in `src/lavlab/cli.py`: import the command module, call its
   `add_parser(subparsers)` in `build_parser()`.
4. If the command talks to OMERO, use `src/lavlab/commands/_shared.py`'s
   `add_creds_args`/`connect_from_args` -- don't invent another
   `--server`/`--user`/`--password` flag set. Every OMERO-facing command
   group already shares this; don't reintroduce a split.

## Conventions to follow

- **Lazy `omero` imports.** Only modules that actually need `omero`
  import it, and only inside the functions that use it (see
  `src/lavlab/geojson/omero_io.py`, `src/lavlab/omero_client.py`). This is what
  lets `--dry-run` modes and most of the test suite run on a machine with
  no OMERO/Ice installed at all. Don't move an `import omero` to module
  level "for convenience" -- it breaks that property for the whole import
  chain above it.
- **Docstrings** are Sphinx-style (`:param:`, `:type:`, `:raises:`,
  `:return:`, `:rtype:`), matching the rest of the codebase. Keep them
  factual and about the *why* where it's non-obvious (see almost any
  docstring in `src/lavlab/geojson/geometry.py` for the bar to hit), not a
  restatement of the function signature.
- **Raise specific exceptions, not generic ones.** `FileNotFoundError` for
  a missing path, `ValueError` for bad/mismatched data, a custom exception
  (like `ConversionError`, `ConfigError`) where neither fits. Don't let a
  third-party library's internal traceback (a raw `pydicom` stack trace, an Ice
  connection error) be the only signal a user gets --
  catch it and re-raise with a message that says what to actually do.
- **No comments that restate the code.** A comment earns its place only
  when it explains a non-obvious constraint, a workaround for a specific
  library quirk, or a decision someone would otherwise "fix" by accident
  (see the comment above `PYDICOM_NUITKA_FLAGS` in `build_native.py` for
  an example -- it exists precisely so nobody strips those flags thinking
  they're dead weight).

## Adding a new bundled default (fs_map, seg template, ...)

Follow `src/lavlab/config.py`'s `_default_fs_map_path()` /
`src/lavlab/seg.py`'s `_default_seg_template_path()` pattern: put the file in
`src/lavlab/data/`, load it via `importlib.resources`, and add its extension
pattern to `pyproject.toml`'s `[tool.setuptools.package-data]` so it
actually ships in the wheel.

## Before you consider something done

- `pytest` passes.
- If you touched anything OMERO-facing, you've actually run it against a
  real (or at least a test) OMERO server -- argparse wiring correctness is
  not the same as the command working, and unit tests built on fakes are
  not a substitute either: real production data has surfaced genuine bugs
  in this codebase (a matplotlib-gated skimage function that crashed on
  any Rectangle ROI, a false-positive hole detector that silently dropped
  a whole ROI on export) that no amount of fake-based unit testing would
  have caught. Don't skip the live check.
- If you touched the Nuitka build (`setup.py`, `build_native.py`,
  `Dockerfile`), actually run a build and smoke-test the resulting binary
  (`lavlab --help` and the specific subcommand you touched), not just read
  the change and reason it should work. Compiled-binary failures are
  frequently invisible from source review alone -- see the pydicom
  example under the README's "Building the compiled wheel" section for
  what that class of failure looks like.
