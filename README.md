# lavlab-cli-utils

LavLab's CLI toolbox for OMERO: pulling large-recon and ROI mask images,
cutting whole-slide images into training tiles, moving QuPath GeoJSON
annotations in and out of OMERO, filling in ROI metadata, and converting
between DICOM SEG and NIfTI segmentation masks.

More documentation lives in [`docs/`](docs/index.md):
[`docs/reference.md`](docs/reference.md) is the exhaustive flag-by-flag CLI reference
plus the importable Python API; [`CONTRIBUTING.md`](CONTRIBUTING.md) covers
development conventions.

## Architecture, in brief

One Python package, `lavlab`, with six subcommands
(`lr`/`roi`/`tile`/`meta`/`geojson`/`seg`) sharing common OMERO-connection
and argparse plumbing:

```text
lavlab/
├── cli.py                 top-level argparse entry point
├── commands/               one module per subcommand -- argparse + orchestration only
│   └── _shared.py            shared creds/output-path/connection helpers
├── geojson/                 GeoJSON<->OMERO conversion logic (no argparse in here)
├── seg.py                   DICOM SEG<->NIfTI conversion logic
├── tiling.py                whole-slide tiling logic (no argparse in here)
├── omero_client.py, config.py, naming.py, imaging.py, roi.py, palettes.py
└── data/                    bundled default config, loaded via importlib.resources
```

The package is built to run two different ways, and it matters which one
you're using:

- **Compiled binary** (what us lab members are designed to do): the
  `lavlab` console-script just `os.execv`s a self-contained native
  executable (`lavlab/launcher.py` -> `lavlab/bin/dist/lavlab-bin`),
  compiled ahead of time with [Nuitka](https://nuitka.net). The small
  `lavlab` launcher only starts that bundled executable; it never falls back
  to running the Python CLI. No `omero-py` or Ice bindings are needed on the
  machine running it.
- **Source checkout** (development): `python -m lavlab ...` runs the real
  Python CLI directly (`src/lavlab/__main__.py` -> `src/lavlab/cli.py`). This is
  what you use while developing, and it needs the full dependency stack
  installed (see below) since nothing is precompiled.

## Install

**From a built wheel** (recommended for end users, i.e. lab members who
just want the `lavlab` command):

```sh
pip install lavlab_cli_utils-<version>-<platform>.whl
lavlab --help
```

The wheel at the moment can be found in dist and the file is there. Or
in the actions menu there are builds there that are automatically made
through github.

**From source, for development only** (this intentionally runs Python code
instead of the bundled executable):

```sh
pip install -e ".[dev]"
python -m lavlab --help
```

The `dev` extra (and the equivalent `requirements.txt`) installs everything
`lavlab` needs to run: `numpy`, `pyvips`, `tifffile`, `scikit-image`,
`PyYAML`, `tqdm`, `omero-py`, `highdicom`, `nibabel`, `SimpleITK`,
`pydicom`, `pytest`. **`omero-py` additionally needs the Glencoe/ZeroC Ice
wheel for your platform, installed separately first** -- it isn't on PyPI.
See `build-requirements.txt`'s comment for where to get it, or use the
[Docker build environment](#building-with-docker) below, which handles
this the same way.

## Connecting to OMERO

Every command that talks to OMERO (`lr`, `roi`, `meta`, `geojson` -- not
`seg`, which never touches OMERO) takes the same four flags, or reads the
same four environment variables:

| Flag | Env var | Meaning |
|---|---|---|
| `-u` / `--user` | `OMERO_USER` | your OMERO username |
| `-w` / `--password` | `OMERO_PASSWORD` | your OMERO password |
| `-s` / `--host` | `OMERO_HOST` | server hostname |
| `-p` / `--port` | `OMERO_PORT` | server port |

A CLI flag always wins over the matching environment variable. A missing
value from both is a hard error (`error: OMERO user was not provided: ...`)
-- there's no interactive password prompt.

**The OMERO group model:** object lookups happen in OMERO's "dummy group"
(`-1`), which makes objects visible across every group you belong to;
once a command is about to actually operate on a specific object, it
switches the connection into that object's real group. You don't need to
think about this day to day -- it's just why `lavlab lr 12345` works
regardless of which group image 12345 lives in, and why batch commands
accept `-g/--group` to scope a run to one group explicitly.

## Commands

The sections below give the shape of each command with the most common
flags; **[`docs/reference.md`](docs/reference.md) has every flag for every command**,
exhaustively.

### `lavlab lr` -- pull large-recon (downsampled) images

```sh
lavlab lr 12345 -o ./out/slide.jp2 --downsample 10 -s omero.example.edu -u you
lavlab lr batch -g 3 --workers 8 -s omero.example.edu -u you
```

"LR" is lab shorthand for a downsampled export of a whole-slide image.
Output filenames follow `LR${downsample}_${stem}.${format}`, where `stem`
is the image name with everything after the *first* `.` stripped (so
`foo.ome.tiff` -> `foo` -- this matters for OME-TIFFs specifically) and
`format` defaults to `jp2` (`--format jpg`/`jpeg`/`png`/`tif`/`tiff` picks
a different one).

**Three tiers, tried in order, so `lr` works from any workstation -- not
just the cluster:**

1. **Cached OMERO annotation.** If a `LargeRecon.${downsample}` file
   annotation already exists on the image in the requested format,
   download it directly. Fast, works from anywhere.
2. **Locally-mounted source file.** If OMERO's managed repository happens
   to be mounted where `lr` is running (true on the cluster, not on a
   workstation), generate the recon straight from the source file.
3. **OMERO's tile API, over the network.** Otherwise, fetch tiles directly
   from the server and assemble the image locally. Slower (minutes, for a
   large slide) but works from any machine with just OMERO credentials --
   this is what makes `lr` usable off-cluster at all.

A tier that cannot deliver falls forward to the next one rather than
failing the image: an annotation that won't download, or a source file
that won't decode (unreadable, truncated, or in a compression scheme the
build has no codec for), drops through to the tier below. **Falling back
from tier 2 to tier 3 is logged at `WARNING`** with the reason, because
the run still "succeeds" while doing the slow thing for every image --
if you see one of those per image, the mounted repository is not actually
usable and the run is taking minutes per slide for nothing. Errors that
indicate a bug in `lr` itself are deliberately *not* caught, so they still
fail loudly instead of hiding behind a slow retry.

Whatever tier 2 or 3 produces gets uploaded back to OMERO under the same
`LargeRecon.${downsample}` namespace, as `LR${downsample}_${image
name}.${format}` -- independent of wherever `-o` points locally -- so a
later run at the same downsample and format hits tier 1 instead of
regenerating. Only an annotation matching the requested format is ever
touched or replaced; an existing attachment in some other format (e.g. an
older, manually-uploaded `.png`) is left alone. `--regenerate` skips the
tier-1 lookup and forces a fresh tier-2/3 fetch, re-uploading the result;
`--skip-upload` fetches without writing anything back to OMERO at all.

Two more flags exist for running this across a whole group:

- **`--skip-existing`** skips any image that already has a recon for this
  downsample and format, on one cheap "does this annotation exist" check --
  no download, no regeneration. Cached images cost almost nothing, only the
  missing ones do real work. Contradicts `--regenerate`, so those two are
  rejected together.
- **`--skip-local`** keeps no local copy: it fetches/generates to a
  temporary file, uploads that, then deletes it. OMERO's upload API reads
  from a path rather than raw bytes, so a file still has to exist briefly
  -- it just doesn't survive the run. Incompatible with `-o` (nothing to
  point at) and with `--skip-upload` (that pair would generate an image and
  throw it away).

So the backfill-a-whole-group-into-OMERO-only invocation is:

```sh
lavlab lr batch -g 3 --workers 8 --skip-existing --skip-local
```

If you don't pass `-o`, output location falls back to `fs_map` -- a YAML
file mapping OMERO group -> filesystem destination by regex match on the
image name (`src/lavlab/data/default_fs_map.yaml` ships a lab default; `--fs-map
custom.yaml` overrides it). No match, or the mapped directory doesn't
exist on disk: single-image mode writes to the current directory instead;
batch mode skips that image and continues, warning rather than failing the
whole run. An `fs_map` entry whose `base_dir` doesn't exist at all *does*
abort the run -- that means the whole storage target is unmounted, not
just one path.

### `lavlab roi` -- pull ROI mask images

```sh
lavlab roi 12345 --all -o ./out/mask.jp2 -s omero.example.edu -u you
lavlab roi batch -g 3 -t tumor -t stroma --palette -s omero.example.edu -u you
```

Renders OMERO ROI annotations to a raster mask instead of pulling the
slide itself. You must pass either `--all` (every annotation) or one-or-more
`-t/--text-filter` values (a whitelist by the shape's `textValue`) --
"give me nothing" isn't a sensible default for a mask export, so the
command refuses to run with neither. `--palette` swaps the RGB color mask
for a single-channel label mask, numbered in the order your `-t` filters
were given; it's incompatible with `--all` since there's no sane numbering
for "everything." Output format defaults to `jp2` (`--format
jpg`/`jpeg`/`png`/`tif`/`tiff` picks a different one) -- except with
`--palette`, which refuses to combine with `jpg`/`jpeg`: a palette mask's
pixel values are exact integer labels (`0`, `1`, `2`, ...), and JPEG's
lossy compression would silently corrupt them.

By default `roi` only ever *reads* from OMERO. `--upload` additionally
attaches the rendered mask to the image, under the
`LargeRecon.${downsample}.roi` namespace (the one legacy `batch_roi.py`
used), named `LR${downsample}_${image name}_${suffix}.${format}`. It's
opt-in precisely because everything else this command does is read-only.
`--skip-existing` then skips any image that already has that exact mask
attached, and `--skip-local` renders to a temporary file and deletes it
after uploading -- so the group-wide, OMERO-only backfill is:

```sh
lavlab roi batch -g 3 --workers 4 --all --upload --skip-local --skip-existing
```

`--skip-local` requires `--upload` here (unlike `lr`, where uploading is
the default, so the equivalent conflict is with `--skip-upload`) -- without
it the mask would be rendered and immediately thrown away.

**`roi` masks and `lr` recons never collide**, despite the shared namespace
prefix: OMERO filters annotations by exact namespace equality, so
`LargeRecon.10` and `LargeRecon.10.roi` are separate buckets. `lr`'s lookup
matches on file extension alone, so that separation is the only thing
keeping it from mistaking a mask for a recon -- there's a regression test
(`test_roi_and_lr_annotations_do_not_collide`) pinning it down. `roi`'s own
lookup additionally matches the whole filename, since `_annot` and
`_exclude` masks legitimately share one namespace and differ only by
suffix.

### `lavlab tile` -- cut slides into training tiles

Cuts fixed-size tiles for a Gleason-grade classifier (or any
ROI-labelled dataset), plus whole-slide tiles for inference. Exactly one of
`--whole` or `--roi` is required; `--roi` also needs `--all` or at least one
`-t/--text-filter` (a `textValue` or a class folder name, case-insensitive).
`lavlab tile --help` lists every flag in groups and ends with examples.

#### Recipes

**1. Check before you extract.** `--dry-run` works out every tile and
prints counts per class and per ROI, plus any ROIs too small to hold a
tile, without reading a single full-resolution pixel or writing anything:

```sh
lavlab tile 12345 --roi --all --size 512 --dry-run -o ./tiles
lavlab tile batch -g 3 --roi --all --size 512 --dry-run -o ./tiles   # whole group, totals at the end
```

This is the cheap way to answer "do I have enough G4 cribriform?", and to
check that `--erode` (on by default, see below) isn't wiping out your small
ROIs.

**2. Build a training set.** The usual run: every annotation, 512 px tiles
at 0.5 um/px, JPEG, one combined index for the whole dataset, resumable:

```sh
lavlab tile batch -g 3 --roi --all --size 512 --format jpg \
    --skip-existing --manifest ./tiles/dataset.csv -o ./tiles
```

Only some classes:

```sh
lavlab tile batch -g 3 --roi -t G3 -t G4cg -t G4fg -t G5 --size 512 \
    --manifest ./tiles/dataset.csv -o ./tiles
```

Point several commands (different groups, classes or settings) at the same
`--manifest` and it accumulates into one file; rows already in it aren't
repeated.

**3. Two scales per tile (detail + context).** `--scales` cuts concentric
crops around the same centre in one pass -- far cheaper than extracting
twice. The pairs share a name apart from the scale:

```sh
lavlab tile batch -g 3 --roi --all --size 512 --scales 0.5,1.0 -o ./tiles
# -> ..._x76800_y112640_mpp0.5.png  and  ..._x76800_y112640_mpp1.0.png
```

The finest scale decides which tiles are kept and their labels; the coarser
crop is context around it. `--erode` is measured at that finest scale too,
so the context crop can reach past the ROI edge -- the run logs by how much.
If you need *every* scale inside the ROI, erode by half the coarsest crop:
`--size 512 --scales 0.5,1.0 --erode 256` (512 px x 1.0 um/px / 2).

**4. Whole-slide tiles for inference.** Every tissue tile, overlapping by
half:

```sh
lavlab tile 12345 --whole --size 512 --stride 256 -o ./inference
```

**5. Interrupted? Run the same command again with `--skip-existing`.**
Finished slides are skipped, and a slide that was cut off partway picks up
from the tiles it already wrote instead of starting over.

**6. A quick look at a few tiles.** Cap the per-slide count:

```sh
lavlab tile 12345 --roi --all --max-tiles 50 -o ./scratch
```

#### Tuning

| If... | Try |
|---|---|
| Cribriform / big-lumen glands are missing | `--tissue-threshold 0.25` (default `0.5`): lumens read as glass |
| Small ROIs produce no tiles | Check with `--dry-run`; smaller `--size`, or `--erode 0` to fall back to coverage-only labelling |
| Too many benign tiles | `--max-background-tiles 500` (default `2000` per slide), or `--no-background` |
| Classes badly imbalanced | `--max-tiles-per-label N` |
| Disk filling up | `--format jpg` (quality `90`; `--quality` to change) |
| Network fetches are the bottleneck | `--connections 8` (default `4` per slide), and/or more `--workers` |
| Want the same tiles every time | They already are: sampling is seeded (`--seed`, default `0`) |

#### How it decides

**Scale.** `--mpp` (default `0.5`) is read against the image's own
`getPixelSizeX`/`Y` to pick a pyramid level and a resize factor, so tiles
from a 20x and a 40x scan come out at the same magnification. An image
with no physical pixel size recorded is a hard error naming `--downsample
N`, which sets the factor directly. `--mpp`, `--scales` and `--downsample`
are mutually exclusive; `--stride` and `--overlap` are two ways to set the
same spacing.

**Which tiles get kept.** Everything is decided on one low-resolution
thumbnail per slide -- tissue by Otsu on saturation, ROI coverage by
rasterising the annotations and reading integral images -- and full-res
pixels are only read for tiles that already passed. In order:

- below `--tissue-thresh` (default `0.5`) tissue: dropped.
- overlapping an exclusion ROI *at all*: dropped. `--exclude-text`
  (repeatable, default `exclusion roi` -- the palette's own label) sets
  which `textValue`s count, and it applies in `--whole` mode too.
- at or above `--min-coverage` (default `0.5`) of one class, **and** centred
  far enough inside that class's ROI (`--erode`): takes that class,
  highest-covering one if several qualify. Enough coverage but too near the
  edge: dropped -- never called benign.
- otherwise, possibly benign -- see below.

**`--erode`** (default `auto` = half a tile). A tile may only take a class
if its centre lies at least this far inside that class's ROI; at half a
tile, that means the whole tile lies inside the ROI, so a tile can't be
labelled by an annotation it only half overlaps. Give micrometres
(`--erode 128`) to set it explicitly. `--erode 0` turns it off and warns
every run: coverage alone lets a tile that straddles an ROI edge take that
class, which is the label noise erosion is there to remove. The inset
actually applied is shown by `--dry-run` and saved in `tile_params.json`.
ROIs too small to hold a tile are listed by `--dry-run` and counted in
every summary -- useful information about your annotation sizes in itself.

**The benign rule (`--roi` only).** Annotators mark *everything* on a slide
they work on: cancer, atrophy, HGPIN, exclusions. So tissue that falls
inside no ROI on an annotated slide is genuinely benign, and gets
`--background-label` (default `benign`). Two guards keep that honest:

- `--background-margin` (default `200` um) dilates the union of *every*
  ROI on the slide -- including ones `-t` didn't select -- before the test,
  so a tile straddling an annotation edge is dropped rather than called
  benign.
- A slide with **no ROIs at all** is unannotated, not benign. It is skipped
  entirely and counted separately in the batch summary. `--no-background`
  turns the whole rule off.

Background still vastly outnumbers the graded classes, so
`--max-background-tiles` (default `2000`) caps it per slide. `--max-tiles`
and `--max-tiles-per-label` cap the rest. All three sample randomly under
`--seed` (default `0`), so the same slide and settings yield the same
tiles every time.

`--whole` is deliberately separate from all of this: its output goes to a
flat `whole/` folder and is never treated as a benign class. Use it for
inference, or for slides nobody has annotated.

#### What you get

```text
<out>/<subject>/<slide_stem>/
  <label>/<slide_stem>_x{X}_y{Y}.png            # --roi (X, Y = level-0 top-left)
  <label>/<slide_stem>_x{CX}_y{CY}_mpp{S}.png   # --roi --scales (CX, CY = level-0 centre)
  whole/<slide_stem>_x{X}_y{Y}.png              # --whole
  manifest.csv
  tile_params.json
```

`<slide_stem>` comes from `naming.get_stem`, so it matches `lr`/`roi`
output names. `<subject>` is the leading `N###` token of the image name
(`N101_S08_HE` -> `N101`), used verbatim -- it is deliberately *not*
translated to the on-disk subject directory, whose naming the bundled
`fs_map` gets wrong in known ways. A name with no such token lands in
`unknown_subject/` with a warning.

**`manifest.csv`** has one row per tile file, starting with the columns a
dataset loader needs:

```text
tile_path,omero_image_id,case_id,slide_id,roi_id,label,x,y,mpp,size,...
N101/N101_S08_HE/G3/N101_S08_HE_x76288_y112128.png,12345,N101,N101_S08_HE,551,G3,76800,112640,0.5,512,...
```

`case_id` is the subject, for patient-level train/test splits; `roi_id` is
the OMERO ROI the tile came from; `x`/`y` are the tile's level-0 *centre*
(shared by every `--scales` crop of one tile). The remaining columns
(`shape_id`, the level-0 box, pyramid level, coverage, tissue fraction,
tier) record how each tile was chosen -- see
[`docs/reference.md`](docs/reference.md) for all of them. `--manifest`
writes the same columns, with `tile_path` relative to the CSV's own
folder so the index and tiles can be moved together.

**Every tile describes itself**, too: the image ID, ROI ID, label, centre
and mpp are embedded in the file (a PNG text chunk, or JPEG EXIF), so a
tile that ends up on its own is still traceable:

```python
from PIL import Image
Image.open("tile.png").text["lavlab"]      # PNG
Image.open("tile.jpg").getexif()[270]      # JPEG
```

`tile_params.json` records every setting that affects the output;
`--skip-existing` compares against it, so a slide tiled with *different*
settings is warned about and skipped rather than mixing two settings in one
directory -- the global `--override` flag (before the subcommand: `lavlab
--override tile ...`) re-tiles it. Output from an older lavlab version
counts as different settings.

#### Where the pixels come from

Like `lr`, but with only two tiers -- there is no cached-recon tier for
tiles:

1. **Local source file**, when OMERO's managed repository is mounted where
   `tile` runs (true on the cluster, not on a workstation). Regions are
   cropped straight out of the source pyramid with pyvips random access --
   never a whole level into memory, since these slides reach ~220k x 260k.
2. **OMERO's tile API**, otherwise. Only regions containing kept tiles are
   fetched, batched a grid-band at a time over `--connections` parallel
   connections, rather than a round trip per tile.

The tier used is logged per slide and recorded in every manifest row. A
local source that won't decode falls forward to the network tier with a
`WARNING`, exactly as `lr` does.

**JPEG-2000 sources** are a special case of that fall-forward: the bundled
libvips has no jp2k loader, so random reads would go through Pillow, which
decodes the entire codestream *per crop*. A bare `.jp2` source therefore
falls forward to the network tier with a `WARNING` rather than grinding
through 10^5 whole-file decodes. `--force-local` overrides it if you really
want that.

**Tier parity caveat.** Both tiers share one grid, one region-reader
interface and one resize call, so the same slide and settings select the
same tiles and the same pyramid level either way. What they cannot
guarantee is bit-identical *pixels*: tier 2 reads the source file's own
pyramid and tier 3 reads OMERO's, and for lossily-compressed levels those
can differ by a quantization step. Don't mix tiers within one training set
if that matters to you.

**Class folders** come from a `{folder: [textValue aliases]}` YAML
(`src/lavlab/data/default_tile_labels.yaml`, overridable with `--labels`),
matched case-insensitively and trimmed. The default maps the palette's
`G4CG`/`G4FG` spellings onto the `G4cg`/`G4fg` folders the classifier
expects, and lists the non-Gleason palette labels (`Atrophy`, `HGPIN`,
`Seminal_Vesicles`, `Vessel`, `Urethra`) so `--all` doesn't warn about
them. With `--all`, an unmapped `textValue` becomes a sanitized folder of
its own, warned about once per slide.

Batch mode matches `lr batch`/`roi batch`: per-slide failures warn and
continue, and the run ends with per-tier, per-label, ROI/erosion and
done/skipped/unannotated/failed totals, with the image IDs for each. Any
error on one slide -- including one nobody anticipated -- marks that slide
failed and the batch carries on; only a slide whose *connection* dropped is
retried, after logging in again. Workers stagger their first login, and if
OMERO's session service starts refusing logins (an Ice
`ProtocolException`), retries back off longer and at random intervals
instead of hammering it.

### `lavlab meta roi textvalue` -- backfill ROI comments from stroke color

```sh
lavlab meta roi textvalue default 12345 12346 -s omero.example.edu -u you
lavlab meta roi textvalue ./my_palette.yaml -g 3 --tolerance 15 -s omero.example.edu -u you
```

The one command that edits ROIs in place rather than producing a file: for
each shape, it matches the stroke color against a palette (color -> label
name) and writes the label into `textValue` -- but only when that field is
currently blank, so it never overwrites something someone already typed.
`default` refers to a built-in palette; point it at your own YAML/JSON
instead if needed. `--tolerance` (default 10) is the per-channel color-match
slack, since colors don't always round-trip through OMERO's storage
byte-for-byte.

This is the one command that writes to OMERO by default. Run it with
`--dry-run` first -- it reports exactly what would change (each matched
shape's ID and label included) without touching anything -- especially
since a plausible-but-wrong match is easy to get: many drawing tools
default an unset stroke color to plain black, which is indistinguishable
from an intentionally-black palette entry once it's in OMERO. Spot-check a
few matched shape IDs in OMERO.web before re-running without the flag.

### `lavlab geojson import` / `export` -- QuPath GeoJSON <-> OMERO ROIs

```sh
lavlab geojson import slide.geojson --image 12345 -s omero.example.edu -u you
lavlab geojson import ./backups/ --dataset 5 -s omero.example.edu -u you       # match files to images by filename
lavlab geojson import --map restore.csv -s omero.example.edu -u you            # explicit path,image_id CSV -- the auditable choice

lavlab geojson export --project 2 --out ./archive/ --datestamp -s omero.example.edu -u you
```

QuPath draws annotations and exports them as GeoJSON; OMERO stores
annotations as ROI database rows. The two models disagree in ways that
lose data if you're not careful:

- **Holes.** OMERO's polygon has no concept of an interior ring -- a donut
  shape is just one flat point list. A hole gets "bridged" in as a
  zero-width slit cut from the outer ring to the nearest hole vertex,
  which renders correctly as a hole under the standard nonzero fill rule
  (`src/lavlab/geojson/geometry.py`'s `bridge_hole`). `export` reverses this
  automatically (`unbridge_ring`); `--keep-bridges` turns that off if you
  specifically want the raw bridged shape back. Detecting a genuine bridge
  vs. an ordinary duplicate point (e.g. a dense freehand trace revisiting
  the same rounded pixel at its own closing seam) is the fiddly part --
  `unbridge_ring` drops consecutive duplicate points before looking for a
  bridge, specifically so a shape like that doesn't get misread as "one
  giant hole" and silently dropped for having under 3 points left.
- **Mixed shape kinds in one ROI.** An OMERO ROI can hold shapes of
  different kinds together -- a Polygon and a Point, say -- but a plain
  GeoJSON geometry can only be one type. `export` falls back to a
  `GeometryCollection` feature for those ROIs instead of dropping them;
  `import` reads `GeometryCollection`, `Point`/`MultiPoint`, and
  `LineString`/`MultiLineString` features right back into the matching
  shape kinds (not just `Polygon`/`MultiPolygon`), so nothing that a
  from-scratch export can produce fails to come back on import.
- **Coordinate precision.** Whole-slide coordinates are large enough that
  naive `%g` formatting would silently round to 6 significant digits --
  on a slide past ~100,000px that's more than a pixel of drift on the
  majority of vertices. Coordinates are written out explicitly instead.
- **Provenance.** OMERO has nowhere to record "this ROI came from QuPath
  object such-and-such." A small JSON tag is stashed in `Roi.description`
  on import and read back out on export, so a round trip doesn't lose the
  original QuPath UUID.

`import` accepts exactly one of `--image` (single file), `--dataset`
(match files to images by filename), or `--map` (an explicit
`path,image_id` CSV -- reach for this one when it matters, since you can
eyeball it before running rather than trusting a filename match).
`export` accepts exactly one of `--image`, `--dataset`, or `--project`.
`--datestamp` writes into `<out>/YYYY-MM-DD/` so repeated archive runs
don't collide; without it, an existing output file blocks the run unless
you pass `--overwrite`.

`export` can also archive straight into OMERO rather than to disk.
`--group <id>` exports every image in a group; `--upload` attaches each
export to its own image under the `lavlab.geojson` namespace, named the
same as the local file would be (`<image name>__omero-<id>.geojson`);
`--skip-existing` skips images already archived that way; and
`--skip-local` writes to a temporary file, uploads it and deletes it, so
`--out` isn't needed at all:

```sh
lavlab geojson export --group 3 --upload --skip-local --skip-existing
```

The `lavlab.geojson` namespace is deliberately *not* under `LargeRecon.*`
like the `lr` and `roi` attachments: those are artifacts of one specific
downsample, while a GeoJSON export is vector data with no resolution
attached to it. Since OMERO matches namespaces by exact equality, all
three kinds of attachment coexist on an image without any command picking
up another's file -- there's a test (`test_geojson_namespace_does_not_
collide_with_lr_or_roi`) covering all three at once.

Unlike `lr batch` and `roi batch`, this walks the group sequentially with
no worker pool. Exporting ROIs is comparatively cheap -- database rows,
not pixel data -- and staying single-process avoids the fork-safety
problems that multiprocessing with Ice brings on macOS.

### `lavlab seg dcm2nii` / `nii2dcm` -- DICOM SEG <-> NIfTI

```sh
lavlab seg dcm2nii segmentation.dcm reference.nii.gz --out ./out/
lavlab seg nii2dcm mask.nii.gz ./dicom_series/ --out ./mask_seg.dcm --label "Tumor" --comment "reviewed by X"
```

Local file conversion; neither direction talks to OMERO.

`dcm2nii` splits a DICOM SEG object into one NIfTI file per segment,
aligned to a reference NIfTI image, named
`<reference-stem>_<segment-name>.nii.gz`.

`nii2dcm` writes a NIfTI label mask back out as a DICOM SEG object.
Because a NIfTI file carries no patient/study/geometry metadata of its
own, you point it at the reference DICOM series the mask was drawn
against, and it borrows that. It writes a `LABELMAP`-type segmentation --
the modern multi-class variant of the DICOM SEG standard (one integer
label per voxel, `0` = background), as opposed to the older `BINARY`
variant (one frame stack per segment). What segments exist and how
they're described (label + SNOMED code) comes from a JSON template --
`src/lavlab/data/default_seg_template.json` unless you pass `--template
your.json`. **This is not the dcmqi metainfo format** if you've used
`dcmqi`/`pydicom-seg` templates before -- it's a simpler schema built
around `highdicom.seg.SegmentDescription`; see
[`docs/reference.md`](docs/reference.md#--template-schema-lavlab-seg-nii2dcm) for the
exact shape.

## Development

```sh
pip install -e ".[dev]"
pytest
```

Tests that need the imaging/DICOM stack (`SimpleITK`, `pydicom`) are
skipped automatically if those packages aren't installed; the GeoJSON
tests have no such dependency and always run. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the patterns to follow when
adding to this codebase.

## Building the compiled wheel

The native executable is always compiled with Python 3.12. Published wheels
are `py3-none` wheels and contain that same executable, so one platform wheel
supports Python 3.11, 3.12, 3.13, and 3.14 as the interpreter used to install
and launch it. Builds are supported only on Apple Silicon macOS and x86_64
Linux; the macOS wheel is not universal.

For a local build, use Python 3.12 and the target-specific command:

```sh
pip install -r build-requirements.txt   # + the Ice wheel for omero-py, see that file's comment
make build-mac                         # Apple Silicon macOS only
# or
make build-linux                       # Linux x86_64 only
```

**Use `setup.py bdist_wheel` directly, not `pip wheel .`.** `pip wheel`
builds in an isolated PEP 517 environment that (a) can't see the
manually-installed Ice wheel and tries to rebuild `zeroc-ice` from source
instead, and (b) fails with `ModuleNotFoundError: No module named
'build_native'`, since the repo root isn't on `sys.path` under pip's build
hooks even with `--no-build-isolation`. Running `setup.py` directly puts
its own directory on the path and uses the environment you already set up.

Without `LAVLAB_PREBUILT_DIST`, this shells out to Nuitka (`setup.py`'s
`build_py` override, or run `build_native.py` standalone if you just want the
compiled binary without a full wheel) to compile `src/lavlab/__main__.py` into a standalone
executable, bundled into the wheel as `lavlab/bin/dist/`; the
`lavlab` console-script just execs `lavlab/bin/dist/lavlab-bin`. The
build is `--standalone` rather than `--onefile`: the artifact is a
directory, because `lavlab-bin` resolves its bundled shared libraries
through `RPATH=$ORIGIN` and has to stay next to them. Onefile was
dropped deliberately -- its bootstrap reports a child killed by a signal
as exit 0, which turned a crash into a silent success everywhere,
including in CI. The `.github/workflows/build.yml` workflow builds each
platform's executable once using
Python 3.12, then packages and smoke-tests it with Python 3.11 through 3.14.
The workflow rejects builds on Intel macOS and non-x86_64 Linux. See
[Install](#install) above for how a wheel gets to a lab member.

**Both `setup.py` and `build_native.py` include a fixed set of extra
`--include-module=`/`--include-package-data=` flags for `pydicom`**
(`PYDICOM_NUITKA_FLAGS` in `build_native.py`), on top of the OMERO Ice
modules `omero_ice_modules()` already discovers dynamically. pydicom 3.x
loads its pixel data decoders/encoders as a plugin-style set of submodules
rather than through top-level imports, which Nuitka's static import
analysis misses -- without these flags, a compiled binary can build and
even launch fine, but fail the first time `lavlab seg` actually needs to
decode/encode DICOM pixel data. If you hit a similar "works from source,
breaks in the compiled binary" issue with some other dependency, that's
almost always this same class of problem: something doing dynamic/plugin
imports Nuitka can't see statically.

**`IMAGECODECS_NUITKA_FLAGS` is the second instance of that class.**
`tifffile` parses a TIFF container but hands tile decompression to
`imagecodecs`, which resolves each codec through a module-level
`__getattr__` doing `importlib.import_module('.' + name, 'imagecodecs')`.
Tier 2 (reading a source file off a mounted OMERO repository) needs this
for the JPEG- and JPEG2000-compressed OME-TIFFs the lab actually stores;
tier 3 never did, because the tile API returns already-decoded pixels.
Two traps worth knowing if you extend the list:

- On a failed import `imagecodecs` substitutes a **stub** that only raises
  when the codec is called, so a wrongly-flagged build still imports and
  launches cleanly and only dies at the first compressed tile.
- Every codec extension `cimport`s `imagecodecs._shared_cython` at the C
  level, with no Python `import` statement anywhere for Nuitka to follow.
  Omit it and *all* the codecs fail, each reporting its own
  `DelayedImportError` rather than the one shared cause.

`tests/test_imaging.py` guards both: it asserts every flagged module still
exists upstream, and decodes a file in each compression scheme.

`LAVLAB_NUITKA_ARGS` (an environment variable) lets you pass additional
raw Nuitka flags for a one-off build without editing the build scripts --
e.g. `LAVLAB_NUITKA_ARGS="--include-package=some_other_thing" pip wheel .`.
`LAVLAB_OUTPUT_DIR` (read by `build_native.py` only) controls where the
standalone-script build writes its output (default: `./native/`).

## Building with Docker

A `Dockerfile` in the repo root provides a reproducible build environment
-- a pinned image with the C compiler, Nuitka, and the full dependency
stack (including the Ice wheel) pre-installed, so the build doesn't depend
on what happens to already be on your machine. It builds against your
*live* source tree (mounted in at `docker run` time), not a snapshot baked
into the image, so you don't need to rebuild the image after every source
change.

```sh
# once: get the lab's Ice wheel for linux-x86_64 into ./vendor/
mkdir -p vendor
curl -L -o vendor/zeroc_ice-<version>-<platform>.whl <lab-internal-URL>

# once (or after a dependency change): build the environment image
docker build -t lavlab-builder .

# every time you want a fresh wheel from your current source:
docker run --rm -v "$(pwd)":/src -v "$(pwd)/dist":/out lavlab-builder
```

The resulting wheel lands in `./dist/` on your host, same as running
`pip wheel . -w dist/` directly would -- this just guarantees the
environment it ran in. See the comments at the top of `Dockerfile` for
more detail. `.dockerignore` excludes `legacy/` from the build context for
the same reason `.gitignore` does (see below) -- nothing in there should
ever leave your machine.

## Troubleshooting

- **A compiled binary crashes on something `python -m lavlab` handled
  fine.** Almost always a Nuitka static-analysis miss -- some dependency
  doing dynamic/plugin-style imports that the build didn't know to
  include. See the pydicom example under "Building the compiled wheel"
  above, and `LAVLAB_NUITKA_ARGS` for the escape hatch while you figure out
  the right permanent flag to add to `PYDICOM_NUITKA_FLAGS`/the OMERO Ice
  discovery.
- **`SimpleITK`/GDCM can't read a DICOM SEG file `lavlab seg nii2dcm`
  wrote.** Expected for the `LABELMAP` segmentation type this command
  writes -- SimpleITK/GDCM doesn't support parsing it directly yet, even
  though `highdicom` (which `lavlab seg dcm2nii` uses) reads it fine. This
  doesn't affect `lavlab seg dcm2nii`'s own output, which works around it
  automatically (`dcmseg_to_nifti` falls back to the reference NIfTI's
  geometry when `sitk.ReadImage` can't read a `LABELMAP` SEG's directly --
  see the `try`/`except RuntimeError` in `src/lavlab/seg.py`) -- it only
  matters if you're trying to open the file with some *other* tool that
  goes through SimpleITK.
- **`Image N: OMERO could not read this image's pixels`** (an
  `omero.ResourceError`, "Error instantiating pixel buffer"). The image's
  file is missing or corrupt on the OMERO server -- nothing on your side
  will fix it, so `lavlab` skips that image, lists it under the failed IDs
  and carries on. Send the IDs to whoever runs the OMERO server.
- **`pip install -e ".[dev]"` fails on `omero-py`.** You need the Ice
  wheel installed first -- see the Install section above.
