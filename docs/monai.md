# Training with MONAI: `lavlab.monai`

`lavlab tile` decides which tiles a slide yields: the grid, the tissue mask,
ROI coverage, `--erode`, exclusions, the benign rule, and the seeded caps.
`lavlab.monai` hands those same tiles to [MONAI](https://monai.io) as
datasets, along one of two paths:

| | `LavlabTileDataset` | `OmeroPatchDataset` |
|---|---|---|
| Pixels come from | tiles `lavlab tile` already wrote | OMERO, on demand |
| Needs first | one `lavlab tile ... --manifest dataset.csv` run | nothing |
| Cost per epoch | decode a PNG/JPEG from disk | an OMERO read per patch (a crop on the cluster, round trips over the network) |
| Use it for | **training** | exploring, prototyping, small or changing selections |

Both yield the same sample dict, built on the same tile selection, so a
model can be prototyped on live patches and then trained on the
extracted set without anything changing shape.

## Install

`lavlab.monai` is **not** part of the `lavlab` binary, and installing the CLI
doesn't pull in MONAI or PyTorch. It is a Python-only extra:

```sh
# The Ice wheel first, exactly as for a source checkout (see the README).
pip install 'lavlab-cli-utils[monai]'
```

From a source checkout, install `requirements.txt` plus `monai torch Pillow`
and put `src/` on `PYTHONPATH`. Without the extra, importing a dataset
raises an `ImportError` that names it. `build_datalist` needs only
lavlab's own dependencies.

## When to use which

Train on extracted tiles. A live read costs at least a pyramid crop on
the cluster, and several OMERO round trips per patch everywhere else,
every epoch. Over 10^5 patches and tens of epochs that is far slower than
decoding a JPEG, and it puts a training job's load on the OMERO server.
Extracting once (around 135 GB as JPEG for the full Gleason set) and
reading from disk is the fast, reproducible path.

The live dataset is for everything before that: checking that a class
has enough tiles to learn from, trying a size or scale, or running a quick
experiment on a few slides without committing disk to them. `cache_rate`
can hold patches in memory for a few epochs over a small selection.

## Path B: train on extracted tiles

```sh
lavlab tile batch -g 3 --roi -t G3 -t G4cg -t G4fg -t G5 --size 512 \
    --format jpg --skip-existing --manifest /data/tiles/dataset.csv -o /data/tiles
```

```python
from monai.data import DataLoader
from monai.transforms import RandFlipd, ScaleIntensityRanged
from lavlab.monai import LavlabTileDataset

transform = [
    ScaleIntensityRanged("image", a_min=0, a_max=255, b_min=0.0, b_max=1.0),
    RandFlipd("image", prob=0.5, spatial_axis=1),
]
train = LavlabTileDataset("/data/tiles/dataset.csv", "training", transform, seed=0)
val = LavlabTileDataset("/data/tiles/dataset.csv", "validation", transform[:1], seed=0)

loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=8)
print(train.class_names)  # ['G3', 'G4cg', 'G4fg', 'G5', 'benign']
```

`tile_path` is resolved against the CSV's own folder, never the working
directory, so the CSV and its tiles can move together. A per-slide
`manifest.csv` holds paths relative to the `-o` directory instead; pass
`root="/data/tiles"` for one of those.

`classes=[...]` fixes the label order and leaves out other labels. Pass it
whenever datasets are built from different manifests, so label indices
agree across them.

## Path A: read live from OMERO

```python
from lavlab.monai import OmeroPatchDataset

train = OmeroPatchDataset(
    image_ids=[1962, 7775],
    labels=["G3", "G4cg", "G4fg", "G5"],
    size=512, mpp=0.5, erode=64, min_coverage=0.75,
    transform=transform,
)
```

The selection arguments are `lavlab tile`'s flags under Python names, run
through the CLI's own validation and defaults (`lavlab.monai.tile_params`
shows the mapping): `labels` is `-t`, `all_labels` is `--all`, `scales`
is `--scales`, `background_label=None` is `--no-background`, and so on.
`tile_seed` is `--seed`. `seed` is the split seed, as in every `monai.apps`
dataset. `group=3` replaces `image_ids` for a whole OMERO group, like
`lavlab tile batch -g 3`.

Building the dataset analyses every slide (one thumbnail and the ROIs
each, no full-resolution pixels). To build several sections or folds
without repeating that, build the datalist once and pass it in:

```python
from lavlab.monai import build_datalist

datalist = build_datalist(group=3, all_labels=True, size=512, mpp=0.5)
train = OmeroPatchDataset(datalist=datalist, section="training")
val = OmeroPatchDataset(datalist=datalist, section="validation")
```

Credentials come from `creds=` or the usual `OMERO_USER` /
`OMERO_PASSWORD` / `OMERO_HOST` / `OMERO_PORT` variables. Patches come
through the same two tiers as `lavlab tile`: a pyramid crop when the
managed repository is mounted, OMERO's tile API otherwise. The same
JPEG-2000 rule and fall-forward-with-`WARNING` apply. The tier that served
each patch is in `sample["image"].meta["tier"]`.

### DataLoader workers

Every process opens its own OMERO session on its first read. A session
inherited from the parent is never used or closed, and each worker closes
its own on exit. Local-tier reads need no session at all.

**Use `multiprocessing_context="spawn"` (or `"forkserver"`) for
`OmeroPatchDataset` workers:**

```python
loader = DataLoader(train, batch_size=32, num_workers=8,
                    multiprocessing_context="spawn")
```

libvips cuts the tiles, and once it has run in a process, a `fork` of
that process hangs on its first libvips call, on Linux as on macOS.
Building the dataset already runs it, for the thumbnail. With `fork` (the
Linux default) the worker stops with an error naming this setting
instead of hanging. `LavlabTileDataset` reads tiles with Pillow and has no
such restriction.

## Samples

Both datasets yield:

| Key | |
|---|---|
| `image` | `MetaTensor`, `[C, size, size]`, `uint8`, channel-first |
| `label` | class index into `dataset.class_names` |
| `label_name` | the class folder name, e.g. `G4cg` |
| `omero_image_id`, `case_id`, `slide_id` | as in the manifest |
| `roi_id`, `shape_id` | the OMERO ROI and shape that labelled the tile; `-1` for background |
| `x`, `y` | level-0 tile centre |
| `x0`, `y0`, `w0`, `h0` | level-0 box |
| `mpp`, `tile_size`, `level` | output scale, edge in pixels, pyramid level read |
| `coverage`, `tissue_frac` | why the tile was kept |
| `tier` | `local` or `network` |

The values match the manifest row of the same tile, except that `-1` and
`NaN` stand in for the manifest's blanks so batches collate.

**Multi-scale.** With `scales=[0.5, 1.0]` (or a `--scales` extraction), each
sample is one tile *location*: `image` is the finest crop, the labelling
scale, and `image_1`, `image_2`, ... are the coarser concentric crops, with
`scales` listing their mpps. Separate keys, rather than one stacked
tensor, let MONAI dictionary transforms apply the same random flip to
every scale: `RandFlipd(["image", "image_1"], ...)`. A combined manifest
that mixes runs at different scales needs `scales=` to say which crops
make a sample.

## Splits are by case

`section` (`"training"` / `"validation"` / `"test"`, with `val_frac` and
`test_frac`) is drawn over **`case_id`s**, not tiles. Every tile of a case
lands in one section, because tiles from one patient on both sides of a
split inflate every metric. `monai.apps.CrossValidation` folds by case
too:

```python
from monai.apps import CrossValidation

folds = CrossValidation(dataset_cls=LavlabTileDataset, nfolds=5, seed=0,
                        manifest="/data/tiles/dataset.csv", transform=transform)
fold0_train = folds.get_dataset(folds=[1, 2, 3, 4])
fold0_val = folds.get_dataset(folds=0)
```

This works because `_split_datalist` is given one entry per case, so any
override of it, `CrossValidation`'s included, partitions cases. Fractions
round down, as in `MedNISTDataset`, so with few cases a section can come
out empty, and a warning says so.

## What is guaranteed to match

- **The tiles.** `build_datalist` and `lavlab tile` both call
  `lavlab.tiling.select_tiles`, so the same slide, settings and seed name
  the same locations, labels and ROIs. A test pins `build_datalist`
  against `lavlab tile --coords-only` row for row, and against the
  `--dry-run` report.
- **The pixels**, for PNG tiles read through the same tier: the live
  reader uses `lavlab tile`'s region readers and its resize, and the test
  suite checks the two datasets tensor for tensor. JPEG tiles differ by
  their compression. The two tiers can also differ slightly on lossily
  compressed pyramids (see *Tier parity caveat* in the README).

## Using the reader on its own

`OmeroWSIReader` is a `monai.data.BaseWSIReader`, so it plugs into anything
that takes one:

```python
from monai.data import PatchWSIDataset
from lavlab.monai import OmeroWSIReader, build_datalist

data = build_datalist(1962, labels=["G3"], size=512, mpp=0.5)
ds = PatchWSIDataset(data, reader=OmeroWSIReader(resize_to=512),
                     center_location=False)
```

Images are named `omero://<image id>` (a bare integer ID also works).
`location` is the level-0 top-left `(row, col)` and `size` is at `level`,
as for every MONAI reader. A datalist's `level` and `size` are where
`lavlab tile` reads, usually the next pyramid level finer than the target
mpp, and `resize_to` shrinks that to the output edge with the same kernel.
Pass `center_location=False`: MONAI's centring subtracts a level-sized
offset from a level-0 location, which is only right at level 0.
`get_mpp` raises the same `MissingPixelSizeError` as `lavlab tile --mpp`
for an image with no pixel size recorded.
