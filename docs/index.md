# lavlab-cli-utils docs

`lavlab-cli-utils` is LavLab's CLI toolbox for OMERO -- pulling large-recon
and ROI mask images, cutting whole-slide images into training tiles,
filling in ROI metadata, moving QuPath GeoJSON annotations in and out of
OMERO, and converting between DICOM SEG and NIfTI segmentation masks. It
ships as a single `lavlab` command, either as
a self-contained [Nuitka](https://nuitka.net)-compiled binary (no Python
environment needed on the target machine) or run directly from source.

## Where to go

- **[../README.md](../README.md)** -- start here. Install instructions,
  every command with examples, the Docker build environment, and
  troubleshooting.
- **[reference.md](reference.md)** -- exhaustive reference: every CLI flag, and the
  importable Python API (`lavlab.geojson`, `lavlab.seg`) for use outside
  the CLI, e.g. from a notebook.
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** -- setting up a dev
  environment, running tests, the pattern to follow when adding a new
  command group, and the conventions this codebase expects.
