# Mosaic docs

This directory holds the sources of the [Mosaic docs site](https://docs.pasteurlabs.ai/projects/mosaic/stable/), built with [Quarto](https://quarto.org).

## Building the site

You need [uv](https://docs.astral.sh/uv/) and Quarto 1.7.31, the version CI uses. Quarto is a standalone binary, available from the [Quarto releases](https://github.com/quarto-dev/quarto-cli/releases/tag/v1.7.31). The Makefile runs everything else through uv in the project environment.

```bash
cd docs
make html      # render the site into _site/
make preview   # live-reloading preview in the browser
make clean     # remove _build/ and _site/
```

Use these targets rather than calling `quarto` directly. Before each build, the Makefile writes the solver reference and the results pages from the solver configurations and the benchmark results. It then renders a copy of this directory in `_build/`, so that Quarto's cache and intermediate files stay out of the sources. Both `_build/` and `_site/` are gitignored.

## Results data

The results pages are filled from `mosaic-results/` at the repository root, and show placeholders if it is missing. To build them with real data, either run the benchmark locally or fetch the published baseline with `make data`. The baseline is several hundred megabytes, and fetching it needs the [GitHub CLI](https://cli.github.com).

The [contributing guide](https://docs.pasteurlabs.ai/projects/mosaic/stable/contributing.html#building-the-docs) covers the build in more detail.
