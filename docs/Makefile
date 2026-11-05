# Build the docs site, Sphinx-style: `make html` in docs/.
# Run `make help` for the targets.
#
# Quarto writes its cache (.quarto/) and each page's intermediate files next to
# the sources, so it renders a copy of docs/ in _build/ instead, into _site/.
# The generators write the files that some pages include into _build/_generated/.

RUN ?= uv run
BUILD := _build

# Excluded paths in _build/ also survive --delete.
SYNC = rsync -a --exclude /$(BUILD)/ --exclude /_site/ --exclude /scripts/ \
	--exclude /Makefile --exclude __pycache__/ --exclude /_generated/ \
	--exclude /.quarto/ ./ $(BUILD)/

.PHONY: help html preview generated data figures clean

help:
	@echo "make html       render the full site into _site/"
	@echo "make preview    live-reloading preview"
	@echo "make generated  regenerate the results pages, solver reference and plots"
	@echo "make data       fetch the published baseline into mosaic-results/ (needs gh)"
	@echo "make figures    redraw the setup schematics"
	@echo "make clean      remove _build/ and _site/"

html: generated
	$(SYNC) --delete
	cd $(BUILD) && $(RUN) quarto render --output-dir ../_site

# Copies edits into _build/ every second while the preview runs. Without
# --delete, so the copy loop leaves the files of a page being rendered alone.
preview: generated
	$(SYNC) --delete
	while sleep 1; do $(SYNC); done & trap 'kill $$!' EXIT; \
		cd $(BUILD) && $(RUN) quarto preview --output-dir ../_site

# Quarto resolves includes before it runs any pre-render script, so the pages
# that include _generated/ need it in place before Quarto starts. The solver
# reference reads the per-solver checks that the results script writes.
generated:
	$(RUN) python scripts/generate_results.py
	$(RUN) python scripts/generate.py

# The baseline stores data only, so the plots are regenerated from it.
data:
	cd .. && GITHUB_TOKEN=$$(gh auth token) GITHUB_REPO=pasteurlabs/mosaic \
		bash .github/scripts/fetch-artifact.sh baseline mosaic-results/
	cd .. && $(RUN) mosaic run --plots-only

figures:
	$(RUN) python scripts/make_domain_figures.py

clean:
	rm -rf $(BUILD) _site
