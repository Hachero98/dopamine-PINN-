# =============================================================
#  Makefile — Dopamine-PINN reproducibility pipeline
# =============================================================
#  Companion to B3_Dopamine_PINN_Paper.tex (PLOS Comp Bio).
#  All targets assume Python 3.9+ with the dependencies pinned in
#  requirements.txt (run `make install` once).
#
#  Typical workflow:
#    make install       # one-time
#    make quick-all     # ~3 min sanity check end-to-end
#    make all           # full run: ~5 min metrics + ~45 min sweep
#    make pdf           # compile the manuscript
#
#  Run `make help` to see every target.
# =============================================================

PYTHON   := python3
DDE_ENV  := DDE_BACKEND=pytorch
TEX      := B3_Dopamine_PINN_Paper
FIG_DIR  := figures
METRICS  := $(FIG_DIR)/metrics.json
SENS     := $(FIG_DIR)/sensitivity.json

.PHONY: all help install quick quick-all metrics sensitivity sweep \
        preview validate restore pdf open clean distclean

# ──────────────────────────────────────────────────────────────
# Default: full reproducibility pipeline
# ──────────────────────────────────────────────────────────────
all: metrics sensitivity validate
	@echo ""
	@echo "  ✓ Full pipeline complete."
	@echo "    Inspect figures/  and run \`make pdf\` to compile."

# ──────────────────────────────────────────────────────────────
# Help — auto-prints when no target is given but `all` is missing.
# ──────────────────────────────────────────────────────────────
help:
	@printf "Dopamine-PINN reproducibility targets:\n\n"
	@printf "  Setup\n"
	@printf "    make install      Install Python dependencies\n\n"
	@printf "  Sanity checks (no GPU needed)\n"
	@printf "    make quick        ~30 s; standard run at reduced fidelity\n"
	@printf "    make quick-all    ~3 min; full pipeline in --quick mode\n"
	@printf "                      (read-only preview, won't commit)\n\n"
	@printf "  Experiments\n"
	@printf "    make metrics      Standard forward + inverse run\n"
	@printf "                      → figures/metrics.json (~5 min GPU)\n"
	@printf "    make sensitivity  All three sweeps\n"
	@printf "                      → figures/sensitivity.json (~45 min GPU)\n"
	@printf "    make sweep        Alias for sensitivity\n\n"
	@printf "  Manuscript auto-fill\n"
	@printf "    make preview      Dry-run validate_metrics.py (show diff)\n"
	@printf "    make validate     Apply substitutions (with .tex.bak backup)\n"
	@printf "    make restore      Revert from .tex.bak\n\n"
	@printf "  LaTeX\n"
	@printf "    make pdf          pdflatex + bibtex + pdflatex x2\n"
	@printf "    make open         Open the compiled PDF (macOS)\n\n"
	@printf "  Cleanup\n"
	@printf "    make clean        Remove TeX aux files only\n"
	@printf "    make distclean    Also remove figures/ and the PDF\n\n"
	@printf "  Full pipeline\n"
	@printf "    make all          metrics + sensitivity + validate\n"

# ──────────────────────────────────────────────────────────────
# Dependencies (one-time)
# ──────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install -r requirements.txt

# ──────────────────────────────────────────────────────────────
# Quick sanity checks — useful before committing to a real run.
# ──────────────────────────────────────────────────────────────
# Single experiment at reduced fidelity (~30 s on CPU)
quick:
	$(DDE_ENV) $(PYTHON) dopamine_PINN.py --quick

# Full pipeline in --quick mode, READ-ONLY substitution check.
# Use this to verify metrics.json + sensitivity.json + validate
# all hook together correctly. Will NOT modify B3.tex; the final
# step is a dry-run (`validate_metrics.py` without --apply).
quick-all:
	$(DDE_ENV) $(PYTHON) dopamine_PINN.py --quick
	$(DDE_ENV) $(PYTHON) dopamine_PINN.py --quick --sweep all
	$(PYTHON) validate_metrics.py
	@echo ""
	@echo "  ✓ quick-all completed (read-only)."
	@echo "    To commit substitutions, run: make validate"

# ──────────────────────────────────────────────────────────────
# Forward + inverse → figures/metrics.json
# ──────────────────────────────────────────────────────────────
metrics: $(METRICS)

$(METRICS):
	$(DDE_ENV) $(PYTHON) dopamine_PINN.py

# ──────────────────────────────────────────────────────────────
# Sensitivity sweeps → figures/sensitivity.json
# ──────────────────────────────────────────────────────────────
sensitivity sweep: $(SENS)

$(SENS):
	$(DDE_ENV) $(PYTHON) dopamine_PINN.py --sweep all

# ──────────────────────────────────────────────────────────────
# Manuscript auto-fill
# ──────────────────────────────────────────────────────────────
preview:
	$(PYTHON) validate_metrics.py

validate:
	$(PYTHON) validate_metrics.py --apply --mirror

restore:
	$(PYTHON) validate_metrics.py --restore

# ──────────────────────────────────────────────────────────────
# LaTeX compile (pdflatex + bibtex + pdflatex x2)
# ──────────────────────────────────────────────────────────────
pdf:
	pdflatex -interaction=nonstopmode $(TEX).tex
	bibtex   $(TEX)
	pdflatex -interaction=nonstopmode $(TEX).tex
	pdflatex -interaction=nonstopmode $(TEX).tex
	@echo ""
	@echo "  ✓ PDF written to $(TEX).pdf"

# Open the compiled PDF (macOS-specific; the `open` command).
# On Linux replace with `xdg-open`; on Windows with `start`.
open:
	@open $(TEX).pdf 2>/dev/null || \
	 (echo "PDF not yet built. Run \`make pdf\` first." && exit 1)

# ──────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────
# Remove only LaTeX auxiliary files (preserve PDF + figures).
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot \
	      *.fdb_latexmk *.fls *.synctex.gz

# Full clean: also drop the PDF, the .tex backup, and the
# generated figures (forces a fresh experimental run on next make).
distclean: clean
	rm -f $(TEX).pdf $(TEX).tex.bak
	rm -rf $(FIG_DIR)
