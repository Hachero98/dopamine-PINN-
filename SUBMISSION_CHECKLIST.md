# PLOS Computational Biology — Submission Checklist

Companion to `B3_Dopamine_PINN_Paper.tex`. Walks through everything
that has to be done from "manuscript looks finished" to "submitted
in Editorial Manager". Estimated total effort: **1 week of part-time
work plus a 2-3 hour submission session**.

Submission portal:
<https://www.editorialmanager.com/ploscompbiol/>

Official author guidelines:
<https://journals.plos.org/ploscompbiol/s/submission-guidelines>

---

## Phase 1 — One week before submission

### A. Run the full experimental pipeline (one afternoon)

- [ ] `make install` — confirm dependencies install cleanly
- [ ] `make quick-all` — verify the pipeline end-to-end (~3 min)
- [ ] `make all` — full real run; ~50 min on Colab T4 GPU
- [ ] Inspect `figures/metrics.json` — sanity-check the numbers
- [ ] Inspect `figures/sensitivity.json` — sanity-check the sweeps
- [ ] `make pdf` — compile the manuscript with real numbers
- [ ] Open `B3_Dopamine_PINN_Paper.pdf` and scan it cover-to-cover

### B. Close the 5 remaining `\TODO` items

- [ ] **Line 739** — replace the loss-weight reporting TODO with
      the actual unit weights you used (`\lambda_r = \lambda_i =
      \lambda_b = \lambda_d = 1`)
- [ ] **Line 1391** — fill in PhD advisor name in Acknowledgments
- [ ] **Line 1399** — write the Funding statement
      (use exactly *"The author(s) received no specific funding
      for this work."* if there's no funding — PLOS rejects
      manuscripts that omit this)
- [ ] **Line 1408** — replace Data Availability placeholder URL
      with the real GitHub repo URL once pushed (see step C)
- [ ] **Line 1447** — optional CRediT "Supervision: [name]" if
      your advisor co-authored

### C. Create the public artifact (one afternoon)

- [ ] Create GitHub repo at `github.com/<your-username>/dopamine-PINN`
- [ ] Push: `B3_Dopamine_PINN_Paper.tex`, `refs.bib`,
      `dopamine_PINN.py`, `dopamine_PINN.ipynb`,
      `validate_metrics.py`, `Makefile`, `requirements.txt`,
      `LICENSE`, `README.md`
- [ ] Do **not** push: `figures/*.png` (regenerate from source)
      or `*.tex.bak`
- [ ] Tag a release: `git tag v1.0-submission && git push --tags`
- [ ] Link GitHub to Zenodo (zenodo.org → log in via GitHub →
      enable the repo)
- [ ] Trigger a Zenodo deposit by creating a release on GitHub;
      Zenodo will mint a DOI like `10.5281/zenodo.XXXXXX`
- [ ] Paste the Zenodo DOI into the Data Availability section

### D. Prepare figures for submission

PLOS requires figures as **separate TIFF or EPS files**, not embedded
in the PDF. Each figure has its own file named `Fig1.tif`, `Fig2.tif`,
`Fig3.tif`.

- [ ] Convert the three figures produced by `dopamine_PINN.py`:
   - [ ] `figures/forward_snapshots.png` → `Fig1.tif`
   - [ ] `figures/forward_heatmap.png` → `Fig2.tif`
   - [ ] `figures/inverse_convergence.png` → `Fig3.tif`
- [ ] Target resolution: **300 DPI** minimum, **600 DPI** preferred
- [ ] Quick converter (ImageMagick):
      `convert -density 600 figures/forward_snapshots.png Fig1.tif`
- [ ] Verify each TIFF opens without errors and fonts are readable

### E. Draft the cover letter (one evening)

PLOS strongly recommends a 1-page cover letter. Template content:

- [ ] Date and editor's name (current PLOS Comp Bio EIC: check website)
- [ ] Manuscript title
- [ ] One-paragraph summary of what the paper does and why it's novel
- [ ] One-paragraph justification for why PLOS Comp Bio specifically
      (mention Yazdani 2020 + Wiencke 2020 as direct topical precedent)
- [ ] One sentence: "This work has not been submitted elsewhere"
- [ ] One sentence on AI tool use (mirroring the manuscript disclosure)
- [ ] Signature with affiliation

### F. Identify 3-5 suggested reviewers

PLOS Comp Bio asks for suggested reviewers. Pick people who:
- Work on PINNs or scientific machine learning
- Or work on computational dopamine / FSCV
- Are NOT current/recent collaborators or trainees

Good candidate pools to consider (verify recent activity):

- [ ] Paris Perdikaris (UPenn) — PINN gradient pathologies, Sahli
- [ ] George Em Karniadakis (Brown) — PINN foundational
- [ ] Lu Lu (Yale / UPenn) — DeepXDE author
- [ ] Stephanie Cragg (Oxford) — striatal dopamine
- [ ] R. Mark Wightman (UNC, retired) — FSCV foundational (verify)
- [ ] Paola Antonietti (Politecnico di Milano) — PD reaction-diffusion

For each, you'll need: name, institution, email, brief one-line
justification.

### G. Final manuscript proofread

- [ ] Spell-check the entire `.tex` file
- [ ] Check that `\ref{fig:...}` and `\eqref{eq:...}` all resolve
      (no `??` in the rendered PDF)
- [ ] Verify Vancouver references render correctly (no `?` for keys)
- [ ] Check that figure captions begin "Fig 1." not "Figure 1:"
- [ ] Verify Author Summary word count (150-200)
- [ ] Verify Abstract word count (≤300)
- [ ] Check no remaining `\TODO{}` or `\DATA{}` placeholders
      (`grep -c '\\TODO{\|\\DATA{' B3_Dopamine_PINN_Paper.tex`)

---

## Phase 2 — Editorial Manager submission day (~2-3 hours)

Set aside an uninterrupted block. The EM session times out after
inactivity; save frequently.

### Step 1 — Create account / log in
- [ ] Go to <https://www.editorialmanager.com/ploscompbiol/>
- [ ] Register as "Author" if first time
- [ ] Verify your ORCID is attached to the EM profile

### Step 2 — Start new submission
- [ ] Click "Submit New Manuscript"
- [ ] Article type: **Research Article**
- [ ] Section: select **Methods** if the paper is methodology-led,
      or **Neuroscience** if you want the biology framing

### Step 3 — Title
- [ ] Paste the full title
- [ ] Short title (≤50 chars) for running header

### Step 4 — Authors
- [ ] Add yourself as corresponding author
- [ ] Affiliation: School of Mathematics and Natural Sciences,
      University of Southern Mississippi
- [ ] ORCID
- [ ] Tick "corresponding author"

### Step 5 — Abstract
- [ ] Paste the 250-300 word Abstract (copy from `.tex`, strip LaTeX)

### Step 6 — Author Summary
- [ ] Paste the 192-word Author Summary (strip LaTeX)

### Step 7 — Keywords
- [ ] Enter the seven keywords from the manuscript:
      Physics-informed neural networks; dopamine diffusion;
      synaptic transmission; Parkinson's disease;
      reaction-diffusion equations; inverse problem; deep learning

### Step 8 — Funding statement
- [ ] Paste your final Funding statement (PLOS requires it even
      if "no specific funding")

### Step 9 — Competing interests
- [ ] Paste: "The author has declared that no competing interests
      exist."

### Step 10 — Data Availability
- [ ] Paste the Data Availability statement from the manuscript,
      with the real GitHub URL and Zenodo DOI now substituted

### Step 11 — Manuscript file
- [ ] Upload the compiled PDF as the **main manuscript file**
- [ ] Also upload the `.tex` source and `refs.bib` as source files
- [ ] (Optional) Upload `dopamine_PINN.py` and notebook as
      Supporting Information

### Step 12 — Figures
- [ ] Upload `Fig1.tif`, `Fig2.tif`, `Fig3.tif` as separate items
- [ ] Each figure: enter the caption text in the EM caption box
      (it can repeat what's in the manuscript)

### Step 13 — Cover letter
- [ ] Upload the cover letter PDF

### Step 14 — Suggested reviewers
- [ ] Enter 3-5 suggested reviewers from step F above

### Step 15 — Opposed reviewers (optional)
- [ ] If anyone has a known conflict, list them with brief reason
- [ ] Otherwise skip

### Step 16 — AI use disclosure (PLOS-specific)
- [ ] Tick "Yes, generative AI was used"
- [ ] Paste the Artificial Intelligence Disclosure section verbatim
      from the manuscript

### Step 17 — Final checks
- [ ] EM generates a preview PDF — open it and skim
- [ ] Verify figures appear in the preview
- [ ] Verify references render
- [ ] Click "Approve Submission"

### Step 18 — Submit
- [ ] Click the final "Submit" button
- [ ] Note the EM tracking number
- [ ] Confirmation email arrives within ~10 minutes

---

## Phase 3 — After submission

### Immediate (within 1 week)
- [ ] Editorial assistant verifies formatting; may request fixes
- [ ] If fixes requested, address within 5 business days
- [ ] Editor-in-Chief assigns to an academic editor (or desk-rejects)

### Review (typical timeline: 6-10 weeks)
- [ ] Track status weekly via EM
- [ ] Possible outcomes: accept (rare), minor revision, major
      revision, reject
- [ ] If revision: PLOS allows ~30-60 days; address every reviewer
      comment in a point-by-point response document

### Post-acceptance
- [ ] PLOS sends proofs ~2 weeks after acceptance
- [ ] Article Processing Charge (APC): currently ~$2,800 USD
      for PLOS Comp Bio; institutional waivers may apply
- [ ] Publication typically ~4-6 weeks after acceptance

---

## Common rejection reasons to preempt

| Reason | How this manuscript guards against it |
|---|---|
| Out of scope for PLOS Comp Bio | Direct topical precedent (Yazdani 2020, Wiencke 2020) cited in the manuscript header and cover letter |
| Insufficient validation | Validated against BOTH analytical solution AND FD reference |
| No reproducibility | GitHub repo + Zenodo DOI + seed-controlled scripts + Makefile |
| Missing PLOS-required sections | Author Summary, CRediT, Data Availability, AI Disclosure all present |
| Author Summary too jargony | Avoids PINN/PDE/L2/loss-function terminology, 192 words |
| Over-claimed clinical relevance | Conditional language ("would be consistent with...") not declarative |
| Wrong reference format | `vancouver.bst` Vancouver-style with DOIs |

---

## Backup plan if PLOS Comp Bio rejects

Without restructuring, the manuscript can be redirected to:

1. **Frontiers in Computational Neuroscience** — same biological framing,
   IF ~3, faster review
2. **Journal of Computational Physics** — reframe as methodology paper
   (drop §2.1 biological background, expand §3 methods)
3. **PLOS One** — broader scope, same author summary requirements
4. **Neural Networks** (Elsevier) — ML methods framing
5. **Computer Methods in Applied Mechanics and Engineering** —
   Haghighat 2021 precedent

Keep the manuscript source under version control so the redirect is a
one-day edit, not a rewrite.

---

## Quick reference — PLOS Comp Bio specifics

| Field | Limit |
|---|---|
| Abstract | ≤300 words |
| Author Summary | 150-200 words |
| Keywords | 3-7 |
| Suggested reviewers | 3-5 |
| Cover letter | ≤2 pages |
| Figure resolution | 300 DPI minimum |
| Figure format | TIFF or EPS |
| Reference style | Vancouver numbered |
| Body word count | Typically <8000, but no hard limit |
| Article Processing Charge | ~$2,800 USD (check current rate) |
| Review timeline | 6-10 weeks typical |
