# Medical & Biological Engineering & Computing (MBEC) — Submission Checklist

Companion to `B3_Dopamine_PINN_Paper.tex`, which was originally written
and formatted for **PLOS Computational Biology** (see
`SUBMISSION_CHECKLIST.md`). This document is the migration plan for
retargeting the same manuscript to **Medical & Biological Engineering &
Computing** (Springer, journal 11517), official journal of the
International Federation of Medical and Biological Engineering (IFMBE).

Submission portal: <https://www.editorialmanager.com/MBEC>
Author guidelines: <https://link.springer.com/journal/11517/submission-guidelines>
Aims and scope: <https://link.springer.com/journal/11517/aims-and-scope>

---

## 0. Fit and risk assessment — read this first

MBEC's submission guidelines state manuscripts are **rejected without
review** if they: lack clinical relevance, merely apply existing
methods to public/synthetic data without methodological innovation, or
show poor writing/data presentation. This is a materially different
bar than PLOS Comp Bio, where the current abstract deliberately uses
*conditional* language about clinical relevance ("would be consistent
with...") to avoid over-claiming.

For MBEC, under-claiming clinical relevance is itself a desk-reject
risk. The cover letter and abstract need to foreground:
- the Parkinson's disease motivation more assertively (still
  evidence-based, not overstated — MBEC wants a stated *path* to
  clinical/experimental relevance, not a discovery already made),
- that no biological data is used yet (state this plainly, don't bury
  it) and be explicit that the immediate next step (stated in the
  current abstract) is application to real fast-scan cyclic
  voltammetry (FSCV) data,
- the methodological novelty (identifiability-aware inverse-PINN
  framework) as a deliverable other groups can use on their own
  neurotransmitter transport data — this is the "practical relevance"
  MBEC screens for, distinct from PLOS Comp Bio's tolerance for
  purely computational/methods contributions.

**Recommendation:** keep the science identical, rewrite the framing
(abstract, intro hook, cover letter, and conclusion's "next steps"
paragraph) to lead with the clinical/translational angle before the
methodological one. Do this before investing in reformatting.

**Article type:** submit as an **Original Research Article** (max 18
pages) — not Brief Report or Perspective, given the scope of validation
work already done.

---

## 1. Structural gap analysis (current manuscript → MBEC requirements)

| Element | Current state (PLOS-formatted) | MBEC requirement | Action |
|---|---|---|---|
| Length | 45 pages (`article`, 12pt, single column) | **18 pages max**, 11pt Times or 10pt Arial | Cut ~55-60%. See §2. |
| Abstract | 249 words, unstructured | **≤200 words**, unstructured | Trim ~50 words |
| Author Summary | Present (`\section*{Author Summary}`, plain-language, 192 words) | **Not an MBEC section — delete** | Remove; fold any essential plain-language framing into cover letter instead |
| Keywords | 7 keywords | **Up to 5**, preferably MeSH terms | Cut to 5, check MeSH database for closest matches |
| Sections | Introduction, Background, Materials & Methods, Inverse Problem, Results, Discussion, Conclusion | Required: **Introduction, Methods, Results, Discussion, Conclusion** | Merge "Background" into Introduction; merge "Inverse Problem" section into Methods (or a Methods subsection) |
| End-of-paper metadata | Separate `\section*{}` blocks: Funding, Competing Interests, Data Availability, Code Availability, AI Disclosure | Single **"Declarations"** section with subheadings (Funding, Conflicts of Interest, Ethics approval, Consent, Data availability, Author contributions) | Consolidate into one Declarations section; add an Author Contributions (CRediT-style) statement — MBEC requires this explicitly, PLOS treats it separately in the portal, not the manuscript body |
| Ethics/consent statements | Absent (not applicable — no human/animal subjects, but statement should still say so explicitly) | Must state N/A explicitly if not applicable | Add one-line "Not applicable" statements for Ethics Approval and Informed Consent |
| Reference style | `vancouver.bst`, numbered by citation order | **Springer Basic (SPBASIC)**: alphabetical by first author surname, then numbered consecutively, cited as `[1]`, `[2-4]` | Swap `.bst`; regenerate bibliography (see §3) |
| DOIs in references | Inconsistent | Full DOI links required (`https://doi.org/...`) on every reference that has one | Audit `refs.bib` for missing DOI fields |
| Document class | `\documentclass[12pt]{article}` | Springer LaTeX macro package for MBEC (based on `svjour3`/`sn-jnl`) | Download the MBEC-specific LaTeX template from the journal's Editorial Manager submission page (linked from the guidelines page — grab this at submission-prep time, template links there are session-specific) |
| Figures | PNG → converted to 600 DPI TIFF (`figures_tiff/`) | Line art: EPS, ≥1200 DPI. Halftone: TIFF, ≥300 DPI. Combination art: ≥600 DPI. RGB, 8 bits/channel. Width must be exactly 39/84/129/174 mm. | Check each figure's category (most PINN plots = "combination" — line + raster); resize to a valid column width; re-export |
| Figure captions | "Fig 1." (PLOS style, plain) | "**Fig.**" bold, followed by number, must define every symbol/element used in the panel | Reformat captions |
| Graphical abstract | **Not present** | **Mandatory.** 32.93mm × 37.63mm at 100%, 300/600/1200 DPI by content type, formats tiff/eps/jpg/bmp/doc/pdf, filename `Graphical_Abstract_Hackman` | New deliverable — see §4 |
| Supplementary material | S1_Text, S2_Text as separate PLOS-style "Supporting Information" PDFs | Called **"Online Resource"**, numbered sequentially, PDF/csv/xlsx/video accepted | Rename S1/S2 → "Online Resource 1/2"; the `\iffalse`-wrapped appendix sections and the still-inline "Posterior Comparison: Laplace vs HMC" section (currently ~95 lines counted toward the 45-page total) should all move here — this alone recovers meaningful page budget |
| Word processor | LaTeX (fine — MBEC accepts LaTeX) | LaTeX or Word **.doc** (not .docx) if going the Word route | Stay on LaTeX; use the MBEC macro package |
| Title page | Not separated from manuscript | Must be a distinct title page with: title, authors/affiliations, corresponding author contact, **total word count, abstract word count, figure count, table count** | Add a title page with these declared counts — new requirement vs. PLOS |

---

## 2. Cutting the manuscript to 18 pages

The 45-page count is with the PLOS-era `\iffalse`-wrapped appendices
already excluded. Real cuts needed:

- [ ] Move "Posterior Comparison: Laplace vs Hamiltonian Monte Carlo"
      (`\label{app:hmc}`, ~95 lines) out of the main body entirely →
      Online Resource. It's currently the only appendix-like section
      *not* commented out, so it's inflating page count for no reason
      it wouldn't also serve in supplementary form.
- [ ] Merge "Background" (`\section{Background}`, line 274) into a
      trimmed Introduction — MBEC's 5-section skeleton doesn't have
      room for a separate literature-survey section at this length.
- [ ] Merge "Inverse Problem: Parameter Recovery" (line 738) into
      Methods as a subsection.
- [ ] Audit Results (line 853–1488, ~635 lines — the single largest
      section) for redundant reporting: noise-sweep, param-grid, and
      obs-density-scaling sweeps are each getting full narrative
      treatment; consider condensing 2 of the 3 to compact table form
      with a one-paragraph summary, keeping full figures in
      supplementary if needed.
- [ ] Discussion (line 1488–1853, ~365 lines) likely has room to
      tighten — check for repetition with Results.
- [ ] After cuts, recompile and check actual MBEC-template page count,
      not the current 12pt `article`-class count (the Springer
      template is typeset tighter — but budget for it, don't assume
      the switch alone solves the overage).

## 3. Reference style migration

- [ ] Obtain `spbasic.bst` (Springer Basic, commonly bundled with
      `natbib`/Springer LaTeX packages, or downloadable from CTAN) —
      confirm it produces alphabetical-then-numbered Springer Basic
      output, not a different Springer variant.
- [ ] Replace `\bibliographystyle{vancouver}` → `\bibliographystyle{spbasic}`
      (exact package name to confirm against the MBEC macro package
      once downloaded — the journal's LaTeX kit usually pins this).
- [ ] Recompile bibliography; spot-check that in-text citations render
      as `[1]`, `[2, 3]`, `[4-6]` rather than the Vancouver
      superscript/bracket style currently used.
- [ ] Add missing DOIs to `refs.bib` entries (grep for entries without
      a `doi = {...}` field).

## 4. New deliverable: graphical abstract

- [ ] Design a single-panel schematic summarizing the paper: PDE
      domain → PINN forward/inverse solve → recovered $D, k$ with
      identifiability caveat. The existing `figures/` concept-figure
      schematic (added in the most recent commit) may already be a
      usable starting point — check whether it fits the mandatory
      32.93mm × 37.63mm aspect ratio or needs a redesign at that
      aspect ratio specifically (it's nearly square, not
      landscape — most existing figures likely aren't).
- [ ] Export at the required resolution/format, name it
      `Graphical_Abstract_Hackman.tif` (or `.pdf`/`.eps`).

## 5. Cover letter rewrite

The existing `cover_letter.md` is PLOS Comp Bio-specific (references
Yazdani 2020 + Wiencke 2020 as topical precedent for that journal).
For MBEC:

- [ ] Rewrite the justification paragraph around MBEC's scope
      (biomedical modeling and simulation, AI-based biomedicine, neural
      engineering) and IFMBE's engineering-practice audience rather
      than Comp Bio's computational-biology audience.
- [ ] Lead with clinical/translational motivation per §0.
- [ ] Update editor name/EIC (verify current MBEC Editor-in-Chief —
      distinct from PLOS Comp Bio's).
- [ ] Keep the "not submitted elsewhere" and AI-disclosure sentences —
      both still apply.

## 6. Suggested reviewers

The PLOS reviewer pool (Perdikaris, Karniadakis, Lu, Cragg, Wightman,
Antonietti) skews toward PINN-methods and dopamine-neuroscience
specialists. MBEC's audience is more engineering-practice-oriented —
consider adding 1-2 names from biomedical signal processing / neural
engineering to better match reviewer-editor fit, keeping the strongest
2-3 PINN/dopamine names from the existing list.

- [ ] Re-verify all reviewer emails/affiliations are current (this
      list predates today; people move).

## 7. Reference papers / precedent check

- [ ] Before finalizing the intro's journal-fit justification,
      spot-check whether MBEC has published any PINN or physics-informed
      ML papers in the last 2-3 years (search Scopus/Google Scholar:
      `site:link.springer.com/journal/11517 physics-informed`) — a
      direct precedent (like Yazdani 2020 / Wiencke 2020 served for
      PLOS) strengthens the fit argument in both the cover letter and
      the manuscript's own framing.

## 8. Mechanical steps (once content work above is done)

- [ ] Download the actual MBEC LaTeX macro package from the
      submission page in Editorial Manager (requires starting a
      submission or checking the journal's "Instructions for Authors"
      PDF — the template download link isn't stable enough to hardcode
      here).
- [ ] Port `B3_Dopamine_PINN_Paper.tex` content into the new
      document class.
- [ ] Regenerate all figures at MBEC's width/DPI/format matrix (§1).
- [ ] Rebuild `refs.bib` citations against `spbasic.bst`.
- [ ] Produce the title page as a separate front-matter section per
      MBEC's declared-counts requirement.
- [ ] Consolidate the five separate end-of-manuscript statements into
      one "Declarations" section with an added Author Contributions
      statement.
- [ ] Produce the graphical abstract file.
- [ ] Rename S1_Text/S2_Text → Online Resource 1/2, add the HMC
      appendix as Online Resource 3.
- [ ] New cover letter.
- [ ] Register/log in at Editorial Manager MBEC portal, ORCID linked.

---

## What does NOT need to change

- The science, all numerical results, and all validation work are
  unaffected by the venue switch — this is purely a
  formatting/framing migration.
- JAX code, notebooks, `validate_metrics.py`, Makefile pipeline:
  unaffected.
- MIT license, GitHub/Zenodo reproducibility artifacts: unaffected,
  and MBEC's "type 1 research data policy" (encourages public
  repository deposit) is if anything a better fit for the existing
  Zenodo-DOI setup than a stricter mandate would be.

---

## Suggested order of operations

1. Resolve §0 (framing decision) before touching LaTeX — it changes
   what gets cut vs. kept in §2.
2. Do the structural merges and page cuts (§2) while still in the
   current document class — easier to edit in familiar formatting
   first.
3. Only then port into the MBEC template (§8) — porting an
   already-trimmed manuscript avoids re-doing template work twice.
4. Reference style swap (§3) and figure re-export (§1/§4) can happen
   in parallel with step 3.
5. Cover letter and reviewer list last, once the paper's final framing
   (from step 1) is locked.
