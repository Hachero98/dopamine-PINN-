#!/usr/bin/env python3
"""
validate_metrics.py — Substitute experimental \DATA{...} placeholders
in B3_Dopamine_PINN_Paper.tex with real numeric values from the
figures/metrics.json file produced by dopamine_PINN.py (or .ipynb).

DESIGN PRINCIPLES
-----------------
1. Safe by default — runs as a dry-run unless --apply is given.
2. Reversible — --apply writes a .bak backup; --restore reverts.
3. Context-aware — placeholders like \\DATA{[Y%]} mean different things
   in the abstract vs. the conclusion of your draft. Each substitution
   uses surrounding text as an anchor so the right value lands in the
   right place.
4. Idempotent — re-running after --apply finds no more matches and
   reports them as already substituted.
5. Mirrored — --mirror copies the final result to the Reference Papers
   duplicate so both .tex copies stay in sync.

USAGE
-----
  python3 validate_metrics.py                  # dry run — show what would change
  python3 validate_metrics.py --apply          # commit, with backup
  python3 validate_metrics.py --apply --mirror # commit + sync to Reference Papers
  python3 validate_metrics.py --restore        # revert from .bak

PLACEHOLDERS HANDLED
--------------------
Abstract:
  • L2 relative error of \\DATA{[X%]} compared with the analytical solution
  • \\DATA{[Y%]} compared with a finite-difference reference solver
  • $D =$ \\DATA{[value]} ...
  • $k =$ \\DATA{[value]} ...
  • relative errors of \\DATA{[Z%]} and \\DATA{[W%]}, respectively

Conclusion:
  • L2 relative error of \\DATA{[X%]} against the analytical solution
  • relative errors below \\DATA{[Y%]} (max of D and k recovery errors)
  • even at \\DATA{[Z%]} observational noise
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Callable

BASE = Path(__file__).resolve().parent
DEFAULT_TARGET      = BASE / "B3_Dopamine_PINN_Paper.tex"
MIRROR_TARGET       = BASE / "Reference Papers" / "B3_Dopamine_PINN_Paper.tex"
DEFAULT_METRICS     = BASE / "figures" / "metrics.json"
DEFAULT_SENSITIVITY = BASE / "figures" / "sensitivity.json"


# =============================================================
# Formatters
# =============================================================
def fmt_pct(x: float) -> str:
    return f"{x:.2f}\\%"

def fmt_D(x: float) -> str:
    return f"{x:.4f}"

def fmt_k(x: float) -> str:
    return f"{x:.4f}"


# =============================================================
# Substitution table
# =============================================================
@dataclass
class Substitution:
    label:       str
    pattern:     str        # regex (raw)
    replacement: str        # plain string (no regex groups)
    description: str


def build_substitutions(metrics: dict) -> list[Substitution]:
    """Return only the substitutions for which we have JSON data.

    Each pattern is anchored by enough surrounding text to be unique
    in the manuscript — preventing the abstract and conclusion from
    being merged by accident.
    """
    subs: list[Substitution] = []
    fwd = metrics.get("forward", {})
    inv = metrics.get("inverse", {})

    # ---- L2 vs analytical (used in BOTH abstract and conclusion) ----
    if "L2_vs_analytical_pct" in fwd:
        val = fmt_pct(fwd["L2_vs_analytical_pct"])
        # Abstract: "L2 relative error of \DATA{[X\%]} compared with the analytical solution"
        subs.append(Substitution(
            label="L2 vs analytical (abstract)",
            pattern=r"L2\s+relative\s+error\s+of\s+\\DATA\{\[X\\%\]\}\s+compared\s+with\s+the\s+analytical",
            replacement=f"L2 relative error of {val} compared with the analytical",
            description=f"abstract X% → {val}",
        ))
        # Conclusion: "an L2 relative error of \DATA{[X\%]} against the analytical"
        subs.append(Substitution(
            label="L2 vs analytical (conclusion)",
            pattern=r"L2\s+relative\s+error\s+of\s+\\DATA\{\[X\\%\]\}\s+against\s+the\s+analytical",
            replacement=f"L2 relative error of {val} against the analytical",
            description=f"conclusion X% → {val}",
        ))

    # ---- L2 vs FD solver (abstract only) ----
    if "L2_vs_fd_pct" in fwd:
        val = fmt_pct(fwd["L2_vs_fd_pct"])
        subs.append(Substitution(
            label="L2 vs FD (abstract)",
            pattern=r"\\DATA\{\[Y\\%\]\}\s+compared\s+with\s+a\s+finite-difference",
            replacement=f"{val} compared with a finite-difference",
            description=f"abstract Y% → {val}",
        ))

    # ---- D recovered (abstract) ----
    if "D_recovered" in inv:
        val = fmt_D(inv["D_recovered"])
        subs.append(Substitution(
            label="D recovered (abstract)",
            pattern=r"\$D\s*=\s*\$\s+\\DATA\{\[value\]\}",
            replacement=f"$D =$ {val}",
            description=f"D → {val} µm²/ms",
        ))

    # ---- k recovered (abstract) ----
    if "k_recovered" in inv:
        val = fmt_k(inv["k_recovered"])
        subs.append(Substitution(
            label="k recovered (abstract)",
            pattern=r"\$k\s*=\s*\$\s+\\DATA\{\[value\]\}",
            replacement=f"$k =$ {val}",
            description=f"k → {val} 1/ms",
        ))

    # ---- D and k recovery errors (abstract) ----
    if "D_rel_error_pct" in inv and "k_rel_error_pct" in inv:
        z, w = fmt_pct(inv["D_rel_error_pct"]), fmt_pct(inv["k_rel_error_pct"])
        subs.append(Substitution(
            label="D/k recovery errors (abstract)",
            pattern=r"relative\s+errors\s+of\s+\\DATA\{\[Z\\%\]\}\s+and\s+\\DATA\{\[W\\%\]\}",
            replacement=f"relative errors of {z} and {w}",
            description=f"Z%, W% → {z}, {w}",
        ))

    # ---- Max recovery error (conclusion) ----
    if "D_rel_error_pct" in inv and "k_rel_error_pct" in inv:
        max_err = max(inv["D_rel_error_pct"], inv["k_rel_error_pct"])
        val = fmt_pct(max_err)
        # Loosened whitespace to handle the LaTeX source's line break
        # between "errors" and "below".
        subs.append(Substitution(
            label="Max recovery error (conclusion)",
            pattern=r"with relative\s+errors\s+below\s+\\DATA\{\[Y\\%\]\}",
            replacement=f"with relative errors below {val}",
            description=f"conclusion Y% (max of D, k errors) → {val}",
        ))

    # ---- Observational noise (conclusion) ----
    if "noise_pct" in inv:
        val = fmt_pct(inv["noise_pct"])
        subs.append(Substitution(
            label="Observational noise (conclusion)",
            pattern=r"even\s+at\s+\\DATA\{\[Z\\%\]\}\s+observational\s+noise",
            replacement=f"even at {val} observational noise",
            description=f"conclusion Z% → {val}",
        ))

    return subs


# =============================================================
# Keyed substitutions for table cells
# =============================================================
#
# Tables 2, 3, 4, 5 use named placeholders like \DATA{coll_1000_L2}
# whose values come from sensitivity.json (for sweeps) or from the
# "per_snapshot" subkey of metrics.json (for the per-snapshot
# forward-error table). These builders translate the JSON into a
# {key: replacement_string} dict that apply_keyed() consumes.
# =============================================================
def _fmt_int_noise(pct: float) -> str:
    """Format a noise percentage for use in a key (so 5.0 → '5')."""
    return f"{int(pct)}" if float(pct).is_integer() else f"{pct}".replace(".", "_")


def _fmt_thousands_math(n: int) -> str:
    """Format an integer in LaTeX math with thin-space thousands
    separator: 10000 → '$10{,}000$'."""
    if n < 1000:
        return f"${n}$"
    return f"${n:,}$".replace(",", "{,}")


def _spell_depth(n_h: int) -> str:
    """Spell out a small depth count for natural prose:
    4 → 'four layers'. Falls back to '$n$ layers' for unknown sizes."""
    words = {2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight"}
    return f"{words[n_h]} layers" if n_h in words else f"${n_h}$ layers"


def _saturation_index(entries: list[dict], key: str,
                      tolerance: float = 1.2) -> int:
    """Return the index of the first entry whose value at `key` is
    within `tolerance × best_value`. This is the 'elbow' point of
    a monotonically-decreasing error curve."""
    if not entries:
        return -1
    best = min(e[key] for e in entries)
    for i, e in enumerate(entries):
        if e[key] <= tolerance * best:
            return i
    return len(entries) - 1


def build_per_snapshot_keys(metrics: dict) -> dict[str, str]:
    """Forward-problem per-snapshot errors → Table 2 cells."""
    out = {}
    snap = metrics.get("forward", {}).get("per_snapshot", {})
    for key, vals in snap.items():           # key like "t_1", "t_5"
        t = key.split("_", 1)[1]
        out[f"fwd_t{t}_anal"] = fmt_pct(vals["L2_anal_pct"])
        out[f"fwd_t{t}_fd"]   = fmt_pct(vals["L2_fd_pct"])
    return out


def build_sensitivity_keys(sens: dict) -> dict[str, str]:
    """Sensitivity sweep results → Tables 3, 4, 5 cells."""
    out = {}

    # Table 4 — collocation density
    for entry in sens.get("collocation", []):
        nr = entry["N_r"]
        out[f"coll_{nr}_L2"] = fmt_pct(entry["L2_pct"])
        out[f"coll_{nr}_t"]  = f"{entry['train_min']:.1f}"

    # Table 5 — network depth
    for entry in sens.get("depth", []):
        out[f"depth_{entry['n_h']}_L2"] = fmt_pct(entry["L2_pct"])

    # Table 3 — observational noise (inverse)
    for entry in sens.get("noise", []):
        k = _fmt_int_noise(entry["noise_pct"])
        out[f"noise_{k}_D"]     = fmt_D(entry["D_recovered"])
        out[f"noise_{k}_D_err"] = fmt_pct(entry["D_rel_error_pct"])
        out[f"noise_{k}_k"]     = fmt_k(entry["k_recovered"])
        out[f"noise_{k}_k_err"] = fmt_pct(entry["k_rel_error_pct"])

    # ── Qualitative narrative markers in §5.3 prose ───────────
    # Auto-derived from the same sweep data: elbow points and
    # worst-case error. The keys include literal brackets and
    # underscores so they match \DATA{[N_r_sat]} etc. verbatim
    # via re.escape inside apply_keyed().

    coll = sens.get("collocation", [])
    if coll:
        idx = _saturation_index(coll, "L2_pct")
        out["[N_r_sat]"] = _fmt_thousands_math(coll[idx]["N_r"])

    depth = sens.get("depth", [])
    if depth:
        idx = _saturation_index(depth, "L2_pct")
        out["[depth_sat]"] = _spell_depth(depth[idx]["n_h"])

    noise = sens.get("noise", [])
    if noise:
        max_err = max(
            max(e["D_rel_error_pct"], e["k_rel_error_pct"]) for e in noise
        )
        out["[noise_max_err]"] = fmt_pct(max_err)

    return out


def apply_keyed(text: str, keys: dict[str, str]) -> tuple[str, int]:
    """Substitute every \\DATA{key} → value for (key, value) in keys."""
    total = 0
    for key, value in keys.items():
        pattern = r"\\DATA\{" + re.escape(key) + r"\}"
        text, n = re.subn(pattern, lambda m, r=value: r, text)
        total += n
    return text, total


# =============================================================
# Apply / report
# =============================================================
def apply(text: str, subs: list[Substitution]) -> tuple[str, dict[str, int]]:
    """Apply each substitution in order. Returns (new_text, count_per_label)."""
    counts: dict[str, int] = {}
    new = text
    for s in subs:
        new, n = re.subn(s.pattern, lambda m, r=s.replacement: r, new, flags=re.MULTILINE)
        counts[s.label] = n
    return new, counts


def count_remaining_DATA(text: str) -> int:
    """How many \\DATA{...} placeholders are still in the document."""
    return len(re.findall(r"\\DATA\{[^}]*\}", text))


def diff_preview(before: str, after: str, max_lines: int = 60) -> list[str]:
    return list(unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile="before", tofile="after", lineterm="", n=1,
    ))[:max_lines]


# =============================================================
# Subcommands
# =============================================================
def cmd_restore(target: Path) -> int:
    bak = target.with_suffix(".tex.bak")
    if not bak.exists():
        print(f"  No backup found at {bak}", file=sys.stderr)
        return 1
    shutil.copy2(bak, target)
    print(f"  Restored {target.name} from {bak.name}")
    return 0


def cmd_run(target: Path, metrics_path: Path, sensitivity_path: Path,
            apply_changes: bool, mirror: bool) -> int:
    if not metrics_path.exists() and not sensitivity_path.exists():
        print(f"  No metrics.json at {metrics_path}", file=sys.stderr)
        print(f"  No sensitivity.json at {sensitivity_path}", file=sys.stderr)
        print(f"  Run dopamine_PINN.py first (full / --quick / --sweep) "
              f"to produce one or both.", file=sys.stderr)
        return 1
    if not target.exists():
        print(f"  Target .tex not found: {target}", file=sys.stderr)
        return 1

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    sens    = json.loads(sensitivity_path.read_text()) if sensitivity_path.exists() else {}

    subs = build_substitutions(metrics) if metrics else []
    keyed_metrics = build_per_snapshot_keys(metrics) if metrics else {}
    keyed_sens    = build_sensitivity_keys(sens)     if sens    else {}

    if not subs and not keyed_metrics and not keyed_sens:
        print("  No applicable content found in JSON. Nothing to substitute.")
        return 1

    original = target.read_text()
    new, counts = apply(original, subs)
    new, n_per_snap = apply_keyed(new, keyed_metrics)
    new, n_sens     = apply_keyed(new, keyed_sens)

    # Report
    print(f"\n  Target:      {target}")
    print(f"  Metrics:     {metrics_path}" +
          ("" if metrics else "  (none)"))
    print(f"  Sensitivity: {sensitivity_path}" +
          ("" if sens else "  (none)"))
    print()
    print(f"  {'Label':<45}{'Hits':>6}   Detail")
    print(f"  {'-' * 45}{'-' * 6}   {'-' * 30}")
    # Expected hit counts. Most substitutions appear in multiple
    # places now that Results §5 and Discussion both reference the
    # headline numbers. These counts reflect the canonical
    # manuscript layout; bump them up if you add new sections that
    # reuse the same phrasing.
    expected = {
        "L2 vs analytical (abstract)":    2,  # abstract + §5.1
        "L2 vs FD (abstract)":            2,  # abstract + §5.1
        "D recovered (abstract)":         3,  # abstract + §5.2 + Discussion
        "k recovered (abstract)":         3,  # abstract + §5.2 + Discussion
        "D/k recovery errors (abstract)": 2,  # abstract + §5.2
    }

    total_hits = 0
    for s in subs:
        n   = counts[s.label]
        exp = expected.get(s.label, 1)
        if n == 0:
            marker, note = "·", " (not found)"
        elif n == exp:
            marker, note = "✓", ""
        else:
            marker, note = "!", f" (expected {exp} hit{'s' if exp != 1 else ''})"
        print(f"  {marker} {s.label:<43}{n:>6}   {s.description}{note}")
        total_hits += n

    if n_per_snap > 0:
        print(f"  ✓ {'Per-snapshot table cells':<43}{n_per_snap:>6}   "
              f"{len(keyed_metrics)} cells from metrics.per_snapshot")
        total_hits += n_per_snap
    if n_sens > 0:
        print(f"  ✓ {'Sensitivity table cells':<43}{n_sens:>6}   "
              f"{len(keyed_sens)} cells from sensitivity.json")
        total_hits += n_sens

    remaining_before = count_remaining_DATA(original)
    remaining_after  = count_remaining_DATA(new)
    print(f"\n  \\DATA{{...}} placeholders: {remaining_before} → {remaining_after}")
    print(f"  Total substitutions:    {total_hits}")

    if not apply_changes:
        print("\n  [DRY RUN] No file written. Re-run with --apply to commit.\n")
        diff = diff_preview(original, new)
        if diff:
            print("  Preview (first 60 diff lines):")
            for line in diff:
                print("    " + line)
        return 0

    # Commit
    bak = target.with_suffix(".tex.bak")
    shutil.copy2(target, bak)
    target.write_text(new)
    print(f"\n  Wrote {target.name} (backup at {bak.name}).")

    if mirror:
        if target == MIRROR_TARGET:
            other = DEFAULT_TARGET
        else:
            other = MIRROR_TARGET
        if other.exists():
            shutil.copy2(target, other)
            print(f"  Mirrored to {other}.")
        else:
            print(f"  Skipped mirror (no file at {other}).", file=sys.stderr)
    return 0


# =============================================================
# CLI
# =============================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Substitute \\DATA{...} placeholders in B3.tex using "
                    "metrics from figures/metrics.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--apply", action="store_true",
                        help="Write the changes (default: dry run)")
    parser.add_argument("--restore", action="store_true",
                        help="Restore the .tex file from its .bak backup")
    parser.add_argument("--mirror", action="store_true",
                        help="After --apply, also copy to the Reference "
                             "Papers duplicate (kept in sync with CV root)")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                        help=f"Path to B3.tex (default: {DEFAULT_TARGET})")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS,
                        help=f"Path to metrics.json (default: {DEFAULT_METRICS})")
    parser.add_argument("--sensitivity", type=Path, default=DEFAULT_SENSITIVITY,
                        help=f"Path to sensitivity.json from "
                             f"dopamine_PINN.py --sweep "
                             f"(default: {DEFAULT_SENSITIVITY})")
    args = parser.parse_args()

    if args.restore and args.apply:
        print("  --restore and --apply are mutually exclusive.", file=sys.stderr)
        return 2

    if args.restore:
        return cmd_restore(args.target)
    return cmd_run(args.target, args.metrics, args.sensitivity,
                   args.apply, args.mirror)


if __name__ == "__main__":
    sys.exit(main())
