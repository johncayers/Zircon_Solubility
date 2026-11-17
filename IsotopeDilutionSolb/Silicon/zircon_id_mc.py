#!/usr/bin/env python3
"""
zircon_id_mc.py -- Zircon solubility in H2O by 30Si isotope dilution,
with Monte Carlo error propagation.

The fluid is spiked with enriched 30SiO2. Zircon dissolution adds Si of
natural isotopic composition, lowering the fluid 30Si/28Si. A mass balance
on 30Si and 28Si gives the moles of zircon-derived Si:

    n_x = n_A * (A30_A - R_m * A28_A) / (R_m * A28_nat - A30_nat)

where A = initial solution (spike + any Si blank), R_m = measured 30Si/28Si.
Solubility (congruent dissolution) is m_ZrSiO4 = n_x / W_H2O  [mol/kg].

Outputs, per sample row:
  * endpoint 30Si/28Si estimate (primary)
  * 29Si/28Si cross-check estimate (if R29_28 given)
  * common-ion-corrected pure-water solubility s0 = sqrt(m_Zr * m_Si,total)
  * error magnification factor for R_m
Per experiment with >= 1 rows carrying measured Si (time series):
  * precipitation-corrected Si released, by integrating
        dSi_B = integral_{f_A}^{f_M} n(f) / (f_B - f) df
    over the 28Si atom fraction f, evaluated as int n du with
    u = -ln(f_B - f) (trapezoidal rule, t = 0 point included).
    Without precipitation this equals the endpoint result; with
    precipitation during dissolution the endpoint result is an upper bound.

Usage:
    python zircon_id_mc.py zircon_id_input_template.csv
    python zircon_id_mc.py input.csv -n 200000 --seed 1 -o results --plots

Uncertainties in the input file are 1-sigma absolute and are sampled as
independent normal distributions. Parameters shared by an experiment
(water mass, spike mass and composition, blank) are drawn once per
experiment so they are correctly correlated across its time-series rows.

J. C. Ayers, Vanderbilt University. Requires numpy and pandas
(matplotlib only for --plots).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (edit here if you want different reference values)
# ---------------------------------------------------------------------------
# IUPAC representative isotopic composition of Si (atom fractions, 1 s.d.
# taken as the tabulated uncertainty). A28 is computed as 1 - A29 - A30.
NAT_A29, NAT_A29_SD = 0.04685, 0.00008
NAT_A30, NAT_A30_SD = 0.03092, 0.00011

# Atomic masses (g/mol)
M_28SI = 27.9769265
M_29SI = 28.9764947
M_30SI = 29.9737701
M_O = 15.9994
M_ZR = 91.224
M_ZRSIO4 = 183.307

# ---------------------------------------------------------------------------
# Input columns
# ---------------------------------------------------------------------------
REQUIRED = ["experiment_id", "sample_id", "W_H2O_mg", "spike_A28", "spike_A29",
            "R30_28"]
# (value, sd) pairs; sd columns default to 0 if missing/blank
PAIRS = {
    "W_H2O_mg": "W_H2O_mg_sd",
    "spike_SiO2_ug": "spike_SiO2_ug_sd",
    "spike_Si_mmolal": "spike_Si_mmolal_sd",
    "spike_A28": "spike_A28_sd",
    "spike_A29": "spike_A29_sd",
    "R30_28": "R30_28_sd",
    "R29_28": "R29_28_sd",
    "Si_meas_mmolal": "Si_meas_mmolal_sd",
    "Zr_meas_mmolal": "Zr_meas_mmolal_sd",
    "blank_Si_nmol": "blank_Si_nmol_sd",
}
SHARED = ["W_H2O_mg", "spike_SiO2_ug", "spike_Si_mmolal", "spike_A28",
          "spike_A29", "blank_Si_nmol"]

COLUMN_DOC = {
    "experiment_id": "Run/capsule identifier. Rows sharing it form a time series.",
    "sample_id": "Label for this fluid analysis. Need not be unique on its own: "
                 "results and histograms are keyed on experiment_id + sample_id.",
    "time_h": "Run duration or sampling time (h). Orders a time series.",
    "W_H2O_mg": "Mass of H2O in the charge (mg).",
    "spike_SiO2_ug": "Mass of 30SiO2 spike loaded (ug). Give this OR spike_Si_mmolal.",
    "spike_Si_mmolal": "Spike Si molality if added as a stock solution (mmol/kg H2O).",
    "spike_A28": "28Si atom fraction of spike (certificate or measured).",
    "spike_A29": "29Si atom fraction of spike. A30 = 1 - A28 - A29.",
    "R30_28": "Measured 30Si/28Si of the fluid, mass-bias and interference corrected.",
    "R29_28": "Optional measured 29Si/28Si of the fluid (cross-check).",
    "Si_meas_mmolal": "Optional measured total dissolved Si (mmol/kg). Needed for the "
                      "precipitation-corrected time-series integral.",
    "Zr_meas_mmolal": "Optional measured dissolved Zr (mmol/kg), congruency check.",
    "blank_Si_nmol": "Optional procedural Si blank of natural composition (nmol).",
    "notes": "Free text; ignored.",
}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def make_labels(exp_ids, sample_ids) -> list[str]:
    """Unique label per row: sample_id if it already starts with the
    experiment_id (e.g. Zir026 / Zir026_03 -> Zir026_03), otherwise
    experiment_id_sample_id (e.g. Zir026_S / Surface -> Zir026_S_Surface).
    Any remaining duplicates get a numeric suffix."""
    labels, seen = [], {}
    for e, s in zip(exp_ids, sample_ids):
        lab = s if s.startswith(e) else f"{e}_{s}"
        k = seen.get(lab, 0)
        seen[lab] = k + 1
        labels.append(lab if k == 0 else f"{lab}_{k + 1}")
    return labels


def safe_filename(name: str) -> str:
    """Replace characters that are not safe in Windows/macOS file names."""
    import re
    return re.sub(r'[^A-Za-z0-9._-]+', "_", str(name)).strip("_") or "sample"


def read_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Input is missing required columns: {missing}")
    df = df.dropna(subset=["experiment_id", "R30_28"]).copy()
    for v, s in PAIRS.items():
        df[v] = _num(df, v)
        df[s] = _num(df, s).fillna(0.0)
    df["time_h"] = _num(df, "time_h")
    df["experiment_id"] = df["experiment_id"].astype(str).str.strip()
    sid = df["sample_id"].astype("string").str.strip()
    df["sample_id"] = sid.fillna(df["experiment_id"]).replace("", pd.NA).fillna(
        df["experiment_id"]).astype(str)
    df["label"] = make_labels(df["experiment_id"], df["sample_id"])
    if "blank_Si_nmol" in df:
        df["blank_Si_nmol"] = df["blank_Si_nmol"].fillna(0.0)
    bad = df[df["spike_SiO2_ug"].isna() & df["spike_Si_mmolal"].isna()]
    if len(bad):
        sys.exit("Each row needs spike_SiO2_ug or spike_Si_mmolal: "
                 f"{list(bad['experiment_id'])}")
    return df


# ---------------------------------------------------------------------------
# Core equations (vectorised; all arguments may be arrays)
# ---------------------------------------------------------------------------
def id_moles(n_A, A_num_A, A28_A, R_m, A_num_B, A28_B):
    """Moles of end-member B (natural Si) in a two-component mix with A.
    Ratio R_m = (numerator isotope)/28Si of the mixture."""
    return n_A * (A_num_A - R_m * A28_A) / (R_m * A28_B - A_num_B)


def error_magnification(R_m, R_A, R_B):
    """|d ln n_x / d ln R_m| for isotope dilution with ratio R = num/28."""
    return abs(R_m * (R_A - R_B) / ((R_A - R_m) * (R_m - R_B)))


def trapz(y, x):
    """Trapezoidal integral along axis 0 (rows = points, cols = MC draws)."""
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]), axis=0)


def summarise(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=float)
    ok = x[np.isfinite(x)]
    if ok.size == 0:
        return {f"{prefix}_{k}": np.nan for k in
                ["mean", "sd", "median", "p2.5", "p97.5", "frac_le0"]}
    return {
        f"{prefix}_mean": ok.mean(),
        f"{prefix}_sd": ok.std(ddof=1),
        f"{prefix}_median": np.median(ok),
        f"{prefix}_p2.5": np.percentile(ok, 2.5),
        f"{prefix}_p97.5": np.percentile(ok, 97.5),
        f"{prefix}_frac_le0": np.mean(ok <= 0),
    }


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def draw(rng, mean, sd, n):
    if not np.isfinite(mean):
        return np.full(n, np.nan)
    if sd is None or not np.isfinite(sd) or sd <= 0:
        return np.full(n, float(mean))
    return rng.normal(mean, sd, n)


def run(df: pd.DataFrame, n_draws: int, seed: int | None, nat_sd: bool = True):
    rng = np.random.default_rng(seed)
    nat29_sd = NAT_A29_SD if nat_sd else 0.0
    nat30_sd = NAT_A30_SD if nat_sd else 0.0
    row_out, exp_out, draws_store = [], [], {}

    for exp_id, g in df.groupby("experiment_id", sort=False):
        g = g.sort_values("time_h", na_position="last")
        first = g.iloc[0]

        # --- natural (zircon) Si composition, one draw set per experiment ---
        B29 = draw(rng, NAT_A29, nat29_sd, n_draws)
        B30 = draw(rng, NAT_A30, nat30_sd, n_draws)
        B28 = 1.0 - B29 - B30

        # --- shared experiment parameters ---
        W_kg = draw(rng, first["W_H2O_mg"], first["W_H2O_mg_sd"], n_draws) * 1e-6
        S28 = np.clip(draw(rng, first["spike_A28"], first["spike_A28_sd"], n_draws), 0, 1)
        S29 = np.clip(draw(rng, first["spike_A29"], first["spike_A29_sd"], n_draws), 0, 1)
        S30 = 1.0 - S28 - S29
        if np.isfinite(first["spike_SiO2_ug"]):
            M_sp = S28 * M_28SI + S29 * M_29SI + S30 * M_30SI + 2 * M_O
            n_sp = draw(rng, first["spike_SiO2_ug"], first["spike_SiO2_ug_sd"],
                        n_draws) * 1e-6 / M_sp                       # mol
        else:
            n_sp = draw(rng, first["spike_Si_mmolal"], first["spike_Si_mmolal_sd"],
                        n_draws) * 1e-3 * W_kg                       # mol
        n_bl = draw(rng, first["blank_Si_nmol"], first["blank_Si_nmol_sd"],
                    n_draws) * 1e-9                                  # mol

        # initial solution A = spike + blank
        n_A = n_sp + n_bl
        A28 = (n_sp * S28 + n_bl * B28) / n_A
        A29 = (n_sp * S29 + n_bl * B29) / n_A
        A30 = (n_sp * S30 + n_bl * B30) / n_A

        # nominal (no-noise) composition of A, for design diagnostics
        B28_0 = 1 - NAT_A29 - NAT_A30
        S28_0, S29_0 = first["spike_A28"], first["spike_A29"]
        S30_0 = 1 - S28_0 - S29_0
        W0 = first["W_H2O_mg"] * 1e-6
        if np.isfinite(first["spike_SiO2_ug"]):
            nsp0 = first["spike_SiO2_ug"] * 1e-6 / (
                S28_0 * M_28SI + S29_0 * M_29SI + S30_0 * M_30SI + 2 * M_O)
        else:
            nsp0 = first["spike_Si_mmolal"] * 1e-3 * W0
        nbl0 = first["blank_Si_nmol"] * 1e-9
        RA_nom = (nsp0 * S30_0 + nbl0 * NAT_A30) / (nsp0 * S28_0 + nbl0 * B28_0)
        RB_nom = NAT_A30 / B28_0

        ts_f, ts_n = [], []   # for time-series integral (per-row draws)

        for _, r in g.iterrows():
            R30 = draw(rng, r["R30_28"], r["R30_28_sd"], n_draws)
            # natural Si added to A (zircon-derived only, since blank is in A)
            n_x = id_moles(n_A, A30, A28, R30, B30, B28)
            m_x = n_x / W_kg                                           # mol/kg
            m_A = n_A / W_kg

            out = {"label": r["label"], "experiment_id": exp_id,
                   "sample_id": r["sample_id"], "time_h": r["time_h"]}
            out.update(summarise(m_x * 1e3, "mZrSiO4_mmolal"))
            out.update(summarise(m_x * M_ZRSIO4 * 1e3, "ZrSiO4_mg_per_kgH2O"))
            out.update(summarise(m_x * M_ZR * 1e3, "Zr_ug_per_gH2O"))

            # common-ion correction to zero added silica
            if np.isfinite(r["Si_meas_mmolal"]):
                m_Si_tot = draw(rng, r["Si_meas_mmolal"], r["Si_meas_mmolal_sd"],
                                n_draws) * 1e-3
            else:
                m_Si_tot = m_A + m_x
            with np.errstate(invalid="ignore"):
                s0 = np.sqrt(m_x * m_Si_tot)
            out.update(summarise(s0 * 1e3, "s0_pure_water_mmolal"))
            out.update(summarise(m_A * 1e3, "mSi_initial_mmolal"))

            # 29Si/28Si cross-check
            if np.isfinite(r["R29_28"]):
                R29 = draw(rng, r["R29_28"], r["R29_28_sd"], n_draws)
                m_x29 = id_moles(n_A, A29, A28, R29, B29, B28) / W_kg
                out.update(summarise(m_x29 * 1e3, "mZrSiO4_R29_mmolal"))

            # Zr congruency check
            if np.isfinite(r["Zr_meas_mmolal"]):
                zr = draw(rng, r["Zr_meas_mmolal"], r["Zr_meas_mmolal_sd"], n_draws)
                out.update(summarise(zr / (m_x * 1e3), "Zr_meas_over_mZrSiO4"))

            # error magnification at nominal values
            out["error_magnification_R30"] = error_magnification(
                r["R30_28"], RA_nom, RB_nom)
            out["optimal_R30_28"] = math.sqrt(RA_nom * RB_nom)
            row_out.append(out)
            draws_store[r["label"]] = m_x * 1e3

            # how well R_m is resolved from the natural ratio; when it is not,
            # n_x is unbounded (denominator of Eq. 1 -> 0) and the result is
            # meaningless
            out["R30_minus_Rnat_in_sd"] = ((r["R30_28"] - RB_nom) / r["R30_28_sd"]
                                           if r["R30_28_sd"] > 0 else np.inf)

            # store for time-series integral: 28Si fraction of the solution
            if np.isfinite(r["Si_meas_mmolal"]):
                q = n_x / n_A
                f28 = (A28 + q * B28) / (1 + q)
                ts_f.append(f28)
                ts_n.append(m_Si_tot)                 # mol/kg in solution

        # --- precipitation-corrected time-series integral -----------------
        if ts_f:
            f = np.vstack([A28] + ts_f)                # rows: t0, t1, ...
            n = np.vstack([m_A] + ts_n)
            # Substitute u = -ln(f_B - f): df/(f_B - f) = du, so the integral
            # becomes int n du. The trapezoid rule on n(u) is much less
            # biased for sparse sampling than on n/(f_B - f) vs f.
            with np.errstate(divide="ignore", invalid="ignore"):
                u = -np.log(B28 - f)
            dSi = trapz(n, u)                          # mol/kg released
            e = {"experiment_id": exp_id, "n_points": len(ts_f),
                 "last_sample_id": g["sample_id"].iloc[-1],
                 "last_label": g["label"].iloc[-1]}
            e.update(summarise(dSi * 1e3, "dSiB_integral_mmolal"))
            e.update(summarise(dSi * M_ZRSIO4 * 1e3, "ZrSiO4_integral_mg_per_kgH2O"))
            exp_out.append(e)

    return pd.DataFrame(row_out), pd.DataFrame(exp_out), draws_store


BUDGET_GROUPS = {
    "water mass": ["W_H2O_mg_sd"],
    "spike amount": ["spike_SiO2_ug_sd", "spike_Si_mmolal_sd"],
    "spike composition": ["spike_A28_sd", "spike_A29_sd"],
    "measured 30Si/28Si": ["R30_28_sd"],
    "measured Si": ["Si_meas_mmolal_sd"],
    "Si blank": ["blank_Si_nmol_sd"],
    "natural Si composition": [],
}


def uncertainty_budget(df, n_draws, seed):
    """Variance share of each input group, by one-group-at-a-time MC."""
    sd_cols = [c for c in PAIRS.values() if c in df.columns]
    res_rows, res_exp = {}, {}
    for grp, cols in BUDGET_GROUPS.items():
        d = df.copy()
        for c in sd_cols:
            if c not in cols:
                d[c] = 0.0
        r, e, _ = run(d, n_draws, seed, nat_sd=(grp == "natural Si composition"))
        res_rows[grp] = r.set_index("label")["mZrSiO4_mmolal_sd"] ** 2
        if len(e):
            res_exp[grp] = e.set_index("experiment_id")["dSiB_integral_mmolal_sd"] ** 2
    rows = pd.DataFrame(res_rows)
    rows = rows.div(rows.sum(axis=1), axis=0).add_prefix("var_share_")
    rows = rows.mask(rows.abs() < 1e-12, 0.0)
    exps = pd.DataFrame(res_exp)
    if len(exps):
        exps = exps.div(exps.sum(axis=1), axis=0).add_prefix("var_share_")
    return rows.reset_index(), exps.reset_index() if len(exps) else exps


def make_plots(draws: dict, outdir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    written = []
    for sid, x in draws.items():
        x = x[np.isfinite(x)]
        if x.size == 0:
            print(f"  {sid}: no finite draws, histogram skipped")
            continue
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        # clip the binning range so heavy tails (R_m near natural) do not
        # squash the histogram into one bar
        r0, r1 = np.percentile(x, [0.5, 99.5])
        if r1 <= r0:
            r0, r1 = r0 - 0.5, r1 + 0.5
        ax.hist(x, bins=80, range=(r0, r1), color="#4a6fa5", alpha=0.85)
        lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
        for v, ls in [(lo, ":"), (med, "-"), (hi, ":")]:
            ax.axvline(v, color="k", ls=ls, lw=1)
        ax.set_xlabel("ZrSiO$_4$ solubility (mmol/kg H$_2$O)")
        ax.set_ylabel("MC draws")
        ax.set_title(f"{sid}: {med:.4g} [{lo:.4g}, {hi:.4g}]", fontsize=9)
        fig.tight_layout()
        f = outdir / f"mc_hist_{safe_filename(sid)}.png"
        fig.savefig(f, dpi=200)
        plt.close(fig)
        written.append(f)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Zircon solubility by 30Si isotope dilution with Monte Carlo errors.")
    p.add_argument("input", nargs="?", default=None,
                   help="input CSV (default: zircon_id_input_template.csv in the "
                        "same folder as this script)")
    p.add_argument("-n", "--draws", type=int, default=100_000,
                   help="Monte Carlo draws (default 100000)")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("-o", "--outdir", default=None,
                   help="output folder (default: next to input)")
    p.add_argument("--plots", dest="plots", action="store_true", default=True,
                   help="write a histogram PNG for each sample (default)")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   help="do not write histograms")
    p.add_argument("--budget", action="store_true",
                   help="also write an uncertainty budget (variance share by input)")
    p.add_argument("--describe", action="store_true",
                   help="print the input column definitions and exit")
    a = p.parse_args(argv)

    if a.describe:
        for k, v in COLUMN_DOC.items():
            sd = f"  (+ {PAIRS[k]})" if k in PAIRS else ""
            print(f"{k:18s} {v}{sd}")
        return

    script_dir = Path(__file__).resolve().parent
    inp = Path(a.input) if a.input else script_dir / "zircon_id_input_template.csv"
    if not inp.is_absolute() and not inp.exists() and (script_dir / inp).exists():
        inp = script_dir / inp          # e.g. run from VS Code with another cwd
    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")
    inp = inp.resolve()
    df = read_input(inp)
    # all outputs (CSVs and histograms) go in the input file's folder
    outdir = Path(a.outdir).resolve() if a.outdir else inp.parent
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {inp}  ({len(df)} samples, "
          f"{df['experiment_id'].nunique()} experiments)")
    print(f"Output: {outdir}")

    rows, exps, draws_store = run(df, a.draws, a.seed)
    stem = inp.stem
    f1 = outdir / f"{stem}_results_samples.csv"
    rows.to_csv(f1, index=False)
    print(f"Wrote {f1}")
    if len(exps):
        f2 = outdir / f"{stem}_results_timeseries.csv"
        exps.to_csv(f2, index=False)
        print(f"Wrote {f2}")
    if a.budget:
        br, be = uncertainty_budget(df, a.draws, a.seed)
        f3 = outdir / f"{stem}_uncertainty_budget.csv"
        if len(be):
            be = be.rename(columns={"experiment_id": "id"}).assign(result="time-series integral")
        br = br.rename(columns={"label": "id"}).assign(result="endpoint")
        pd.concat([br, be], ignore_index=True).to_csv(f3, index=False)
        print(f"Wrote {f3}")
    if a.plots:
        pngs = make_plots(draws_store, outdir)
        print(f"Wrote {len(pngs)} histograms to {outdir}")

    # brief console summary
    cols = ["label", "mZrSiO4_mmolal_median", "mZrSiO4_mmolal_p2.5",
            "mZrSiO4_mmolal_p97.5", "s0_pure_water_mmolal_median",
            "error_magnification_R30", "R30_minus_Rnat_in_sd"]
    with pd.option_context("display.width", 140, "display.float_format", "{:.4g}".format):
        print(rows[cols].to_string(index=False))
        if len(exps):
            print(exps[["experiment_id", "n_points", "dSiB_integral_mmolal_median",
                        "dSiB_integral_mmolal_p2.5",
                        "dSiB_integral_mmolal_p97.5"]].to_string(index=False))

    weak = rows[(rows["R30_minus_Rnat_in_sd"] < 3) | (rows["error_magnification_R30"] > 10)]
    for _, w in weak.iterrows():
        print(f"WARNING {w['label']}: measured 30Si/28Si is "
              f"{w['R30_minus_Rnat_in_sd']:.1f} sd from the natural ratio "
              f"(error magnification {w['error_magnification_R30']:.0f}); "
              "the spike is nearly swamped and this result is not reliable.")


if __name__ == "__main__":
    main()
