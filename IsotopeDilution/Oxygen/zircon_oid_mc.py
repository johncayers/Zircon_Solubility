#!/usr/bin/env python3
"""
zircon_oid_mc.py -- Zircon solubility in H2O by oxygen isotope dilution,
with Monte Carlo error propagation.

The capsule holds isotopically light water (USGS49, d18O = -50.55 permil
VSMOW) and a 30SiO2 spike (d18O = +20.77 permil) that dissolves at once to
give a homogeneous fluid of composition d0. Dissolving Mud Tank zircon
(d18O = +5.0 permil) adds 4 mol O per mol ZrSiO4 and raises d18O of the
fluid. Oxygen mass balance, written exactly in 18O atom fractions x, gives

    N0 = n_w + 2 n_sp
    n_z = N0 * (x_fl - x_0) / (4 * (x_z - x_fl))
    m_ZrSiO4 = n_z / W

with n_w = W / M_H2O. In delta notation (linear mixing, error < 0.01 permil)
this is  m = (55.508 + 2 m_sp) (d_fl - d_0) / (4 (d_z - d_fl)).

d0 is calculated from the loaded water and spike, or taken from a measured
zircon-free blank run (d18O_0_meas). An optional correction converts the
measured d18O of extracted water to bulk-fluid d18O when dissolved/quench
solids hold O fractionated by Delta_solute_water relative to water.

Outputs, per sample row:
  * m_ZrSiO4 in mmol/kg, mg ZrSiO4/kg H2O and ug Zr/g H2O
  * d0 used, isotope shift d_fl - d0, sensitivity (permil per mmol/kg)
    and a 2-sigma detection limit (mmol/kg)
  * common-ion-corrected pure-water solubility s0 = sqrt(m_Zr * m_Si,total)
  * ratio to an independent Si isotope-dilution result, if supplied

Usage:
    python zircon_oid_mc.py zircon_oid_input_template.csv
    python zircon_oid_mc.py input.csv -n 200000 --seed 1 --budget --plots
    python zircon_oid_mc.py --describe

Uncertainties in the input file are 1-sigma absolute and are sampled as
independent normal distributions. Parameters shared by an experiment
(water mass, spike amount and composition, end-member d18O, blank d0,
Delta_solute_water) are drawn once per experiment, so they are correctly
correlated across the rows of a time series.

J. C. Ayers, Vanderbilt University. Requires numpy and pandas
(matplotlib only for --plots). Companion to zircon_id_mc.py (30Si ID).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (edit here if you want different reference values)
# ---------------------------------------------------------------------------
# VSMOW oxygen isotope ratios and the triple-isotope exponent used to
# estimate 17O from 18O (17O only enters the atom-fraction normalisation).
R18_VSMOW = 0.0020052      # 18O/16O (Baertschi 1976)
R17_VSMOW = 0.0003799      # 17O/16O (Li et al. 1988)
LAMBDA_17 = 0.528

# Default end-member d18O (permil VSMOW) and 1-sigma, used when the input
# cell is blank.
DEFAULTS = {
    "d18O_H2O": (-50.55, 0.04),   # USGS49 (Lorenz et al., 2017)
    "d18O_SiO2": (20.77, 0.0),    # 30SiO2 spike, as supplied
    "d18O_zrc": (5.0, 0.0),       # Mud Tank zircon, adopted value
    "spike_A28": (0.0046, 0.0),   # spike 28Si atom fraction
    "spike_A29": (0.0002, 0.0),   # spike 29Si atom fraction
    "Delta_solute_water": (0.0, 0.0),
}

# Masses (g/mol)
M_H2O = 18.01528
M_28SI = 27.9769265
M_29SI = 28.9764947
M_30SI = 29.9737701
M_O = 15.9994
M_ZR = 91.224
M_ZRSIO4 = 183.307
MOL_H2O_PER_KG = 1000.0 / M_H2O          # 55.508

# ---------------------------------------------------------------------------
# Input columns
# ---------------------------------------------------------------------------
REQUIRED = ["experiment_id", "sample_id", "W_H2O_mg", "d18O_fl"]
PAIRS = {
    "W_H2O_mg": "W_H2O_mg_sd",
    "spike_SiO2_ug": "spike_SiO2_ug_sd",
    "spike_Si_mmolal": "spike_Si_mmolal_sd",
    "spike_A28": "spike_A28_sd",
    "spike_A29": "spike_A29_sd",
    "d18O_H2O": "d18O_H2O_sd",
    "d18O_SiO2": "d18O_SiO2_sd",
    "d18O_zrc": "d18O_zrc_sd",
    "d18O_0_meas": "d18O_0_meas_sd",
    "d18O_fl": "d18O_fl_sd",
    "Delta_solute_water": "Delta_solute_water_sd",
    "mZrSiO4_SiID_mmolal": "mZrSiO4_SiID_mmolal_sd",
}
SHARED = ["W_H2O_mg", "spike_SiO2_ug", "spike_Si_mmolal", "spike_A28",
          "spike_A29", "d18O_H2O", "d18O_SiO2", "d18O_zrc", "d18O_0_meas",
          "Delta_solute_water"]

COLUMN_DOC = {
    "experiment_id": "Run/capsule identifier. Rows sharing it share draws.",
    "sample_id": "Unique label for this fluid analysis.",
    "time_h": "Run duration or sampling time (h).",
    "W_H2O_mg": "Mass of USGS49 H2O loaded (mg).",
    "spike_SiO2_ug": "Mass of 30SiO2 spike loaded (ug). Give this OR spike_Si_mmolal.",
    "spike_Si_mmolal": "Spike Si molality if added as a stock solution (mmol/kg H2O).",
    "spike_A28": "28Si atom fraction of spike (sets spike molar mass). Default 0.0046.",
    "spike_A29": "29Si atom fraction of spike. Default 0.0002.",
    "d18O_H2O": "d18O of starting water, permil VSMOW. Default -50.55 +/- 0.04 (USGS49).",
    "d18O_SiO2": "d18O of 30SiO2 spike, permil VSMOW. Default +20.77.",
    "d18O_zrc": "d18O of Mud Tank zircon, permil VSMOW. Default +5.0.",
    "d18O_0_meas": "Optional measured d18O of a zircon-free spiked blank (permil). "
                   "Replaces the calculated d0 when given.",
    "d18O_fl": "Measured d18O of the quenched fluid (permil VSMOW).",
    "Delta_solute_water": "Optional d18O(solute/quench solids) - d18O(water), permil. "
                          "Converts measured water d18O to bulk fluid. Default 0.",
    "mZrSiO4_SiID_mmolal": "Optional independent solubility from 30Si ID (mmol/kg) "
                           "for comparison.",
    "notes": "Free text; ignored.",
}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def read_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Input is missing required columns: {missing}")
    df = df.dropna(subset=["experiment_id", "d18O_fl"]).copy()
    for v, s in PAIRS.items():
        df[v] = _num(df, v)
        df[s] = _num(df, s)
        if v in DEFAULTS:
            val, sd = DEFAULTS[v]
            blank = df[v].isna()
            df.loc[blank, v] = val
            df.loc[blank & df[s].isna(), s] = sd
        df[s] = df[s].fillna(0.0)
    df["time_h"] = _num(df, "time_h")
    bad = df[df["spike_SiO2_ug"].isna() & df["spike_Si_mmolal"].isna()]
    if len(bad):
        sys.exit("Each row needs spike_SiO2_ug or spike_Si_mmolal (0 if unspiked): "
                 f"{list(bad['sample_id'])}")
    return df


# ---------------------------------------------------------------------------
# Isotope algebra (vectorised; all arguments may be arrays)
# ---------------------------------------------------------------------------
def x18_from_delta(d):
    """18O atom fraction from d18O (permil VSMOW)."""
    r18 = R18_VSMOW * (1.0 + d / 1000.0)
    r17 = R17_VSMOW * (1.0 + d / 1000.0) ** LAMBDA_17
    return r18 / (1.0 + r17 + r18)


def delta_from_x18(x, iters=3):
    """Invert x18_from_delta (fixed-point; converges in 2-3 steps)."""
    d = np.zeros_like(np.asarray(x, dtype=float))
    for _ in range(iters):
        r17 = R17_VSMOW * (1.0 + d / 1000.0) ** LAMBDA_17
        r18 = x * (1.0 + r17) / (1.0 - x)
        d = (r18 / R18_VSMOW - 1.0) * 1000.0
    return d


def oid_moles(N0, x0, x_fl, x_z):
    """Moles ZrSiO4 dissolved into a fluid holding N0 mol O of 18O
    fraction x0, from the fluid 18O fraction x_fl (4 O per ZrSiO4)."""
    return N0 * (x_fl - x0) / (4.0 * (x_z - x_fl))


def solute_O_fraction(m_si, m_z):
    """Fraction of fluid O held by dissolved Si and Zr, counted as oxides
    (SiO2, ZrO2) -- 2 m_Si,total + 2 m_Zr = 2 m_sp + 4 m_z per kg H2O."""
    s = 2.0 * m_si + 4.0 * m_z
    return s / (MOL_H2O_PER_KG + s)


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


def solve_row(W_kg, n_sp, d_w, d_s, d_z, d_fl_meas, d0_meas, Dsw):
    """Deterministic solution for one draw set (arrays). Returns dict."""
    n_w = W_kg * MOL_H2O_PER_KG
    N0 = n_w + 2.0 * n_sp
    m_sp = n_sp / W_kg
    x_z = x18_from_delta(d_z)

    # baseline d0: measured blank (water) converted to bulk, or calculated
    if d0_meas is not None:
        d0 = d0_meas + solute_O_fraction(m_sp, 0.0) * Dsw
    else:
        x0c = (n_w * x18_from_delta(d_w) + 2.0 * n_sp * x18_from_delta(d_s)) / N0
        d0 = delta_from_x18(x0c)
    x0 = x18_from_delta(d0)

    # measured water -> bulk fluid; f_s depends on m_z, so iterate
    m_z = np.zeros_like(W_kg)
    for _ in range(3):
        d_fl = d_fl_meas + solute_O_fraction(m_sp, m_z) * Dsw
        x_fl = x18_from_delta(d_fl)
        m_z = oid_moles(N0, x0, x_fl, x_z) / W_kg
    slope = 4.0 * (d_z - d0) / (MOL_H2O_PER_KG + 2.0 * m_sp)  # permil per mol/kg
    return {"m_z": m_z, "d0": d0, "d_fl": d_fl, "shift": d_fl - d0,
            "slope": slope, "m_sp": m_sp}


def run(df: pd.DataFrame, n_draws: int, seed: int | None):
    rng = np.random.default_rng(seed)
    row_out, draws_store = [], {}

    for exp_id, g in df.groupby("experiment_id", sort=False):
        g = g.sort_values("time_h", na_position="last")
        first = g.iloc[0]

        # --- shared experiment parameters ---
        W_kg = draw(rng, first["W_H2O_mg"], first["W_H2O_mg_sd"], n_draws) * 1e-6
        if np.isfinite(first["spike_SiO2_ug"]):
            S28 = np.clip(draw(rng, first["spike_A28"], first["spike_A28_sd"], n_draws), 0, 1)
            S29 = np.clip(draw(rng, first["spike_A29"], first["spike_A29_sd"], n_draws), 0, 1)
            S30 = 1.0 - S28 - S29
            M_sp = S28 * M_28SI + S29 * M_29SI + S30 * M_30SI + 2 * M_O
            n_sp = draw(rng, first["spike_SiO2_ug"], first["spike_SiO2_ug_sd"],
                        n_draws) * 1e-6 / M_sp                        # mol
        else:
            n_sp = draw(rng, first["spike_Si_mmolal"], first["spike_Si_mmolal_sd"],
                        n_draws) * 1e-3 * W_kg                        # mol
        d_w = draw(rng, first["d18O_H2O"], first["d18O_H2O_sd"], n_draws)
        d_s = draw(rng, first["d18O_SiO2"], first["d18O_SiO2_sd"], n_draws)
        d_z = draw(rng, first["d18O_zrc"], first["d18O_zrc_sd"], n_draws)
        Dsw = draw(rng, first["Delta_solute_water"], first["Delta_solute_water_sd"],
                   n_draws)
        d0_meas = (draw(rng, first["d18O_0_meas"], first["d18O_0_meas_sd"], n_draws)
                   if np.isfinite(first["d18O_0_meas"]) else None)

        for _, r in g.iterrows():
            d_fl = draw(rng, r["d18O_fl"], r["d18O_fl_sd"], n_draws)
            s = solve_row(W_kg, n_sp, d_w, d_s, d_z, d_fl, d0_meas, Dsw)
            m_z = s["m_z"]

            out = {"experiment_id": exp_id, "sample_id": r["sample_id"],
                   "time_h": r["time_h"],
                   "d0_source": "measured blank" if d0_meas is not None else "calculated"}
            out.update(summarise(m_z * 1e3, "mZrSiO4_mmolal"))
            out.update(summarise(m_z * M_ZRSIO4 * 1e3, "ZrSiO4_mg_per_kgH2O"))
            out.update(summarise(m_z * M_ZR * 1e3, "Zr_ug_per_gH2O"))
            out.update(summarise(s["d0"], "d18O_0"))
            out.update(summarise(s["shift"], "d18O_shift"))

            # common-ion correction to zero added silica (sign kept so that
            # draws with m_z <= 0 are not silently dropped)
            with np.errstate(invalid="ignore"):
                s0 = np.sign(m_z) * np.sqrt(np.abs(m_z) * (s["m_sp"] + m_z))
            out.update(summarise(s0 * 1e3, "s0_pure_water_mmolal"))
            out.update(summarise(s["m_sp"] * 1e3, "mSi_spike_mmolal"))

            # sensitivity and detection limit (2 sigma on the shift)
            slope_mmol = float(np.nanmedian(s["slope"])) / 1e3   # permil per mmol/kg
            sd_fl = r["d18O_fl_sd"]
            sd_0 = (first["d18O_0_meas_sd"] if d0_meas is not None
                    else float(np.nanstd(s["d0"])))
            out["sensitivity_permil_per_mmolal"] = slope_mmol
            out["detection_limit_2s_mmolal"] = (2.0 * np.hypot(sd_fl, sd_0) / slope_mmol
                                                if slope_mmol > 0 else np.nan)

            # comparison with independent 30Si isotope dilution
            if np.isfinite(r["mZrSiO4_SiID_mmolal"]):
                si = draw(rng, r["mZrSiO4_SiID_mmolal"], r["mZrSiO4_SiID_mmolal_sd"],
                          n_draws)
                out.update(summarise(m_z * 1e3 - si, "O_minus_SiID_mmolal"))
                out["predicted_shift_from_SiID_permil"] = (
                    r["mZrSiO4_SiID_mmolal"] * slope_mmol)

            row_out.append(out)
            draws_store[r["sample_id"]] = m_z * 1e3

    return pd.DataFrame(row_out), draws_store


BUDGET_GROUPS = {
    "water mass": ["W_H2O_mg_sd"],
    "spike amount": ["spike_SiO2_ug_sd", "spike_Si_mmolal_sd"],
    "spike Si composition": ["spike_A28_sd", "spike_A29_sd"],
    "d18O water": ["d18O_H2O_sd"],
    "d18O spike": ["d18O_SiO2_sd"],
    "d18O zircon": ["d18O_zrc_sd"],
    "d18O blank (measured d0)": ["d18O_0_meas_sd"],
    "d18O fluid": ["d18O_fl_sd"],
    "solute-water fractionation": ["Delta_solute_water_sd"],
}


def uncertainty_budget(df, n_draws, seed):
    """Variance share of each input group, by one-group-at-a-time MC."""
    sd_cols = [c for c in PAIRS.values() if c in df.columns]
    res = {}
    for grp, cols in BUDGET_GROUPS.items():
        d = df.copy()
        for c in sd_cols:
            if c not in cols:
                d[c] = 0.0
        r, _ = run(d, n_draws, seed)
        res[grp] = r.set_index("sample_id")["mZrSiO4_mmolal_sd"] ** 2
    t = pd.DataFrame(res)
    t = t.loc[:, t.sum(axis=0) > 0]
    t = t.div(t.sum(axis=1), axis=0).add_prefix("var_share_")
    return t.reset_index()


def make_plots(draws: dict, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for sid, x in draws.items():
        x = x[np.isfinite(x)]
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.hist(x, bins=80, color="#a5634a", alpha=0.85)
        lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
        for v, ls in [(lo, ":"), (med, "-"), (hi, ":")]:
            ax.axvline(v, color="k", ls=ls, lw=1)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlabel("ZrSiO$_4$ solubility, $\\delta^{18}$O ID (mmol/kg H$_2$O)")
        ax.set_ylabel("MC draws")
        ax.set_title(f"{sid}: {med:.4g} [{lo:.4g}, {hi:.4g}]", fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / f"mc_hist_oid_{sid}.png", dpi=200)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Zircon solubility by oxygen isotope dilution with Monte Carlo errors.")
    p.add_argument("input", nargs="?", help="input CSV (see template)")
    p.add_argument("-n", "--draws", type=int, default=100_000,
                   help="Monte Carlo draws (default 100000)")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("-o", "--outdir", default=None,
                   help="output folder (default: next to input)")
    p.add_argument("--plots", action="store_true",
                   help="write histogram PNG for each sample")
    p.add_argument("--budget", action="store_true",
                   help="also write an uncertainty budget (variance share by input)")
    p.add_argument("--describe", action="store_true",
                   help="print the input column definitions and exit")
    a = p.parse_args(argv)

    if a.describe or not a.input:
        for k, v in COLUMN_DOC.items():
            sd = f"  (+ {PAIRS[k]})" if k in PAIRS else ""
            print(f"{k:20s} {v}{sd}")
        return

    inp = Path(a.input)
    df = read_input(inp)
    outdir = Path(a.outdir) if a.outdir else inp.parent
    outdir.mkdir(parents=True, exist_ok=True)

    rows, draws_store = run(df, a.draws, a.seed)
    stem = inp.stem
    f1 = outdir / f"{stem}_results_samples.csv"
    rows.to_csv(f1, index=False)
    print(f"Wrote {f1}")
    if a.budget:
        f3 = outdir / f"{stem}_uncertainty_budget.csv"
        uncertainty_budget(df, a.draws, a.seed).to_csv(f3, index=False)
        print(f"Wrote {f3}")
    if a.plots:
        make_plots(draws_store, outdir)
        print(f"Wrote histograms to {outdir}")

    cols = ["sample_id", "d18O_shift_median", "mZrSiO4_mmolal_median",
            "mZrSiO4_mmolal_p2.5", "mZrSiO4_mmolal_p97.5",
            "mZrSiO4_mmolal_frac_le0", "detection_limit_2s_mmolal"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print(rows[cols].to_string(index=False))


if __name__ == "__main__":
    main()
