#!/usr/bin/env python3
"""Build alpha_calibration.json for the Neuropil Sweep tab from ground-truth
simulations (sim_popcorr_omsi.m -> omsi_input_*.mat, run_omsi_sweep.py ->
omsi_output_*.mat; see sim_alpha_grid.m in the neuropil_fig analysis).

For every simulated condition and correlation level it records, relative to
that level's alpha_min (minimum of the mean pairwise correlation of
F - alpha*Fneu, computed exactly as the app does):
  spike_train  best / near-optimal band of the median COSMIC-style triangular
               correlation between inferred and true spikes (within 5% of best)
  precision    same for precision
  recall       same for recall
  correlation  alpha where the inferred mean pairwise spike-count correlation
               (1 s bins) equals the truth; band = within max(0.01, 25%) of it
               (the independent population is skipped: there is no crossing)
and the recording statistics the app matches on (neuropil_recording_stats,
imported from kurtosis_checker.py so the estimator is identical).

Usage:
  python build_alpha_calibration.py --grid <alpha_grid folder> --out alpha_calibration.json
  python build_alpha_calibration.py --pair omsi_input.mat omsi_output.mat baseline --out ...
  add --print-default to print the table as a Python literal (for the built-in default)
Put the resulting alpha_calibration.json next to kurtosis_checker.py.
"""
import os, sys, glob, json, argparse, datetime, types
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
try:
    import tkinter  # noqa: F401
except ImportError:                       # headless machine: stub what the app module touches at import
    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (), {"__init__": lambda self, *a, **k: None})
    tk = _Stub("tkinter")
    tk.filedialog = _Stub("tkinter.filedialog")
    tk.messagebox = _Stub("tkinter.messagebox")
    sys.modules.update({"tkinter": tk, "tkinter.filedialog": tk.filedialog,
                        "tkinter.messagebox": tk.messagebox})
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.use = lambda *a, **k: None
    sys.modules.setdefault("matplotlib.backends.backend_tkagg", _Stub("matplotlib.backends.backend_tkagg"))
    sys.modules.setdefault("PIL.ImageTk", _Stub("PIL.ImageTk"))
from kurtosis_checker import neuropil_recording_stats  # noqa: E402

TOL = 0.05
TOL_CORR_ABS, TOL_CORR_REL = 0.01, 0.25   # inferred within max(0.01, 25% of truth)
FINE = np.round(np.arange(0, 3.0001, 0.01), 2)


def mean_pair_corr(F, Fn, alphas):
    n = F.shape[0]
    iu = np.triu_indices(n, k=1)
    out = np.full(len(alphas), np.nan)
    for k, a in enumerate(alphas):
        with np.errstate(invalid="ignore"):
            R = np.corrcoef(F - a * Fn)
        v = R[iu]
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = v.mean()
    return out


def band(al, y, keep, j):
    """contiguous run of `keep` (bool) containing index j -> (lo, hi)"""
    keep = keep.copy()
    keep[j] = True
    a = b = j
    while a > 0 and keep[a - 1]:
        a -= 1
    while b < len(al) - 1 and keep[b + 1]:
        b += 1
    return float(al[a]), float(al[b])


def first_crossing(x, y, t):
    d = y - t
    k = np.flatnonzero(d[:-1] * d[1:] <= 0)
    if not len(k):
        return None
    k = k[0]
    if d[k] == d[k + 1]:
        return float(x[k])
    return float(x[k] + (x[k + 1] - x[k]) * d[k] / (d[k] - d[k + 1]))


def condition(in_path, out_path, name):
    I = sio.loadmat(in_path, squeeze_me=True)
    R = sio.loadmat(out_path, squeeze_me=True)
    hz = float(np.squeeze(I["hz"]))
    Fs = [np.asarray(f, float) for f in np.atleast_1d(I["F"])]
    Ns = [np.asarray(f, float) for f in np.atleast_1d(I["Fneu"])]
    al = np.atleast_1d(R["alphas"]).astype(float)
    nL = len(Fs)
    # ring_ratio needs spikes: use the true spikes as an idealised spks array
    # (every spike passes the app's event threshold); level 1 = independent cells
    spk = np.zeros_like(Fs[0])
    if "spikeTimes" in I:
        for i, ts in enumerate(np.atleast_1d(I["spikeTimes"])[0]):
            fr = np.floor(np.atleast_1d(ts).astype(float) * hz).astype(int)
            fr = fr[(fr >= 0) & (fr < spk.shape[1])]
            np.add.at(spk[i], fr, 1e9)
    st = neuropil_recording_stats(Fs[0], Ns[0], hz, spk if spk.any() else None)
    rsc = [float(v) for v in np.atleast_1d(R.get("spkCorrTrueSubset", I.get("spkCorr")))]
    offs = {g: {"best": [], "lo": [], "hi": []} for g in ["spike_train", "precision", "recall", "correlation"]}
    amins = []
    for l in range(nL):
        fc = mean_pair_corr(Fs[l], Ns[l], FINE)
        amin = float(FINE[np.nanargmin(fc)])
        amins.append(amin)
        for g, key in [("spike_train", "triCorr"), ("precision", "precision"), ("recall", "recall")]:
            y = np.nanmedian(np.asarray(R[key], float)[l], axis=1)      # (n_alphas,)
            j = int(np.nanargmax(y))
            lo, hi = band(al, y, y >= (1 - TOL) * y[j], j)
            offs[g]["best"].append(round(float(al[j]) - amin, 3))
            offs[g]["lo"].append(round(lo - amin, 3))
            offs[g]["hi"].append(round(hi - amin, 3))
        yi = np.asarray(R["spkCorrInferred"], float)[l]
        yt = float(np.atleast_1d(R["spkCorrTrueSubset"])[l])
        if yt < 0.005:                      # independent population: nothing to match
            for k_ in ("best", "lo", "hi"):
                offs["correlation"][k_].append(None)
            continue
        e = np.abs(yi - yt)
        a = first_crossing(al, yi, yt)
        j = int(np.argmin(e)) if a is None else int(np.argmin(np.abs(al - a)))
        if a is None:
            a = float(al[j])
        lo, hi = band(al, e, e <= max(TOL_CORR_ABS, TOL_CORR_REL * yt), j)
        offs["correlation"]["best"].append(round(a - amin, 3))
        offs["correlation"]["lo"].append(round(lo - amin, 3))
        offs["correlation"]["hi"].append(round(hi - amin, 3))
    cT = np.atleast_1d(I["cTrue"]).astype(float) if "cTrue" in I else None
    return {"name": name,
            "stats": {k: round(float(st[k]), 4) for k in ["np_fluct", "noise_F", "np_to_F", "ring_ratio"] if k in st},
            "rsc": [round(v, 4) for v in rsc],
            "alpha_min": [round(v, 3) for v in amins],
            "c_true_median": None if cT is None else round(float(np.median(cT)), 3),
            "n_cells": int(Fs[0].shape[0]),
            "offsets": offs}


def absolute_summary(conds):
    """Simulation-wide best alpha per goal (absolute, not relative to alpha_min),
    over every condition x population: median and middle half. Valid for
    contamination like the Allen 2x GCaMP6s recordings that set the simulation."""
    out, pool = {}, []
    for g in ["spike_train", "precision", "recall", "correlation"]:
        v = [a + o for c in conds for a, o in zip(c["alpha_min"], c["offsets"][g]["best"]) if o is not None]
        v = np.asarray(v, float)
        out[g] = {"median": round(float(np.median(v)), 2),
                  "q25": round(float(np.percentile(v, 25)), 2), "q75": round(float(np.percentile(v, 75)), 2),
                  "n": int(len(v))}
        if g != "recall":
            pool += list(v)
    pool = np.asarray(pool, float)
    out["pooled_spike_precision_correlation"] = {
        "median": round(float(np.median(pool)), 2),
        "q25": round(float(np.percentile(pool, 25)), 2), "q75": round(float(np.percentile(pool, 75)), 2),
        "n": int(len(pool))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", help="folder with omsi_input_<tag>.mat / omsi_output_<tag>.mat")
    ap.add_argument("--pair", nargs=3, action="append", metavar=("INPUT", "OUTPUT", "NAME"))
    ap.add_argument("--out", default="alpha_calibration.json")
    ap.add_argument("--provisional", action="store_true", help="mark the table as provisional")
    ap.add_argument("--print-default", action="store_true")
    a = ap.parse_args()
    jobs = []
    if a.grid:
        for o in sorted(glob.glob(os.path.join(a.grid, "omsi_output_*.mat"))):
            tag = os.path.basename(o)[len("omsi_output_"):-4]
            i = os.path.join(a.grid, f"omsi_input_{tag}.mat")
            if os.path.exists(i):
                jobs.append((i, o, tag))
    for p in a.pair or []:
        jobs.append(tuple(p))
    if not jobs:
        sys.exit("nothing to do: give --grid or --pair")
    conds = []
    for i, o, n in jobs:
        print(f"{n}: {os.path.basename(i)} + {os.path.basename(o)}")
        conds.append(condition(i, o, n))
    cal = {"version": datetime.date.today().isoformat(),
           "provisional": bool(a.provisional or len(conds) < 2),
           "description": "alpha offsets (alpha - alpha_min) from Allen-calibrated simulations scored "
                          "with OMSI against ground-truth spikes; see tools/build_alpha_calibration.py",
           "match_on": ["np_fluct", "noise_F", "ring_ratio"],
           "tol": TOL, "tol_corr_abs": TOL_CORR_ABS, "tol_corr_rel": TOL_CORR_REL,
           "absolute_summary": absolute_summary(conds),
           "conditions": conds}
    with open(a.out, "w") as fh:
        json.dump(cal, fh, indent=1)
    print("wrote", a.out)
    if a.print_default:
        import pprint
        print(pprint.pformat(cal, width=110, sort_dicts=False))


if __name__ == "__main__":
    main()
