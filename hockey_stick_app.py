#!/usr/bin/env python3
"""
Hockey Stick — PTC Gain / Photon Flux Checker
==============================================
Loads a Suite2p output folder and runs a photon-transfer-curve (PTC)
"hockey stick" analysis: pixelwise mean/variance (via frame-differencing,
which cancels slow biological signal and isolates shot noise) is binned
and fit to recover the true camera/PMT gain (ADU/photon) after excess
noise factor (ENF) correction, then converted to per-cell photon flux
using Suite2p's F.npy traces.

Registered-movie source, in priority order:
  1. reg_tif/  — Suite2p's saved registered TIFF stack (if the user
     exported it during processing)
  2. data.bin  — Suite2p's own motion-corrected binary (ops['reg_file']),
     present whenever Suite2p ran registration, even if reg_tif wasn't
     exported
  3. raw acquisition TIFFs, motion-corrected on the fly with NoRMCorre
     (via CaImAn) — last resort, and the slowest path

Dependencies: numpy, scipy, matplotlib, pillow, tifffile
Optional (only needed for the raw-tiff/NormCorre fallback): caimain
    pip install caiman      (conda install -c conda-forge caiman recommended)
"""

import os
import glob
import threading
import traceback

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image, ImageTk

# ── palette (matches kurtosis_checker.py) ──────────────────────────────────
BG_DARK     = "#0e0e0e"
BG_MID      = "#1a1a1a"
BG_PANEL    = "#2e2e2e"
ACCENT      = "#e94560"
TEXT_DIM    = "#555555"
TEXT_MAIN   = "#aaaaaa"
TEXT_BRIGHT = "#e8e8e8"
C_FIT       = "#27ae60"   # green   — shot-noise fit line
C_FLOOR     = "#e67e22"   # orange  — read-noise floor
C_EXCL      = "#555555"   # excluded points
C_INCL      = "#3fa7ff"   # included points
C_HIST      = "#e94560"   # photon-flux histogram


# ── small numeric helpers ───────────────────────────────────────────────────

def weighted_linfit(x, y, w):
    """Weighted least-squares line fit. Returns (slope, intercept, r2)."""
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    W = w.sum()
    if W <= 0 or len(x) < 2:
        return np.nan, np.nan, np.nan
    xm = np.sum(w * x) / W
    ym = np.sum(w * y) / W
    sxx = np.sum(w * (x - xm) ** 2)
    sxy = np.sum(w * (x - xm) * (y - ym))
    if sxx <= 0:
        return np.nan, np.nan, np.nan
    slope = sxy / sxx
    intercept = ym - slope * xm
    yhat = slope * x + intercept
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - ym) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2


def bootstrap_median_sem(vals, n_boot=1000, rng=None):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    rng = rng or np.random.default_rng(0)
    n = len(vals)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = np.median(rng.choice(vals, size=n, replace=True))
    return float(np.median(vals)), float(np.std(boots))


# ── registered-movie access layer ───────────────────────────────────────────

class RegisteredMovie:
    """Uniform (T, Y, X) accessor over reg_tif / data.bin / NormCorre output.

    `source` is one of 'reg_tif', 'data.bin', 'normcorre' for status reporting.
    Backed by a numpy array or memmap — either way, standard slicing works.
    """
    def __init__(self, array, source, note=""):
        self.array = array
        self.source = source
        self.note = note
        self.shape = array.shape

    def sample_indices(self, max_frames):
        n = self.shape[0]
        if n <= max_frames:
            return np.arange(n)
        return np.round(np.linspace(0, n - 1, max_frames)).astype(int)


def _find_plane_dirs(folder):
    """Suite2p folders can be the top save dir or an individual plane dir."""
    cands = [folder]
    cands += [os.path.join(folder, d) for d in ("plane0", "plane1", "plane2", "plane3", "combined")]
    cands += [os.path.join(folder, "suite2p", d) for d in ("plane0", "plane1", "plane2", "plane3")]
    return [c for c in cands if os.path.isdir(c)]


def find_suite2p_plane(folder):
    """Return the first plane dir (searched in priority order) containing ops.npy."""
    for d in _find_plane_dirs(folder):
        if os.path.exists(os.path.join(d, "ops.npy")):
            return d
    return None


def load_reg_tif(plane_dir, max_frames):
    """Priority 1: Suite2p's exported registered TIFF stack."""
    try:
        import tifffile
    except ImportError:
        return None

    reg_dir = os.path.join(plane_dir, "reg_tif")
    if not os.path.isdir(reg_dir):
        return None
    files = sorted(glob.glob(os.path.join(reg_dir, "*chan0*.tif*")) or
                    glob.glob(os.path.join(reg_dir, "*.tif*")))
    if not files:
        return None

    stacks = []
    for f in files:
        stacks.append(tifffile.imread(f))
    arr = np.concatenate([s if s.ndim == 3 else s[None] for s in stacks], axis=0)

    if arr.shape[0] > max_frames:
        idx = np.round(np.linspace(0, arr.shape[0] - 1, max_frames)).astype(int)
        arr = arr[idx]
    return RegisteredMovie(arr.astype(np.float32), "reg_tif",
                            f"{len(files)} file(s), {arr.shape[0]} frames sampled")


def load_data_bin(plane_dir, ops, max_frames):
    """Priority 2: Suite2p's own registered binary (data.bin)."""
    Ly, Lx = int(ops.get("Ly", 0)), int(ops.get("Lx", 0))
    nframes = int(ops.get("nframes", 0))
    if not (Ly and Lx and nframes):
        return None

    candidates = []
    rf = ops.get("reg_file")
    if rf:
        candidates.append(rf)
    candidates.append(os.path.join(plane_dir, "data.bin"))

    bin_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if bin_path is None:
        return None

    try:
        mm = np.memmap(bin_path, mode="r", dtype=np.int16, shape=(nframes, Ly, Lx))
    except Exception:
        return None

    idx = RegisteredMovie(mm, "data.bin").sample_indices(max_frames)
    arr = np.asarray(mm[idx]).astype(np.float32)
    return RegisteredMovie(arr, "data.bin", f"{bin_path}  ({arr.shape[0]}/{nframes} frames sampled)")


def _resolve_raw_tiffs(plane_dir, ops):
    """Try to locate the raw acquisition TIFFs Suite2p registered from."""
    for key in ("filelist", "tiff_list", "raw_file"):
        lst = ops.get(key)
        if lst:
            lst = lst if isinstance(lst, (list, tuple, np.ndarray)) else [lst]
            lst = [str(p) for p in lst]
            if all(os.path.exists(p) for p in lst):
                return lst
    dp = ops.get("data_path")
    if dp:
        dp = dp if isinstance(dp, (list, tuple, np.ndarray)) else [dp]
        for d in dp:
            d = str(d)
            if os.path.isdir(d):
                tiffs = sorted(glob.glob(os.path.join(d, "*.tif")) + glob.glob(os.path.join(d, "*.tiff")))
                if tiffs:
                    return tiffs
    return None


def run_normcorre(tiff_paths, pw_rigid, progress_cb=None):
    """Priority 3 (last resort): motion-correct raw TIFFs with CaImAn's NoRMCorre."""
    try:
        import caiman as cm
        from caiman.motion_correction import MotionCorrect
    except ImportError as e:
        raise RuntimeError(
            "CaImAn is not installed, so the raw-TIFF fallback can't run.\n\n"
            "Install it (conda recommended for CaImAn's compiled deps):\n"
            "  conda install -c conda-forge caiman\n"
            "or:\n"
            "  pip install caiman\n\n"
            f"(underlying error: {e})"
        )

    if progress_cb:
        progress_cb("Running NoRMCorre motion correction (this can take a while)...")

    mc = MotionCorrect(
        tiff_paths, dview=None,
        max_shifts=(6, 6), niter_rig=1,
        strides=(48, 48), overlaps=(24, 24),
        max_deviation_rigid=3, shifts_opencv=True,
        nonneg_movie=True, border_nan="copy",
        pw_rigid=pw_rigid,
    )
    mc.motion_correct(save_movie=True)
    mmap_file = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    m = cm.load(mmap_file[0] if isinstance(mmap_file, (list, tuple)) else mmap_file)
    arr = np.asarray(m).astype(np.float32)
    return arr


def load_registered_movie(plane_dir, ops, max_frames, pw_rigid, progress_cb=None, ask_raw_dir=None):
    """Try reg_tif -> data.bin -> raw tiff + NormCorre, in order."""
    mov = load_reg_tif(plane_dir, max_frames)
    if mov is not None:
        return mov

    mov = load_data_bin(plane_dir, ops, max_frames)
    if mov is not None:
        return mov

    tiff_paths = _resolve_raw_tiffs(plane_dir, ops)
    if not tiff_paths and ask_raw_dir is not None:
        d = ask_raw_dir()
        if d:
            tiff_paths = sorted(glob.glob(os.path.join(d, "*.tif")) + glob.glob(os.path.join(d, "*.tiff")))
    if not tiff_paths:
        raise RuntimeError(
            "No reg_tif export, no data.bin, and the raw acquisition TIFFs could not be "
            "located automatically (ops paths don't resolve on this machine)."
        )

    arr = run_normcorre(tiff_paths, pw_rigid, progress_cb=progress_cb)
    if arr.shape[0] > max_frames:
        idx = np.round(np.linspace(0, arr.shape[0] - 1, max_frames)).astype(int)
        arr = arr[idx]
    return RegisteredMovie(arr, "normcorre",
                            f"{len(tiff_paths)} raw TIFF(s), {arr.shape[0]} frames after motion correction")


# ── PTC / hockey-stick math ─────────────────────────────────────────────────

def roi_exclusion_mask(stat, Ly, Lx):
    """Boolean mask, True = background/non-cell pixel."""
    mask = np.ones((Ly, Lx), dtype=bool)
    for cell in stat:
        yp = np.asarray(cell["ypix"]); xp = np.asarray(cell["xpix"])
        valid = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
        mask[yp[valid], xp[valid]] = False
    return mask


def compute_ptc(movie_arr, spatial_bin=1, exclude_mask=None, n_mean_bins=40):
    """Frame-differencing mean/variance PTC estimate.

    For consecutive frame pairs (i, i+1): mean = (f_i+f_{i+1})/2,
    var = (f_i - f_{i+1})^2 / 2  (unbiased shot-noise variance estimator;
    slow biological/structural signal cancels in the difference).

    Returns per-pixel arrays (mu_px, var_px) and binned arrays
    (mu_bin, var_bin, n_bin) for plotting/fitting.
    """
    arr = movie_arr.astype(np.float32)
    T, Y, X = arr.shape
    if T < 2:
        raise RuntimeError("Need at least 2 frames for a PTC estimate.")

    if spatial_bin > 1:
        Yb, Xb = Y // spatial_bin, X // spatial_bin
        arr = arr[:, :Yb * spatial_bin, :Xb * spatial_bin]
        arr = arr.reshape(T, Yb, spatial_bin, Xb, spatial_bin).mean(axis=(2, 4))
        Y, X = Yb, Xb

    diffs = arr[:-1] - arr[1:]
    means = (arr[:-1] + arr[1:]) * 0.5
    var_stack = diffs ** 2 * 0.5

    mu_px = means.mean(axis=0)
    var_px = var_stack.mean(axis=0)

    if exclude_mask is not None:
        m = exclude_mask
        if m.shape != (Y, X) and spatial_bin > 1:
            m = m[:Y * spatial_bin:spatial_bin, :X * spatial_bin:spatial_bin][:Y, :X]
        keep = m
    else:
        keep = np.ones((Y, X), dtype=bool)

    mu_flat = mu_px[keep].ravel()
    var_flat = var_px[keep].ravel()
    good = np.isfinite(mu_flat) & np.isfinite(var_flat) & (mu_flat > 0)
    mu_flat, var_flat = mu_flat[good], var_flat[good]

    lo, hi = np.percentile(mu_flat, [0.5, 99.5])
    edges = np.linspace(lo, hi, n_mean_bins + 1)
    which = np.digitize(mu_flat, edges) - 1
    mu_bin, var_bin, n_bin = [], [], []
    for b in range(n_mean_bins):
        sel = which == b
        if sel.sum() < 5:
            continue
        mu_bin.append(np.median(mu_flat[sel]))
        var_bin.append(np.median(var_flat[sel]))
        n_bin.append(int(sel.sum()))

    return (mu_flat, var_flat,
            np.array(mu_bin), np.array(var_bin), np.array(n_bin))


def fit_gain(mu_bin, var_bin, n_bin, fit_pct_lo, fit_pct_hi, enf):
    """Weighted linear fit over the shot-noise-dominated region.

    Returns dict with slope (apparent gain), gain_true (ENF-corrected),
    intercept, r2, and the boolean mask of bins used in the fit.
    """
    if len(mu_bin) < 3:
        return dict(slope=np.nan, gain_true=np.nan, intercept=np.nan, r2=np.nan,
                     mask=np.zeros(len(mu_bin), dtype=bool))
    lo, hi = np.percentile(mu_bin, [fit_pct_lo, fit_pct_hi])
    mask = (mu_bin >= lo) & (mu_bin <= hi)
    if mask.sum() < 3:
        mask = np.ones(len(mu_bin), dtype=bool)
    slope, intercept, r2 = weighted_linfit(mu_bin[mask], var_bin[mask], n_bin[mask])
    gain_true = slope / (enf ** 2) if slope and np.isfinite(slope) else np.nan
    return dict(slope=slope, gain_true=gain_true, intercept=intercept, r2=r2, mask=mask)


def photon_flux_per_cell(F, Fneu, do_neuropil, neuropil_coeff, gain_true, fs):
    """Photons/cell/s from raw F using Wilt-style baseline (no background subtraction
    for the flux estimate itself — F0 here is only used elsewhere if needed)."""
    Fc = F.copy().astype(float)
    if do_neuropil and Fneu is not None:
        Fc = Fc - neuropil_coeff * Fneu
    if not np.isfinite(gain_true) or gain_true <= 0:
        return np.full(Fc.shape[0], np.nan)
    photons_per_frame = Fc.mean(axis=1) / gain_true
    return photons_per_frame * fs


# ── GUI ──────────────────────────────────────────────────────────────────────

class HockeyStickApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hockey Stick — PTC Gain / Photon Flux")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1520x920")
        self._set_icon()

        self.plane_dir = None
        self.ops = None
        self.F = None
        self.Fneu = None
        self.stat = None
        self.movie = None
        self.results = None

        self._build_controls()
        self._build_canvas()

    # ── icon (reuses icon.png next to this script if present) ──────────────
    def _set_icon(self):
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
            if os.path.exists(icon_path):
                img = Image.open(icon_path).convert("RGBA")
                self._icon_img = ImageTk.PhotoImage(img)
                self.root.wm_iconphoto(True, self._icon_img)
        except Exception:
            pass

    # ── controls ──────────────────────────────────────────────────────────
    def _build_controls(self):
        self._bar = tk.Frame(self.root, bg=BG_MID, pady=6)
        self._bar.pack(fill=tk.X)

        def _lbtn(parent, text, command, bg, fg="white", font_size=12, bold=True, side=tk.LEFT, padx=6):
            weight = "bold" if bold else "normal"
            lbl = tk.Label(parent, text=text, bg=bg, fg=fg,
                            font=("Helvetica", font_size, weight),
                            padx=16, pady=7, cursor="hand2", relief=tk.RAISED, bd=2)
            lbl.pack(side=side, padx=padx, pady=5)
            lbl.bind("<Button-1>", lambda e: command())
            lbl.bind("<Enter>", lambda e: lbl.config(relief=tk.SUNKEN))
            lbl.bind("<Leave>", lambda e: lbl.config(relief=tk.RAISED))
            return lbl

        _lbtn(self._bar, "📂  Load Suite2p Folder", self._load_dialog, bg="#27ae60", padx=(10, 6))

        self._settings_open = False
        self._settings_btn = _lbtn(self._bar, "⚙  Settings", self._toggle_settings, bg="#3a3a3a", bold=False)

        _lbtn(self._bar, "▶  Run Analysis", self.run_analysis, bg="#c0392b", font_size=13, side=tk.RIGHT, padx=10)

        self.status_var = tk.StringVar(value="No data loaded")
        tk.Label(self._bar, textvariable=self.status_var, bg=BG_MID, fg=TEXT_DIM,
                  font=("Helvetica", 10)).pack(side=tk.LEFT, padx=14)

        # ── settings panel ───────────────────────────────────────────────
        self._settings_frame = tk.Frame(self.root, bg="#141414", pady=6)

        sf = self._settings_frame

        def lbl(t, dim=False, parent=sf):
            tk.Label(parent, text=t, bg="#141414", fg=TEXT_DIM if dim else TEXT_MAIN,
                      font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 2))

        def ent(var, w=6, parent=sf):
            e = tk.Entry(parent, textvariable=var, width=w, bg=BG_PANEL, fg="white",
                          insertbackground="white", relief=tk.FLAT, font=("Helvetica", 10))
            e.pack(side=tk.LEFT)
            return e

        lbl("  ENF:")
        self.enf_var = tk.DoubleVar(value=1.2)
        ent(self.enf_var, 5)

        lbl("   |", dim=True); lbl("Fit range (pct of mean):")
        self.fit_lo_var = tk.DoubleVar(value=40.0)
        ent(self.fit_lo_var, 4)
        lbl("–")
        self.fit_hi_var = tk.DoubleVar(value=98.0)
        ent(self.fit_hi_var, 4)

        lbl("   |", dim=True); lbl("Spatial bin (px):")
        self.spatial_bin_var = tk.IntVar(value=1)
        ent(self.spatial_bin_var, 3)

        lbl("   |", dim=True); lbl("Max frames:")
        self.max_frames_var = tk.IntVar(value=2000)
        ent(self.max_frames_var, 6)

        lbl("   |", dim=True)
        self.exclude_roi_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sf, text="Exclude cell ROIs (background only)",
                        variable=self.exclude_roi_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 0))

        lbl("   |", dim=True)
        self.pw_rigid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sf, text="NormCorre: piecewise-rigid",
                        variable=self.pw_rigid_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 0))

        # second row: neuropil + fs, mirroring kurtosis_checker's settings
        sf2 = tk.Frame(self.root, bg="#141414")
        self._settings_frame2 = sf2

        def lbl2(t, dim=False):
            tk.Label(sf2, text=t, bg="#141414", fg=TEXT_DIM if dim else TEXT_MAIN,
                      font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 2))

        def ent2(var, w=6):
            e = tk.Entry(sf2, textvariable=var, width=w, bg=BG_PANEL, fg="white",
                          insertbackground="white", relief=tk.FLAT, font=("Helvetica", 10))
            e.pack(side=tk.LEFT)
            return e

        self.do_neuropil = tk.BooleanVar(value=True)
        tk.Checkbutton(sf2, text="Subtract neuropil  coeff:", variable=self.do_neuropil,
                        bg="#141414", fg=TEXT_MAIN, selectcolor="#141414",
                        activebackground="#141414", activeforeground="white",
                        font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(12, 0))
        self.neuropil_coeff = tk.DoubleVar(value=0.7)
        ent2(self.neuropil_coeff, 5); lbl2("   |", dim=True)

        lbl2("Frame rate fs:")
        self.fs_var = tk.DoubleVar(value=30.0)
        ent2(self.fs_var, 6); lbl2("Hz", dim=True)

    def _toggle_settings(self):
        if self._settings_open:
            self._settings_frame.pack_forget()
            self._settings_frame2.pack_forget()
            self._settings_btn.config(text="⚙  Settings")
            self._settings_open = False
        else:
            self._settings_frame.pack(fill=tk.X, after=self._bar)
            self._settings_frame2.pack(fill=tk.X, after=self._settings_frame)
            self._settings_btn.config(text="⚙  Settings ▲")
            self._settings_open = True

    def _build_canvas(self):
        self.fig = Figure(facecolor=BG_DARK)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_splash()

    def _draw_splash(self):
        self.fig.clear()
        ax = self.fig.add_axes([0.15, 0.15, 0.7, 0.7])
        ax.set_facecolor(BG_DARK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=TEXT_DIM)
        x = np.linspace(0, 10, 200)
        floor = np.full_like(x, 0.6)
        rise = 0.6 + np.clip(x - 4, 0, None) ** 1.15 * 0.35
        y = np.maximum(floor, rise)
        ax.plot(x, y, color="white", lw=2.2, alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.12, "Load a Suite2p folder to begin",
                 color=TEXT_DIM, fontsize=11, ha="center", transform=ax.transAxes)
        self.canvas.draw_idle()

    # ── loading ──────────────────────────────────────────────────────────
    def _load_dialog(self):
        folder = filedialog.askdirectory(title="Select Suite2p output folder")
        if not folder:
            return
        plane_dir = find_suite2p_plane(folder)
        if plane_dir is None:
            messagebox.showerror("Not found", "No ops.npy found in that folder "
                                                "(checked root, plane0/1/2/3, combined).")
            return
        try:
            ops = np.load(os.path.join(plane_dir, "ops.npy"), allow_pickle=True).item()
            F = np.load(os.path.join(plane_dir, "F.npy")).astype(float)
            fneu_p = os.path.join(plane_dir, "Fneu.npy")
            Fneu = np.load(fneu_p).astype(float) if os.path.exists(fneu_p) else None
            stat_p = os.path.join(plane_dir, "stat.npy")
            stat = np.load(stat_p, allow_pickle=True) if os.path.exists(stat_p) else None
            iscell_p = os.path.join(plane_dir, "iscell.npy")
            if os.path.exists(iscell_p):
                iscell = np.load(iscell_p)
                mask = iscell[:, 0].astype(bool)
                F = F[mask]
                if Fneu is not None:
                    Fneu = Fneu[mask]
                if stat is not None:
                    stat = stat[mask]
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        self.plane_dir, self.ops, self.F, self.Fneu, self.stat = plane_dir, ops, F, Fneu, stat
        self.movie = None
        self.results = None
        fs = ops.get("fs")
        if fs:
            self.fs_var.set(round(float(fs), 4))
        n_cells, n_frames_traces = F.shape
        self.status_var.set(f"Loaded {os.path.basename(plane_dir)}  ·  {n_cells} cells  ·  "
                              f"fs={self.fs_var.get()} Hz  ·  ready to run analysis")
        self._draw_splash()

    def _ask_raw_dir(self):
        return filedialog.askdirectory(
            title="Locate the raw acquisition TIFF folder (ops paths didn't resolve)")

    # ── analysis (threaded so the GUI doesn't freeze during NormCorre) ─────
    def run_analysis(self):
        if self.plane_dir is None:
            messagebox.showwarning("No data", "Load a Suite2p folder first.")
            return
        self.status_var.set("Locating registered movie...")
        self.root.update_idletasks()
        t = threading.Thread(target=self._run_analysis_worker, daemon=True)
        t.start()

    def _progress(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _run_analysis_worker(self):
        try:
            max_frames = max(50, int(self.max_frames_var.get()))
            pw_rigid = bool(self.pw_rigid_var.get())

            movie = load_registered_movie(
                self.plane_dir, self.ops, max_frames, pw_rigid,
                progress_cb=self._progress, ask_raw_dir=self._ask_raw_dir)

            self._progress(f"Computing PTC from {movie.source} ({movie.shape[0]} frames)...")

            exclude_mask = None
            if self.exclude_roi_var.get() and self.stat is not None:
                Ly, Lx = movie.shape[1], movie.shape[2]
                exclude_mask = roi_exclusion_mask(self.stat, Ly, Lx)

            spatial_bin = max(1, int(self.spatial_bin_var.get()))
            mu_flat, var_flat, mu_bin, var_bin, n_bin = compute_ptc(
                movie.array, spatial_bin=spatial_bin, exclude_mask=exclude_mask)

            fit = fit_gain(mu_bin, var_bin, n_bin,
                            self.fit_lo_var.get(), self.fit_hi_var.get(), self.enf_var.get())

            fs = max(0.1, float(self.fs_var.get()))
            flux = photon_flux_per_cell(self.F, self.Fneu, self.do_neuropil.get(),
                                          self.neuropil_coeff.get(), fit["gain_true"], fs)
            med, sem = bootstrap_median_sem(flux)

            self.results = dict(movie=movie, mu_flat=mu_flat, var_flat=var_flat,
                                 mu_bin=mu_bin, var_bin=var_bin, n_bin=n_bin,
                                 fit=fit, flux=flux, flux_med=med, flux_sem=sem)

            self.root.after(0, self._render_results)
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Analysis failed", msg))
            self.root.after(0, lambda: self.status_var.set("Analysis failed — see error dialog"))

    # ── plotting ─────────────────────────────────────────────────────────
    def _render_results(self):
        r = self.results
        movie, fit = r["movie"], r["fit"]

        self.fig.clear()
        gs = GridSpec(1, 2, figure=self.fig, left=0.07, right=0.97, top=0.90, bottom=0.12,
                      wspace=0.28, width_ratios=[1.3, 1])
        ax_ptc = self.fig.add_subplot(gs[0, 0])
        ax_hist = self.fig.add_subplot(gs[0, 1])

        def style(ax):
            ax.set_facecolor(BG_DARK)
            for s in ax.spines.values(): s.set_color("#555")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(colors=TEXT_MAIN, labelsize=8)

        # ── PTC / hockey-stick panel ────────────────────────────────────
        style(ax_ptc)
        sub = np.random.default_rng(0).choice(len(r["mu_flat"]), size=min(20000, len(r["mu_flat"])), replace=False)
        ax_ptc.scatter(r["mu_flat"][sub], r["var_flat"][sub], s=2, color="#3a3a3a", alpha=0.35, zorder=1)

        mask = fit["mask"]
        ax_ptc.scatter(r["mu_bin"][~mask], r["var_bin"][~mask], s=28, color=C_EXCL, zorder=2, label="excluded")
        ax_ptc.scatter(r["mu_bin"][mask], r["var_bin"][mask], s=28, color=C_INCL, zorder=3, label="fit bins")

        if np.isfinite(fit["slope"]):
            xline = np.array([r["mu_bin"][mask].min(), r["mu_bin"][mask].max()])
            yline = fit["slope"] * xline + fit["intercept"]
            ax_ptc.plot(xline, yline, color=C_FIT, lw=2, zorder=4,
                        label=f"shot-noise fit (slope={fit['slope']:.2f})")
            floor = np.median(r["var_bin"][~mask]) if (~mask).sum() else fit["intercept"]
            ax_ptc.axhline(floor, color=C_FLOOR, lw=1.2, ls="--", alpha=0.8, zorder=2, label="read-noise floor")

        ax_ptc.set_xlabel("Mean (ADU)", color=TEXT_DIM, fontsize=9)
        ax_ptc.set_ylabel("Variance (ADU²)", color=TEXT_DIM, fontsize=9)
        ax_ptc.legend(fontsize=7, facecolor=BG_MID, edgecolor="#555", labelcolor=TEXT_MAIN, loc="upper left")

        gain_true = fit["gain_true"]
        title = (f"Hockey stick PTC  ·  {movie.source}  ·  gain_apparent={fit['slope']:.2f} ADU/ph"
                  if np.isfinite(fit["slope"]) else f"Hockey stick PTC  ·  {movie.source}  ·  fit failed")
        ax_ptc.set_title(title, color=TEXT_BRIGHT, fontsize=10, pad=8)
        if np.isfinite(gain_true):
            ax_ptc.text(0.98, 0.04,
                        f"ENF={self.enf_var.get():.2f}\ngain_true={gain_true:.2f} ADU/photon\nR²={fit['r2']:.3f}",
                        transform=ax_ptc.transAxes, ha="right", va="bottom",
                        color=TEXT_BRIGHT, fontsize=8,
                        bbox=dict(facecolor=BG_MID, edgecolor="#555", boxstyle="round,pad=0.4"))

        # ── photon-flux histogram panel ──────────────────────────────────
        style(ax_hist)
        flux = r["flux"][np.isfinite(r["flux"])]
        if len(flux):
            ax_hist.hist(flux, bins=30, color=C_HIST, alpha=0.75)
            ax_hist.axvline(r["flux_med"], color="white", lw=1.5, ls="--")
            ax_hist.text(0.98, 0.95,
                        f"median = {r['flux_med']:.1f}\n± {r['flux_sem']:.1f} SEM\nn={len(flux)} cells",
                        transform=ax_hist.transAxes, ha="right", va="top",
                        color=TEXT_BRIGHT, fontsize=8,
                        bbox=dict(facecolor=BG_MID, edgecolor="#555", boxstyle="round,pad=0.4"))
        ax_hist.set_xlabel("Photons / cell / s", color=TEXT_DIM, fontsize=9)
        ax_hist.set_ylabel("Cells", color=TEXT_DIM, fontsize=9)
        ax_hist.set_title("Per-cell photon flux", color=TEXT_BRIGHT, fontsize=10, pad=8)

        self.canvas.draw()

        note = movie.note
        self.status_var.set(
            f"Done  ·  source={movie.source}  ({note})  ·  "
            f"gain_true={gain_true:.2f} ADU/ph  ·  median flux={r['flux_med']:.1f}±{r['flux_sem']:.1f} ph/cell/s")


# ── entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    HockeyStickApp(root)
    root.mainloop()
