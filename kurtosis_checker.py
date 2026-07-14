#!/usr/bin/env python3
"""
Kurtosis / Gain Estimation Checker
===================================
Two modes, toggled from the top bar:

  Kurtosis  — three-panel layout (unchanged from the original tool):
              Left   — traces sorted by kurtosis (max 400), best at top
              Centre — zoom-in: top / middle / bottom 10
              Right  — traces sorted by SNR, colored by group

  Gain Estimation — photon-transfer-curve ("hockey stick") analysis on a
              Suite2p output folder. Pixelwise mean/variance is computed
              from a registered movie via frame-differencing (cancels slow
              biological signal, isolates shot noise), binned, and fit to
              recover true gain (ADU/photon) after ENF correction, then
              converted to per-cell photon flux using F.npy.

              Registered-movie source, in priority order:
                1. reg_tif/   — Suite2p's saved registered TIFF stack
                2. raw acquisition TIFFs, motion-corrected on the fly with
                   NoRMCorre (via CaImAn)

              NOTE: Suite2p's own registered binary (data.bin) is
              deliberately NOT used as a fallback — it's a scratch/working
              file that Suite2p can overwrite (e.g. on a subsequent re-run
              or when processing other planes), so it isn't a trustworthy
              record of the exact registered frames that produced F.npy.
              If reg_tif wasn't exported, we re-derive a registered movie
              from the raw TIFFs via NormCorre instead of trusting data.bin.

              IMPORTANT: the registered movie is only ever loaded when you
              click "Run Analysis" in Gain Estimation mode — never on
              folder load, never on tab switch. It can use several GB of
              RAM; a confirmation dialog with a rough estimate is shown
              first, and PTC computation streams frame-pairs in chunks so
              peak memory stays bounded regardless of movie length.

Fall.mat: reads F, Fneu, ops.fs automatically (kurtosis mode).

Dependencies: numpy, scipy, matplotlib, pillow, tifffile
Optional (only for the raw-TIFF NormCorre fallback in gain mode): caiman
    conda install -n base -c conda-forge mamba
    mamba create -n caiman caiman   (CaImAn's compiled deps mean plain pip is not supported)
"""

# Bump this on every change so a running instance's window title can be checked
# against what's actually in this file -- handy when the app runs on a machine
# separate from wherever this source file is being edited.
APP_VERSION = "2026-07-14.8"

import os
import gc
import glob
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from scipy.ndimage import uniform_filter1d
import scipy.io

# ── palette ───────────────────────────────────────────────────────────────────
BG_DARK     = "#0e0e0e"
BG_MID      = "#1a1a1a"
BG_PANEL    = "#2e2e2e"
ACCENT      = "#e94560"
TEXT_DIM    = "#555555"
TEXT_MAIN   = "#aaaaaa"
TEXT_BRIGHT = "#e8e8e8"
C_TOP    = "#27ae60"   # green
C_MID    = "#e67e22"   # orange
C_BOT    = "#c0392b"   # red
C_UNSEL  = "#dddddd"   # near-white for unselected traces
A_UNSEL  = 0.32        # alpha for unselected — more prominent than before
A_SEL    = 0.90        # alpha for selected / highlighted

MAX_SHOW = 400
N_ZOOM   = 10
SP_L     = 0.26    # y-spacing in large panels
SP_Z     = 1.25    # y-spacing in zoom panels
G_COLORS = {"top": C_TOP, "mid": C_MID, "bot": C_BOT}

# gain-mode plot colors
C_FIT    = "#27ae60"   # shot-noise fit line
C_FLOOR  = "#e67e22"   # read-noise floor
C_EXCL   = "#555555"   # excluded bins
C_INCL   = "#3fa7ff"   # included bins
C_HIST   = "#e94560"   # photon-flux histogram

TAB_ACTIVE_BG   = ACCENT
TAB_INACTIVE_BG = "#3a3a3a"


# ═════════════════════════════════════════════════════════════════════════
# Gain-estimation module-level helpers (Suite2p loading, PTC math, NormCorre)
# ═════════════════════════════════════════════════════════════════════════

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


def contiguous_window(n, max_frames):
    """Pick a centered, contiguous [start, end) frame range of length
    min(n, max_frames). PTC's frame-differencing needs genuinely adjacent
    frames, so subsampling must never pick non-consecutive frames."""
    if n <= max_frames:
        return 0, n
    start = max(0, (n - max_frames) // 2)
    return start, start + max_frames


class RegisteredMovie:
    """Uniform (T, Y, X) accessor over reg_tif / NormCorre output.
    Kept in native dtype (usually int16) as long as possible — compute_ptc
    casts to float32 only per-chunk, so this array is the main memory cost.
    """
    def __init__(self, array, source, note=""):
        self.array = array
        self.source = source
        self.note = note
        self.shape = array.shape
        self.dtype = array.dtype

    def sample_indices(self, max_frames):
        # Contiguous window, not evenly-spaced samples: PTC's frame-differencing
        # needs genuinely adjacent frames (consecutive real acquisition frames),
        # so picking every Nth frame across the whole recording would difference
        # frames that are seconds apart and corrupt the noise estimate.
        n = self.shape[0]
        if n <= max_frames:
            return np.arange(n)
        start = max(0, (n - max_frames) // 2)   # centered window
        return np.arange(start, start + max_frames)

    def estimated_bytes(self):
        return int(np.prod(self.shape)) * self.array.itemsize


def _find_reg_tif_files(reg_dir):
    """Locate registered TIFF chunk files inside a reg_tif folder.
    Case-insensitive (Windows network shares can be inconsistent about this),
    prefers chan0-labeled files, falls back to any .tif/.tiff, and finally
    searches one level of subfolders in case of unexpected nesting."""
    import fnmatch
    def _list(patterns, root):
        try:
            entries = os.listdir(root)
        except Exception:
            return []
        hits = [e for e in entries
                if os.path.isfile(os.path.join(root, e))
                and any(fnmatch.fnmatch(e.lower(), p) for p in patterns)]
        return sorted(os.path.join(root, e) for e in hits)

    files = _list(["*chan0*.tif", "*chan0*.tiff"], reg_dir)
    if files:
        return files
    files = _list(["*.tif", "*.tiff"], reg_dir)
    if files:
        return files
    # Last resort: recurse in case chunks live under an unexpected subfolder.
    hits = (glob.glob(os.path.join(reg_dir, "**", "*.tif"), recursive=True) +
            glob.glob(os.path.join(reg_dir, "**", "*.tiff"), recursive=True) +
            glob.glob(os.path.join(reg_dir, "**", "*.TIF"), recursive=True) +
            glob.glob(os.path.join(reg_dir, "**", "*.TIFF"), recursive=True))
    return sorted(set(hits))


def find_suite2p_plane(folder, max_depth=6, max_dirs_scanned=20000):
    """Recursively search downstream of `folder` for Suite2p plane output
    (any directory containing ops.npy) — deliberately not a fixed candidate
    list like ["plane0", "combined", ...], since real setups nest output
    under arbitrary custom folder names (e.g. a "small" subfolder for a
    downsampled/test run) that no fixed list can anticipate.

    If the tree contains more than one ops.npy (e.g. an older full-resolution
    run sitting alongside a newer downsampled one), directories with a
    *usable* reg_tif export are preferred over ones without — otherwise this
    silently picks a plane with cell traces but no registered movie, and
    everything downstream ends up on the slow raw-TIFF/NormCorre path for no
    good reason. Ties fall back to whichever match is shallowest.
    """
    folder = os.path.abspath(folder)
    root_depth = folder.rstrip(os.sep).count(os.sep)
    candidates = []
    dirs_scanned = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        dirs_scanned += 1
        if dirs_scanned > max_dirs_scanned:
            break
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []   # stop descending past the depth cap
            continue
        if "ops.npy" in filenames:
            candidates.append(dirpath)
            dirnames[:] = []  # plane dirs don't nest further plane output

    if not candidates:
        return None

    def _depth(d):
        return d.rstrip(os.sep).count(os.sep)

    def _has_reg_tif(d):
        reg_dir = os.path.join(d, "reg_tif")
        return os.path.isdir(reg_dir) and len(_find_reg_tif_files(reg_dir)) > 0

    with_reg_tif = sorted((c for c in candidates if _has_reg_tif(c)), key=_depth)
    if with_reg_tif:
        return with_reg_tif[0]
    return sorted(candidates, key=_depth)[0]


def load_reg_tif(plane_dir, max_frames):
    """Priority 1: Suite2p's exported registered TIFF stack. Suite2p usually
    splits this into many smaller chunk files (file000_chan0.tif,
    file001_chan0.tif, ... one per registration batch), so this only decodes
    the chunk files that actually overlap the contiguous frame window we
    need — it never loads the whole recording into memory just to discard
    most of it."""
    try:
        import tifffile
    except ImportError:
        return None

    reg_dir = os.path.join(plane_dir, "reg_tif")
    if not os.path.isdir(reg_dir):
        return None  # legitimately absent -> caller tries the next fallback tier

    files = _find_reg_tif_files(reg_dir)
    if not files:
        # The reg_tif folder IS there -- silently falling back to a slow raw-TIFF
        # NormCorre pass here would be surprising and waste time, so fail loudly
        # with enough detail to diagnose a naming-convention mismatch instead.
        try:
            listing = os.listdir(reg_dir)
        except Exception:
            listing = []
        raise RuntimeError(
            f"Found a reg_tif folder ({reg_dir}) but no .tif/.tiff files inside it "
            f"(checked recursively, case-insensitive). Contents: "
            f"{listing[:20]}{' ...' if len(listing) > 20 else ''}"
        )

    # Probe each chunk's frame count without decoding any pixel data.
    counts = []
    for f in files:
        with tifffile.TiffFile(f) as tf:
            counts.append(len(tf.pages))
    total = int(sum(counts))
    offsets = np.concatenate([[0], np.cumsum(counts)])

    start, end = contiguous_window(total, max_frames)

    # Only decode the chunk files that overlap [start, end).
    file_lo = int(np.searchsorted(offsets, start, side="right") - 1)
    file_hi = int(np.searchsorted(offsets, end - 1, side="right") - 1)

    chunks = []
    for fi in range(file_lo, file_hi + 1):
        f_start = int(offsets[fi])
        local_lo = max(0, start - f_start)
        local_hi = min(counts[fi], end - f_start)
        chunk = tifffile.imread(files[fi], key=range(local_lo, local_hi))
        if chunk.ndim == 2:
            chunk = chunk[None]
        chunks.append(chunk)

    arr = np.concatenate(chunks, axis=0)
    n_files_read = file_hi - file_lo + 1
    return RegisteredMovie(arr, "reg_tif",
                            f"{len(files)} chunk file(s) total ({total} frames), "
                            f"read {n_files_read} of them for frames [{start}:{end})")


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
            "CaImAn's compiled dependencies mean mamba/conda is the supported install "
            "path (plain pip is not):\n"
            "  conda install -n base -c conda-forge mamba\n"
            "  mamba create -n caiman caiman\n"
            "  conda activate caiman\n\n"
            "If you have a reg_tif/ export instead, you don't need CaImAn at all — "
            "double-check the status bar after Load Data confirms "
            "'reg_tif found (N file(s))' before running the analysis.\n\n"
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


def save_reg_tif_chunks(arr, reg_dir, chunk_size=500, progress_cb=None):
    """Write a registered movie out as reg_tif-style chunk files
    (file000_chan0.tif, file001_chan0.tif, ...) matching Suite2p's own
    naming convention, so a future run picks it up automatically via
    load_reg_tif without re-running motion correction."""
    import tifffile
    os.makedirs(reg_dir, exist_ok=True)
    arr_i16 = np.clip(np.round(arr), -32768, 32767).astype(np.int16)
    T = arr_i16.shape[0]
    n_files = 0
    for i, start in enumerate(range(0, T, chunk_size)):
        chunk = arr_i16[start:start + chunk_size]
        tifffile.imwrite(os.path.join(reg_dir, f"file{i:03d}_chan0.tif"), chunk)
        n_files += 1
    if progress_cb:
        progress_cb(f"Saved motion-corrected movie: {n_files} file(s) in {reg_dir}")
    return n_files


def _normcorre_then_maybe_save(tiff_paths, max_frames, pw_rigid, save_mc, save_dir, progress_cb=None):
    """Run NormCorre, optionally save the full result as reg_tif chunks
    (saved BEFORE the max_frames window is applied, so the complete
    registered movie is preserved for next time even if this run only
    analyzed a subset of it), then trim to the analysis window."""
    arr = run_normcorre(tiff_paths, pw_rigid, progress_cb=progress_cb)
    note_suffix = ""
    if save_mc:
        try:
            reg_dir = os.path.join(save_dir, "reg_tif")
            n = save_reg_tif_chunks(arr, reg_dir, progress_cb=progress_cb)
            note_suffix = f"; saved {n} chunk file(s) to {reg_dir} for next time"
        except Exception as e:
            note_suffix = f"; WARNING: failed to save motion-corrected TIFF ({e})"
    if arr.shape[0] > max_frames:
        start, end = contiguous_window(arr.shape[0], max_frames)
        arr = arr[start:end]
    return arr, note_suffix


def probe_tiff_shape(paths):
    """Peek at TIFF page metadata (no pixel data decoded) to get (Ly, Lx, total_frames).
    Used for the manual-TIFF fallback (no Suite2p output found) so we can show a
    memory estimate before actually reading any frames."""
    import tifffile
    total = 0
    Ly = Lx = None
    for p in paths:
        with tifffile.TiffFile(p) as tf:
            n = len(tf.pages)
            if Ly is None:
                Ly, Lx = tf.pages[0].shape[:2]
            total += n
    return Ly, Lx, total


def load_tiffs_direct(paths, max_frames):
    """Read raw/already-registered TIFF frames as-is, no motion correction.
    Like load_reg_tif, only decodes the chunk files overlapping a contiguous
    frame window rather than reading everything then subsampling."""
    import tifffile
    counts = []
    for p in paths:
        with tifffile.TiffFile(p) as tf:
            counts.append(len(tf.pages))
    total = int(sum(counts))
    offsets = np.concatenate([[0], np.cumsum(counts)])

    start, end = contiguous_window(total, max_frames)
    file_lo = int(np.searchsorted(offsets, start, side="right") - 1)
    file_hi = int(np.searchsorted(offsets, end - 1, side="right") - 1)

    chunks = []
    for fi in range(file_lo, file_hi + 1):
        f_start = int(offsets[fi])
        local_lo = max(0, start - f_start)
        local_hi = min(counts[fi], end - f_start)
        chunk = tifffile.imread(paths[fi], key=range(local_lo, local_hi))
        if chunk.ndim == 2:
            chunk = chunk[None]
        chunks.append(chunk)

    arr = np.concatenate(chunks, axis=0)
    return RegisteredMovie(arr, "raw_tiff",
                            f"{len(paths)} file(s) total ({total} frames), "
                            f"frames [{start}:{end}) read, NOT motion-corrected")


def load_registered_movie(plane_dir, ops, max_frames, pw_rigid, progress_cb=None,
                           ask_raw_dir=None, save_mc=False):
    """Try reg_tif -> raw tiff + NormCorre, in order.

    Suite2p's own data.bin is intentionally never used here — it's a
    scratch file Suite2p can overwrite on a later run, so it isn't a
    reliable record of the frames that produced the loaded F.npy.

    Note: if a reg_tif/ folder exists but contains no readable TIFFs,
    load_reg_tif raises rather than silently falling through to NormCorre —
    an unexpected empty/misnamed reg_tif folder should surface as an error,
    not a silent multi-minute detour through motion correction.

    If save_mc, a NormCorre result is written back out as reg_tif chunks in
    plane_dir/reg_tif/, so the next run on this same plane finds it via
    load_reg_tif and skips motion correction entirely.
    """
    mov = load_reg_tif(plane_dir, max_frames)
    if mov is not None:
        return mov

    tiff_paths = _resolve_raw_tiffs(plane_dir, ops)
    if not tiff_paths and ask_raw_dir is not None:
        d = ask_raw_dir()
        if d:
            tiff_paths = sorted(glob.glob(os.path.join(d, "*.tif")) + glob.glob(os.path.join(d, "*.tiff")))
    if not tiff_paths:
        raise RuntimeError(
            "No reg_tif export was found, and the raw acquisition TIFFs could not be "
            "located automatically (ops paths don't resolve on this machine). "
            "Suite2p's data.bin is deliberately not used as a fallback since it can be "
            "overwritten by later runs."
        )

    arr, note_suffix = _normcorre_then_maybe_save(
        tiff_paths, max_frames, pw_rigid, save_mc, plane_dir, progress_cb=progress_cb)
    return RegisteredMovie(arr, "normcorre",
                            f"{len(tiff_paths)} raw TIFF(s), {arr.shape[0]} frames after "
                            f"motion correction{note_suffix}")


def load_registered_movie_manual(raw_tiffs, max_frames, pw_rigid, skip_mc, save_mc, progress_cb=None):
    """Manual-TIFF-mode equivalent of load_registered_movie (no Suite2p ops.npy
    available). Checks for a reg_tif/ folder next to the source TIFFs first —
    including one saved by an earlier run of this same tool — before running
    NormCorre."""
    source_dir = os.path.dirname(raw_tiffs[0])
    mov = load_reg_tif(source_dir, max_frames)
    if mov is not None:
        return mov

    if skip_mc:
        return load_tiffs_direct(raw_tiffs, max_frames)

    arr, note_suffix = _normcorre_then_maybe_save(
        raw_tiffs, max_frames, pw_rigid, save_mc, source_dir, progress_cb=progress_cb)
    return RegisteredMovie(arr, "normcorre",
                            f"{len(raw_tiffs)} raw TIFF(s), {arr.shape[0]} frames after "
                            f"motion correction{note_suffix}")


def roi_exclusion_mask(stat, Ly, Lx):
    """Boolean mask, True = background/non-cell pixel."""
    mask = np.ones((Ly, Lx), dtype=bool)
    for cell in stat:
        yp = np.asarray(cell["ypix"]); xp = np.asarray(cell["xpix"])
        valid = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
        mask[yp[valid], xp[valid]] = False
    return mask


def footprints_to_raw_traces(movie_arr, A, dims, threshold=0.2, min_pixels=9):
    """Convert CNMF spatial footprints into raw pixel-sum traces read directly
    from the movie -- deliberately NOT CNMF's own C trace. CNMF's C is in the
    factorization's internal scale (A is typically close to unit-norm per
    component), not raw ADU, so using it directly would silently break the
    gain/photon-flux conversion, which expects F to be a sum of raw ADU over
    the ROI mask (matching Suite2p's F.npy convention).

    A is (Ly*Lx, n_components), flattened in Fortran order to match CaImAn's
    own convention for reshaping (Ly, Lx) footprints into columns.

    Returns (F, npix): F is (n_components_kept, T), npix is (n_components_kept,).
    Components thresholded down to fewer than min_pixels are dropped.
    """
    Ly, Lx = dims
    A_dense = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
    n_components = A_dense.shape[1]
    T = movie_arr.shape[0]
    # Fortran-order per-frame flatten, to match A's column layout
    flat_movie = movie_arr.transpose(0, 2, 1).reshape(T, Ly * Lx).astype(np.float64)

    F_list, npix_list = [], []
    for i in range(n_components):
        comp = A_dense[:, i]
        peak = comp.max()
        if peak <= 0:
            continue
        mask = comp >= (threshold * peak)
        n = int(mask.sum())
        if n < min_pixels:
            continue
        F_list.append(flat_movie[:, mask].sum(axis=1))
        npix_list.append(n)

    if not F_list:
        return np.zeros((0, T)), np.zeros((0,), dtype=int)
    return np.array(F_list), np.array(npix_list, dtype=int)


def run_cnmf_segmentation(movie_arr, fs, gsig, progress_cb=None):
    """Run CaImAn's CNMF purely for ROI/footprint detection on an already
    motion-corrected movie, then re-derive raw pixel-sum traces from those
    footprints via footprints_to_raw_traces (see its docstring for why).

    NOTE: unlike the rest of this tool's gain-mode pipeline, this specific
    function has not been exercised against a real CaImAn install (not
    available in the dev/test environment this tool was built in) -- treat
    it as a first-pass implementation and sanity-check the extracted traces
    against your data (e.g. do the footprints look like real cells, not
    scrambled or off-target) before trusting the resulting photon flux.
    """
    try:
        from caiman.source_extraction.cnmf import CNMF
        from caiman.source_extraction.cnmf.params import CNMFParams
    except ImportError as e:
        raise RuntimeError(
            "CaImAn is not installed, so CNMF segmentation can't run.\n\n"
            "  conda install -n base -c conda-forge mamba\n"
            "  mamba create -n caiman caiman\n"
            "  conda activate caiman\n\n"
            f"(underlying error: {e})"
        )

    if progress_cb:
        progress_cb("Running CaImAn CNMF segmentation...")

    T, Ly, Lx = movie_arr.shape
    images = movie_arr.astype(np.float32)

    opts = CNMFParams(params_dict=dict(
        fr=fs, decay_time=0.4,
        gSig=[gsig, gsig], p=1, nb=2,
        merge_thr=0.85,
        rf=None,  # whole-FOV, no patches -- simpler and adequate for the
                  # frame-windowed movies this tool works with
    ))
    cnm = CNMF(n_processes=1, dview=None, params=opts)
    cnm = cnm.fit(images)

    n_found = cnm.estimates.A.shape[1]
    if progress_cb:
        progress_cb(f"CNMF found {n_found} candidate component(s); extracting raw traces...")

    F, npix = footprints_to_raw_traces(movie_arr, cnm.estimates.A, (Ly, Lx),
                                        min_pixels=max(4, int(np.pi * (gsig / 2) ** 2 * 0.3)))
    return F, npix


def compute_ptc(movie_arr, spatial_bin=1, exclude_mask=None, n_mean_bins=40, chunk=200):
    """Frame-differencing mean/variance PTC estimate, streamed in chunks so
    peak memory stays ~O(chunk x Y x X) instead of O(T x Y x X).

    For consecutive frame pairs (i, i+1): mean = (f_i+f_{i+1})/2,
    var = (f_i - f_{i+1})^2 / 2  (unbiased shot-noise variance estimator;
    slow biological/structural signal cancels in the difference).

    Returns per-pixel arrays (mu_px, var_px) and binned arrays
    (mu_bin, var_bin, n_bin) for plotting/fitting.
    """
    T = movie_arr.shape[0]
    if T < 2:
        raise RuntimeError("Need at least 2 frames for a PTC estimate.")

    Y, X = movie_arr.shape[1], movie_arr.shape[2]
    if spatial_bin > 1:
        Yb, Xb = Y // spatial_bin, X // spatial_bin
    else:
        Yb, Xb = Y, X

    sum_mu = np.zeros((Yb, Xb), dtype=np.float64)
    sum_var = np.zeros((Yb, Xb), dtype=np.float64)
    n_pairs = 0

    i = 0
    while i < T - 1:
        j = min(i + chunk, T - 1)
        block = np.asarray(movie_arr[i:j + 1]).astype(np.float32)
        if spatial_bin > 1:
            block = block[:, :Yb * spatial_bin, :Xb * spatial_bin]
            # sum (not mean) across the binned pixels: for independent shot-noise
            # pixels, summing keeps slope(var vs mean) == per-pixel gain invariant
            # to the bin size; averaging would deflate the recovered gain by bin^2.
            block = block.reshape(block.shape[0], Yb, spatial_bin, Xb, spatial_bin).sum(axis=(2, 4))

        f0, f1 = block[:-1], block[1:]
        diff = f0 - f1
        meanb = (f0 + f1) * 0.5
        sum_mu += meanb.sum(axis=0)
        sum_var += (diff * diff * 0.5).sum(axis=0)
        n_pairs += (j - i)
        del block, f0, f1, diff, meanb
        i = j

    if n_pairs == 0:
        raise RuntimeError("No frame pairs available for PTC estimate.")

    mu_px = sum_mu / n_pairs
    var_px = sum_var / n_pairs

    if exclude_mask is not None:
        m = exclude_mask
        if m.shape != (Yb, Xb) and spatial_bin > 1:
            m = m[:Yb * spatial_bin:spatial_bin, :Xb * spatial_bin:spatial_bin][:Yb, :Xb]
        keep = m
    else:
        keep = np.ones((Yb, Xb), dtype=bool)

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

    fit_pct_lo/fit_pct_hi default to 0/98: unlike a camera sensor, a GaAsP
    PMT has no strong reason to have a read-noise-dominated floor eating
    into a large fraction of the dynamic range, so by default nothing is
    excluded except a saturation guard at the very top. Narrowing fit_pct_lo
    above 0 is only warranted if the plot actually shows a flattening at
    low mean intensity for your specific setup.

    Returns dict with slope (apparent gain), gain_true (ENF-corrected),
    intercept, r2, mask (bins used in the fit), and low_mask/high_mask
    (bins excluded at the low/high end respectively, tracked separately so
    a "read-noise floor" can only ever be drawn from genuinely low-end-
    excluded bins, not conflated with high-end saturation exclusions).
    """
    n = len(mu_bin)
    if n < 3:
        empty = np.zeros(n, dtype=bool)
        return dict(slope=np.nan, gain_true=np.nan, intercept=np.nan, r2=np.nan,
                     mask=empty, low_mask=empty, high_mask=empty)
    lo, hi = np.percentile(mu_bin, [fit_pct_lo, fit_pct_hi])
    low_mask = mu_bin < lo
    high_mask = mu_bin > hi
    mask = ~low_mask & ~high_mask
    if mask.sum() < 3:
        mask = np.ones(n, dtype=bool)
        low_mask = np.zeros(n, dtype=bool)
        high_mask = np.zeros(n, dtype=bool)
    slope, intercept, r2 = weighted_linfit(mu_bin[mask], var_bin[mask], n_bin[mask])
    gain_true = slope / (enf ** 2) if slope and np.isfinite(slope) else np.nan
    return dict(slope=slope, gain_true=gain_true, intercept=intercept, r2=r2,
                 mask=mask, low_mask=low_mask, high_mask=high_mask)


def photon_flux_per_cell(F, gain_true, fs):
    """Photons/cell/s from raw F only — no neuropil/background subtraction.
    Matches the Wilt convention: F0 is defined to include all detected
    photons (signal + background), so subtracting Fneu here would be
    inconsistent with that definition."""
    if not np.isfinite(gain_true) or gain_true <= 0:
        return np.full(F.shape[0], np.nan)
    photons_per_frame = F.astype(float).mean(axis=1) / gain_true
    return photons_per_frame * fs


# ═════════════════════════════════════════════════════════════════════════
# Main app
# ═════════════════════════════════════════════════════════════════════════

class KurtosisChecker:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Kurtosis / Gain Estimation Checker  —  build {APP_VERSION}")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1760x990")
        self._set_icon()

        # kurtosis-mode state
        self.F    = None
        self.Fneu = None
        self.data_type = None
        self.mode = tk.StringVar(value="kurtosis")   # sort-by metric within kurtosis tab

        # gain-mode state
        self.g_plane_dir = None
        self.g_ops  = None
        self.g_F    = None
        self.g_Fneu = None
        self.g_stat = None
        self.g_results = None   # lightweight dict — never holds the raw movie array

        # gain-mode state: manual-TIFF fallback (no Suite2p output found)
        self.g_raw_tiffs = None       # list of paths, or None if using Suite2p
        self.g_raw_shape = None       # (Ly, Lx, total_frames) probed at load time

        self.tab_var = tk.StringVar(value="kurtosis")  # "kurtosis" | "gain"

        self._build_controls()
        self._build_canvas()

    @staticmethod
    def _build_icon_image(size=512, rounded=True):
        """Generate the Kurtosis app icon — 3-trace bold style (logo2)."""
        K_X_STEM, K_X_ARM = 200, 352
        K_Y_TOP, K_Y_BOTTOM = 96, 420
        K_Y_VERTEX = 288
        K_Y_ARM_TOP, K_Y_ARM_BOTTOM = 214, 420
        ROW_COUNT = 3
        BAND_X0, BAND_X1 = 84, 428
        BAND_Y0, BAND_Y1 = 168, 352
        SAMPLES = 220
        MAX_AMP = 92
        RISE_PX, DECAY_PX = 10, 34
        MINT = (245, 247, 250)

        def k_crossings(y):
            xs = []
            if K_Y_TOP <= y <= K_Y_BOTTOM:
                xs.append(K_X_STEM)
            if K_Y_ARM_TOP <= y <= K_Y_VERTEX:
                t = (K_Y_VERTEX - y) / (K_Y_VERTEX - K_Y_ARM_TOP)
                xs.append(K_X_STEM + t * (K_X_ARM - K_X_STEM))
            if K_Y_VERTEX <= y <= K_Y_ARM_BOTTOM:
                t = (y - K_Y_VERTEX) / (K_Y_ARM_BOTTOM - K_Y_VERTEX)
                xs.append(K_X_STEM + t * (K_X_ARM - K_X_STEM))
            return xs

        step = (BAND_Y1 - BAND_Y0) / (ROW_COUNT - 1)

        SCALE = 2
        W = H = size * SCALE
        RX = min(104 * SCALE, W // 2 - 1)

        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if rounded:
            ImageDraw.Draw(img).rounded_rectangle([0, 0, W - 1, H - 1], radius=RX,
                                                   fill=(13, 18, 24, 255))

        for r in range(ROW_COUNT):
            row_y = BAND_Y0 + r * step
            is_last = (r == ROW_COUNT - 1)
            if is_last:
                events = [BAND_X0 + (BAND_X1 - BAND_X0) * 0.42]
            else:
                events = k_crossings(row_y)
            pts = []
            for s in range(SAMPLES):
                t = s / (SAMPLES - 1)
                x = BAND_X0 + t * (BAND_X1 - BAND_X0)
                deflect = 0.0
                for ex in events:
                    d = x - ex
                    k = max(0.0, 1 + d / RISE_PX) if d < 0 else math.exp(-d / DECAY_PX)
                    deflect += k
                deflect = min(deflect, 1.2) * MAX_AMP
                wobble = math.sin(x * 0.04 + r * 1.9) * 1.4 + math.sin(x * 0.1 + r * 0.5) * 0.7
                pts.append((x * SCALE, (row_y - deflect - wobble) * SCALE))

            alpha = 255

            # Glow pass
            glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(glow).line(pts, fill=(*MINT, int(alpha * 0.7)), width=6)
            glow = glow.filter(ImageFilter.GaussianBlur(radius=6 * SCALE / 2))
            img = Image.alpha_composite(img, glow)

            # Crisp line
            line = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(line).line(pts, fill=(*MINT, alpha), width=int(4.5 * SCALE))
            img = Image.alpha_composite(img, line)

        if rounded:
            # Border
            border = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(border).rounded_rectangle([1, 1, W - 2, H - 2], radius=RX,
                                                      outline=(255, 255, 255, 15), width=2)
            img = Image.alpha_composite(img, border)
            # Rounded-rect mask
            mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], radius=RX, fill=255)
            img.putalpha(mask)

        return img.resize((size, size), Image.LANCZOS)

    def _set_icon(self):
        """Load icon.png (15-row sparse version) and set as window icon."""
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
            if os.path.exists(icon_path):
                toolbar_img = Image.open(icon_path).convert("RGBA")
            else:
                toolbar_img = self._build_icon_image(512)
            self._icon_img = ImageTk.PhotoImage(toolbar_img)
            self.root.wm_iconphoto(True, self._icon_img)
        except Exception:
            pass  # icon is cosmetic — never crash over it


    # ── controls ──────────────────────────────────────────────────────────

    def _build_controls(self):
        # ── top bar — 3-column grid so the tab toggle sits centered ────────
        self._bar = tk.Frame(self.root, bg=BG_MID, pady=6)
        self._bar.pack(fill=tk.X)
        self._bar.columnconfigure(0, weight=1)
        self._bar.columnconfigure(1, weight=0)
        self._bar.columnconfigure(2, weight=1)

        left = tk.Frame(self._bar, bg=BG_MID)
        left.grid(row=0, column=0, sticky="w")
        center = tk.Frame(self._bar, bg=BG_MID)
        center.grid(row=0, column=1)
        right = tk.Frame(self._bar, bg=BG_MID)
        right.grid(row=0, column=2, sticky="e")

        def _lbtn(parent, text, command, bg, fg="white", font_size=12, bold=True, side=tk.LEFT, padx=6):
            """Label-based button — colour always renders on macOS."""
            weight = "bold" if bold else "normal"
            lbl = tk.Label(parent, text=text, bg=bg, fg=fg,
                           font=("Helvetica", font_size, weight),
                           padx=16, pady=7, cursor="hand2", relief=tk.RAISED, bd=2)
            lbl.pack(side=side, padx=padx, pady=5)
            lbl.bind("<Button-1>", lambda e: command())
            lbl.bind("<Enter>",    lambda e: lbl.config(relief=tk.SUNKEN))
            lbl.bind("<Leave>",    lambda e: lbl.config(relief=tk.RAISED))
            return lbl

        _lbtn(left, "📂  Load Data", self._load_dispatch, bg="#27ae60", padx=(10, 6))

        self._settings_open = False
        self._settings_btn = _lbtn(left, "⚙  Settings", self._toggle_settings, bg="#3a3a3a", bold=False)

        # ── center tab toggle ────────────────────────────────────────────
        self._tab_kurtosis_btn = _lbtn(center, "Kurtosis", lambda: self._switch_tab("kurtosis"),
                                        bg=TAB_ACTIVE_BG, font_size=11, padx=(0, 2))
        self._tab_gain_btn = _lbtn(center, "Gain Estimation", lambda: self._switch_tab("gain"),
                                    bg=TAB_INACTIVE_BG, font_size=11, padx=(2, 0))

        self._action_btn = _lbtn(right, "▶  Plot", self._on_action_click,
                                  bg="#c0392b", font_size=13, side=tk.RIGHT, padx=10)

        # "Sort by:" toggle — kurtosis-tab only
        self._sort_frame = tk.Frame(left, bg=BG_MID)
        self._sort_frame.pack(side=tk.LEFT, padx=(16, 0))
        tk.Label(self._sort_frame, text="Sort by:", bg=BG_MID, fg=TEXT_MAIN,
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(0, 6))
        for val, txt in (("kurtosis", "Kurtosis"), ("snr", "SNR")):
            tk.Radiobutton(self._sort_frame, text=txt, variable=self.mode, value=val,
                           bg=BG_MID, fg=TEXT_BRIGHT, selectcolor=BG_MID,
                           activebackground=BG_MID, font=("Helvetica", 11),
                           command=self._replot_if_loaded).pack(side=tk.LEFT, padx=4)

        # ── status row (its own line so it never unbalances tab centering) ─
        status_row = tk.Frame(self.root, bg=BG_MID)
        status_row.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="No data loaded")
        tk.Label(status_row, textvariable=self.status_var, bg=BG_MID, fg=TEXT_DIM,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=14, pady=(0, 4))

        # ── settings panels (hidden by default) ─────────────────────────
        self._settings_shared = tk.Frame(self.root, bg="#141414", pady=6)
        self._settings_kurtosis = tk.Frame(self.root, bg="#141414", pady=6)
        self._settings_gain = tk.Frame(self.root, bg="#141414", pady=6)
        self._build_settings_shared(self._settings_shared)
        self._build_settings_kurtosis(self._settings_kurtosis)
        self._build_settings_gain(self._settings_gain)

    def _build_settings_shared(self, sf):
        def lbl(t, dim=False, parent=sf):
            tk.Label(parent, text=t, bg="#141414",
                     fg=TEXT_DIM if dim else TEXT_MAIN,
                     font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 2))

        def ent(var, w=6, parent=sf):
            e = tk.Entry(parent, textvariable=var, width=w,
                         bg=BG_PANEL, fg="white", insertbackground="white",
                         relief=tk.FLAT, font=("Helvetica", 10))
            e.pack(side=tk.LEFT)
            return e

        lbl("Frame rate fs:")
        self.fs_var = tk.DoubleVar(value=30.0)
        ent(self.fs_var, 6); lbl("Hz", dim=True)

    def _build_settings_kurtosis(self, sf):
        def lbl(t, dim=False, parent=sf):
            tk.Label(parent, text=t, bg="#141414",
                     fg=TEXT_DIM if dim else TEXT_MAIN,
                     font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 2))

        def ent(var, w=5, parent=sf):
            e = tk.Entry(parent, textvariable=var, width=w,
                         bg=BG_PANEL, fg="white", insertbackground="white",
                         relief=tk.FLAT, font=("Helvetica", 10))
            e.pack(side=tk.LEFT)
            return e

        self.do_neuropil = tk.BooleanVar(value=True)
        tk.Checkbutton(sf, text="Subtract neuropil  coeff:",
                       variable=self.do_neuropil, bg="#141414", fg=TEXT_MAIN,
                       selectcolor="#141414", activebackground="#141414",
                       activeforeground="white",
                       font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 0))
        self.neuropil_coeff = tk.DoubleVar(value=0.7)
        ent(self.neuropil_coeff, 5); lbl("   |", dim=True)

        lbl("Boxcar:")
        self.boxcar_sec = tk.DoubleVar(value=0.5)
        ent(self.boxcar_sec, 5); lbl("s", dim=True); lbl("   |", dim=True)

        self.do_dff = tk.BooleanVar(value=False)
        tk.Checkbutton(sf, text="   ΔF/F₀  pct:",
                       variable=self.do_dff, bg="#141414", fg=TEXT_MAIN,
                       selectcolor="#141414", activebackground="#141414",
                       activeforeground="white",
                       font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.dff_pct = tk.IntVar(value=8)
        ent(self.dff_pct, 3)

    def _build_settings_gain(self, sf):
        def lbl(t, dim=False, parent=sf):
            tk.Label(parent, text=t, bg="#141414",
                     fg=TEXT_DIM if dim else TEXT_MAIN,
                     font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 2))

        def ent(var, w=6, parent=sf):
            e = tk.Entry(parent, textvariable=var, width=w,
                         bg=BG_PANEL, fg="white", insertbackground="white",
                         relief=tk.FLAT, font=("Helvetica", 10))
            e.pack(side=tk.LEFT)
            return e

        lbl("   ENF:")
        self.enf_var = tk.DoubleVar(value=1.2)
        ent(self.enf_var, 5)

        lbl("   |", dim=True); lbl("Fit range (pct of mean):")
        self.fit_lo_var = tk.DoubleVar(value=0.0)
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
        tk.Checkbutton(sf, text="Exclude cell ROIs",
                        variable=self.exclude_roi_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 0))

        lbl("   |", dim=True)
        self.pw_rigid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sf, text="NormCorre: piecewise-rigid",
                        variable=self.pw_rigid_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 0))

        lbl("   |", dim=True)
        self.skip_mc_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sf, text="Skip motion correction (manual TIFF already registered)",
                        variable=self.skip_mc_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 0))

        lbl("   |", dim=True)
        self.save_mc_var = tk.BooleanVar(value=True)
        tk.Checkbutton(sf, text="Save motion-corrected TIFF (reg_tif)",
                        variable=self.save_mc_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 0))

        lbl("   |", dim=True); lbl("CNMF cell radius (px):")
        self.cnmf_gsig_var = tk.IntVar(value=6)
        ent(self.cnmf_gsig_var, 3)

    def _toggle_settings(self):
        if self._settings_open:
            self._settings_shared.pack_forget()
            self._settings_kurtosis.pack_forget()
            self._settings_gain.pack_forget()
            self._settings_btn.config(text="⚙  Settings")
            self._settings_open = False
        else:
            self._settings_shared.pack(fill=tk.X, after=self._bar)
            if self.tab_var.get() == "kurtosis":
                self._settings_kurtosis.pack(fill=tk.X, after=self._settings_shared)
            else:
                self._settings_gain.pack(fill=tk.X, after=self._settings_shared)
            self._settings_btn.config(text="⚙  Settings ▲")
            self._settings_open = True

    # ── tab switching ────────────────────────────────────────────────────

    def _switch_tab(self, new_tab):
        if new_tab == self.tab_var.get():
            return
        self.tab_var.set(new_tab)

        if new_tab == "kurtosis":
            self._tab_kurtosis_btn.config(bg=TAB_ACTIVE_BG)
            self._tab_gain_btn.config(bg=TAB_INACTIVE_BG)
            self._action_btn.config(text="▶  Plot")
            self._sort_frame.pack(side=tk.LEFT, padx=(16, 0))
        else:
            self._tab_kurtosis_btn.config(bg=TAB_INACTIVE_BG)
            self._tab_gain_btn.config(bg=TAB_ACTIVE_BG)
            self._action_btn.config(text="▶  Run Analysis")
            self._sort_frame.pack_forget()

        if self._settings_open:
            self._settings_kurtosis.pack_forget()
            self._settings_gain.pack_forget()
            if new_tab == "kurtosis":
                self._settings_kurtosis.pack(fill=tk.X, after=self._settings_shared)
            else:
                self._settings_gain.pack(fill=tk.X, after=self._settings_shared)

        if new_tab == "kurtosis":
            if self.F is not None:
                self.plot()
            else:
                self._draw_splash()
            self.status_var.set(
                self.status_var.get() if self.F is not None else "No data loaded — Kurtosis mode")
        else:
            if self.g_results is not None:
                self._render_gain_results()
            else:
                self._draw_gain_splash()
            if self.g_plane_dir is None and not self.g_raw_tiffs:
                self.status_var.set("No data loaded — Gain Estimation mode "
                                     "(registered movie is only loaded when you click Run Analysis)")

    def _load_dispatch(self):
        if self.tab_var.get() == "kurtosis":
            self._load_folder()
        else:
            self._load_gain_folder()

    def _on_action_click(self):
        if self.tab_var.get() == "kurtosis":
            self.plot()
        else:
            self.run_gain_analysis()

    def _replot_if_loaded(self):
        if self.F is not None:
            self.plot()

    # ── kurtosis-mode loader (unchanged logic) ──────────────────────────

    def _load_folder(self):
        """Scan folder; prefer .npy (fastest), fall back to .mat."""
        folder = filedialog.askdirectory(
            title="Navigate to the folder containing your .mat or .npy file")
        if not folder:
            return

        # Search order: root, then plane0/1/2/combined sub-dirs
        search = [folder] + [os.path.join(folder, d)
                              for d in ("plane0", "plane1", "plane2", "combined")]

        # Priority 1: Suite2p F.npy (fastest — raw numpy)
        for d in search:
            f_path = os.path.join(d, "F.npy")
            if os.path.exists(f_path):
                fneu_p = os.path.join(d, "Fneu.npy")
                ops_p  = os.path.join(d, "ops.npy")
                try:
                    F_raw    = np.load(f_path).astype(float)
                    Fneu_raw = np.load(fneu_p).astype(float) if os.path.exists(fneu_p) else None
                    if os.path.exists(ops_p):
                        ops = np.load(ops_p, allow_pickle=True).item()
                        fs  = self._try_fs(ops)
                        if fs: self.fs_var.set(round(fs, 4))
                    # Filter by iscell if available
                    iscell_p = os.path.join(d, "iscell.npy")
                    cell_mask = None
                    if os.path.exists(iscell_p):
                        iscell = np.load(iscell_p)
                        cell_mask = iscell[:, 0].astype(bool)
                        F_raw = F_raw[cell_mask]
                        if Fneu_raw is not None:
                            Fneu_raw = Fneu_raw[cell_mask]
                    self.F    = F_raw
                    self.Fneu = Fneu_raw
                except Exception as e:
                    messagebox.showerror("Error", str(e)); return
                self.data_type = "suite2p"
                n, t = self.F.shape
                note = " + Fneu" if self.Fneu is not None else ""
                cell_note = f"  [iscell: {n}]" if cell_mask is not None else ""
                self.status_var.set(
                    f"Suite2p: {n} cells × {t} frames{note}{cell_note}  ·  fs={self.fs_var.get()} Hz"
                    f"  [{os.path.relpath(f_path, folder)}]")
                self.plot(); return

        # Priority 2: any .npy at root level
        npy_files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        if npy_files:
            path = os.path.join(folder, npy_files[0])
            try:
                arr = np.load(path, allow_pickle=False).astype(float)
            except Exception as e:
                messagebox.showerror("Error", str(e)); return
            arr = self._orient(arr, npy_files[0])
            if arr is None: return
            self.F = arr; self.Fneu = None; self.data_type = "npy"
            self.status_var.set(
                f"NumPy: {arr.shape[0]} cells × {arr.shape[1]} frames  ·  {npy_files[0]}")
            self.plot(); return

        # Priority 3: Fall.mat or first .mat
        mat_pref = ["Fall.mat"] + [f for f in os.listdir(folder) if f.endswith(".mat")]
        for fname in mat_pref:
            path = os.path.join(folder, fname)
            if not os.path.exists(path): continue
            result = self._read_mat(path)
            if result is None: return
            F, Fneu, fs = result
            arr = self._orient(F, fname)
            if arr is None: return
            self.F = arr; self.Fneu = Fneu; self.data_type = "mat"
            if fs is not None: self.fs_var.set(round(float(fs), 4))
            n, t = self.F.shape
            note = " + Fneu" if self.Fneu is not None else ""
            self.status_var.set(
                f"MATLAB: {n} cells × {t} frames{note}  ·  fs={self.fs_var.get()} Hz  ·  {fname}")
            self.plot(); return

        messagebox.showerror("Nothing found",
                             "No F.npy, .npy, or .mat files found in that folder.")

    def _load_file(self):
        """Pick a single file; dispatch by extension."""
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("NumPy / MATLAB", "*.npy *.mat"),
                       ("NumPy", "*.npy"),
                       ("MATLAB", "*.mat"),
                       ("All", "*.*")])
        if not path: return
        if path.lower().endswith(".npy"):
            self.load_npy(path)
        else:
            self.load_mat(path)

    def _build_canvas(self):
        self.fig = Figure(facecolor=BG_DARK)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # No toolbar — removed intentionally
        self._draw_splash()

    def _draw_splash(self):
        """Draw the K-trace logo as matplotlib lines — no image, no hard edges."""
        self.fig.clear()
        ax = self.fig.add_axes([0.20, 0.15, 0.60, 0.60])
        ax.set_facecolor(BG_DARK)
        ax.set_aspect("equal")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Original 27-row dense logo
        K_X_STEM, K_X_ARM = 200, 352
        K_Y_TOP, K_Y_BOTTOM = 96, 420
        K_Y_VERTEX = 288
        K_Y_ARM_TOP, K_Y_ARM_BOTTOM = 214, 420
        ROW_COUNT = 27
        BAND_X0, BAND_X1 = 62, 452
        BAND_Y0, BAND_Y1 = 78, 444
        SAMPLES = 160
        MAX_AMP, RISE_PX, DECAY_PX = 10, 6, 20

        def k_crossings(y):
            xs = []
            if K_Y_TOP <= y <= K_Y_BOTTOM:
                xs.append(K_X_STEM)
            if K_Y_ARM_TOP <= y <= K_Y_VERTEX:
                t = (K_Y_VERTEX - y) / (K_Y_VERTEX - K_Y_ARM_TOP)
                xs.append(K_X_STEM + t * (K_X_ARM - K_X_STEM))
            if K_Y_VERTEX <= y <= K_Y_ARM_BOTTOM:
                t = (y - K_Y_VERTEX) / (K_Y_ARM_BOTTOM - K_Y_VERTEX)
                xs.append(K_X_STEM + t * (K_X_ARM - K_X_STEM))
            return xs

        step = (BAND_Y1 - BAND_Y0) / (ROW_COUNT - 1)
        row_amp = min(MAX_AMP, step * 0.9)

        for r in range(ROW_COUNT):
            row_y = BAND_Y0 + r * step
            events = k_crossings(row_y)
            xs, ys = [], []
            for s in range(SAMPLES):
                t = s / (SAMPLES - 1)
                x = BAND_X0 + t * (BAND_X1 - BAND_X0)
                deflect = 0.0
                for ex in events:
                    d = x - ex
                    k = max(0.0, 1 + d / RISE_PX) if d < 0 else math.exp(-d / DECAY_PX)
                    deflect += k
                deflect = min(deflect, 1.2) * row_amp
                wobble = math.sin(x * 0.045 + r * 1.9) * 0.9 + math.sin(x * 0.12 + r * 0.5) * 0.45
                xs.append(x)
                ys.append(-(row_y - deflect - wobble))

            depth = r / (ROW_COUNT - 1)
            alpha = 0.5 + 0.42 * (1 - abs(depth - 0.5) * 1.1)
            ax.plot(xs, ys, color="white", lw=0.9, alpha=alpha, solid_capstyle="round")

        span = max(BAND_X1 - BAND_X0, BAND_Y1 - BAND_Y0)
        cx = (BAND_X0 + BAND_X1) / 2
        cy = -(BAND_Y0 + BAND_Y1) / 2
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy - span / 2, cy + span / 2)
        self.canvas.draw_idle()
        self._splash_active = True

    def _draw_gain_splash(self):
        """Simple hockey-stick line as the gain-tab splash."""
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
        ax.text(0.5, -0.12, "Load a Suite2p folder, then click Run Analysis",
                 color=TEXT_DIM, fontsize=11, ha="center", transform=ax.transAxes)
        self.canvas.draw_idle()

    # ── loaders (kurtosis-mode extras, unchanged) ───────────────────────

    def _try_fs(self, ops_obj):
        if ops_obj is None:
            return None
        try:
            if isinstance(ops_obj, dict):
                return float(ops_obj.get("fs", 30.0))
            if hasattr(ops_obj, "item"):
                d = ops_obj.item()
                if isinstance(d, dict):
                    return float(d.get("fs", 30.0))
            return float(np.asarray(ops_obj["fs"]).flat[0])
        except Exception:
            return None

    def load_suite2p(self):
        folder = filedialog.askdirectory(title="Select Suite2p output folder")
        if not folder:
            return
        search = [folder] + [os.path.join(folder, d)
                              for d in ("plane0", "plane1", "plane2", "combined")]
        f_path = fneu_path = ops_path = None
        for d in search:
            cand = os.path.join(d, "F.npy")
            if os.path.exists(cand):
                f_path = cand
                fp = os.path.join(d, "Fneu.npy"); op = os.path.join(d, "ops.npy")
                if os.path.exists(fp): fneu_path = fp
                if os.path.exists(op): ops_path  = op
                break
        if f_path is None:
            messagebox.showerror("Not found", "F.npy not found."); return
        try:
            self.F    = np.load(f_path).astype(float)
            self.Fneu = np.load(fneu_path).astype(float) if fneu_path else None
            if ops_path:
                ops = np.load(ops_path, allow_pickle=True).item()
                fs  = self._try_fs(ops)
                if fs: self.fs_var.set(round(fs, 4))
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self.data_type = "suite2p"
        n, t = self.F.shape
        note = " + Fneu" if self.Fneu is not None else ""
        self.status_var.set(
            f"Suite2p: {n} cells × {t} frames{note}  ·  fs = {self.fs_var.get()} Hz")

    def load_npy(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="NumPy file",
                                              filetypes=[("NumPy", "*.npy"), ("All", "*.*")])
        if not path: return
        try:
            arr = np.load(path, allow_pickle=False).astype(float)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        arr = self._orient(arr, os.path.basename(path))
        if arr is None: return
        self.F = arr; self.Fneu = None; self.data_type = "npy"
        self.status_var.set(f"NumPy: {arr.shape[0]} cells × {arr.shape[1]} frames")

    def load_mat(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="MATLAB file",
                                              filetypes=[("MATLAB", "*.mat"), ("All", "*.*")])
        if not path: return
        result = self._read_mat(path)
        if result is None: return
        F, Fneu, fs = result
        arr = self._orient(F, os.path.basename(path))
        if arr is None: return
        self.F = arr; self.Fneu = Fneu; self.data_type = "mat"
        if fs is not None: self.fs_var.set(round(float(fs), 4))
        n, t = self.F.shape
        note = " + Fneu" if self.Fneu is not None else ""
        self.status_var.set(
            f"MATLAB: {n} cells × {t} frames{note}  ·  fs = {self.fs_var.get()} Hz")

    def _read_mat(self, path):
        def parse(mat, hdf5=False):
            def to2d(k):
                if k not in mat: return None
                try:
                    v = np.array(mat[k]).astype(float)
                    if hdf5: v = v.T
                    return v if v.ndim == 2 else None
                except Exception:
                    return None

            F = None
            for k in ("F", "f", "traces", "data"):
                F = to2d(k)
                if F is not None: break
            if F is None:
                keys = [k for k in mat
                        if not k.startswith("_") and not k.startswith("#")]
                two_d = [k for k in keys if to2d(k) is not None]
                if len(two_d) == 1:
                    F = to2d(two_d[0])
                elif two_d:
                    ch = self._pick_variable(two_d)
                    if ch is None: return None
                    F = to2d(ch)
                else:
                    messagebox.showerror("Error", "No 2-D variable found."); return None

            Fneu = None
            for k in ("Fneu", "fneu", "Fneuropil"):
                v = to2d(k)
                if v is not None: Fneu = v; break

            fs = None
            if "ops" in mat:
                try:
                    ops = mat["ops"]
                    if hdf5:
                        fs = float(np.array(ops["fs"]).flat[0])
                    else:
                        if hasattr(ops, "dtype") and ops.dtype.names:
                            fs = float(ops["fs"].flat[0])
                        elif hasattr(ops, "item"):
                            d = ops.item()
                            if isinstance(d, dict): fs = float(d.get("fs", 30.0))
                        else:
                            fs = float(np.asarray(ops[0, 0]["fs"]).flat[0])
                except Exception:
                    pass

            return F, Fneu, fs

        try:
            mat = scipy.io.loadmat(path)
            return parse(mat, hdf5=False)
        except NotImplementedError:
            pass
        except Exception as e:
            messagebox.showerror("Error", str(e)); return None

        try:
            import h5py
        except ImportError:
            messagebox.showerror("Missing package",
                                 "MATLAB v7.3 needs h5py:\n  pip install h5py")
            return None
        try:
            with h5py.File(path, "r") as f:
                return parse(f, hdf5=True)
        except Exception as e:
            messagebox.showerror("HDF5 error", str(e)); return None

    def _orient(self, arr, name):
        if arr.ndim != 2:
            messagebox.showerror("Shape error",
                                 f"{name}: expected 2-D, got {arr.shape}."); return None
        r, c = arr.shape
        if r == c:
            if not messagebox.askyesno("Square",
                                       f"{name} is {r}×{c}.\nYes → rows = cells"):
                arr = arr.T
        elif r > c:
            if messagebox.askyesno("Orientation",
                                   f"{name}: {r}×{c}, more rows than columns.\n"
                                   "Transpose (rows = frames)?"):
                arr = arr.T
        return arr

    def _pick_variable(self, keys):
        win = tk.Toplevel(self.root)
        win.title("Select variable"); win.configure(bg=BG_DARK); win.grab_set()
        tk.Label(win, text="Choose variable:", bg=BG_DARK, fg=TEXT_BRIGHT,
                 padx=16, pady=10).pack()
        choice = tk.StringVar(value=keys[0])
        for k in keys:
            tk.Radiobutton(win, text=k, variable=choice, value=k,
                           bg=BG_DARK, fg=TEXT_MAIN, selectcolor=BG_PANEL,
                           activebackground=BG_DARK).pack(anchor=tk.W, padx=24)
        result = [None]
        def ok(): result[0] = choice.get(); win.destroy()
        tk.Button(win, text="OK", command=ok, bg=BG_PANEL, fg="white",
                  relief=tk.FLAT, padx=14, pady=5).pack(pady=12)
        self.root.wait_window(win)
        return result[0]

    # ── kurtosis plotting (unchanged) ────────────────────────────────────

    def plot(self):
        if self.F is None:
            messagebox.showwarning("No data", "Load data first."); return

        self._splash_active = False

        # Pre-process
        F = self.F.copy().astype(float)
        if (self.data_type in ("suite2p", "mat")
                and self.do_neuropil.get() and self.Fneu is not None):
            F -= self.neuropil_coeff.get() * self.Fneu
        if self.do_dff.get():
            pct = float(np.clip(self.dff_pct.get(), 1, 99))
            f0  = np.percentile(F, pct, axis=1, keepdims=True)
            f0  = np.where(np.abs(f0) < 1e-6, 1e-6, f0)
            F   = (F - f0) / np.abs(f0)
        fs  = max(0.1, float(self.fs_var.get()))
        sec = max(0.0, float(self.boxcar_sec.get()))
        bc  = max(1, round(sec * fs))
        # Metrics computed on unfiltered traces
        n_cells, n_frames = F.shape
        kurt = stats.kurtosis(F, axis=1, fisher=False)   # Pearson; normal ≈ 3
        snr  = (F.max(axis=1) - F.mean(axis=1)) / (F.std(axis=1) + 1e-12)

        # Boxcar filter applied only for display
        if bc > 1:
            F = uniform_filter1d(F, size=bc, axis=1)

        # Pick a shared pool of cells so both panels show the same population.
        # Subsample evenly from all cell indices (metric-agnostic), then sort
        # that same pool by each metric independently.
        if n_cells <= MAX_SHOW:
            pool = np.arange(n_cells)
        else:
            pool = np.round(np.linspace(0, n_cells - 1, MAX_SHOW)).astype(int)

        kurt_disp = pool[np.argsort(kurt[pool])[::-1]]
        snr_disp  = pool[np.argsort(snr[pool])[::-1]]

        mode = self.mode.get()
        prim  = kurt_disp if mode == "kurtosis" else snr_disp
        n_p   = len(prim)
        mid_s = n_p // 2 - N_ZOOM // 2
        groups = {
            "top": prim[:N_ZOOM],
            "mid": prim[mid_s: mid_s + N_ZOOM],
            "bot": prim[-N_ZOOM:],
        }
        cell_color = {}
        for g, cells in groups.items():
            for c in cells: cell_color[c] = G_COLORS[g]

        # ── figure ────────────────────────────────────────────────────────────
        self.fig.clear()
        gs = GridSpec(3, 3, figure=self.fig,
                      left=0.055, right=0.975, top=0.96, bottom=0.05,
                      wspace=0.22, hspace=0.32,
                      width_ratios=[1.4, 1.4, 1.4])

        ax_l  = self.fig.add_subplot(gs[:, 0])
        ax_r  = self.fig.add_subplot(gs[:, 2])
        ax_zt = self.fig.add_subplot(gs[0, 1])
        ax_zm = self.fig.add_subplot(gs[1, 1])
        ax_zb = self.fig.add_subplot(gs[2, 1])

        t = np.arange(n_frames)

        # ── style helper ──────────────────────────────────────────────────────
        def style(ax, right_spine=False):
            ax.set_facecolor(BG_DARK)
            for s in ax.spines.values(): s.set_color("#555")
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_color("#555")
            ax.spines["bottom"].set_color("#555")
            ax.spines["right"].set_visible(right_spine)
            if right_spine: ax.spines["right"].set_color("#555")
            ax.tick_params(colors=TEXT_MAIN, labelsize=6.5)

        def normed(trace, height=1.0):
            lo, hi = trace.min(), trace.max()
            rng = hi - lo
            return (trace - lo) / rng * height if rng > 0 else np.zeros_like(trace)

        # ── large stacked panels ──────────────────────────────────────────────

        def draw_large(ax, displayed, metric_vals, metric_label, is_source):
            style(ax)
            n = len(displayed)

            # Highlight bands (source panel only)
            if is_source:
                rank_map = {c: i for i, c in enumerate(displayed)}
                for g, cells in groups.items():
                    rows = sorted(rank_map[c] for c in cells if c in rank_map)
                    if not rows: continue
                    col  = G_COLORS[g]
                    y_lo = (n - 1 - max(rows)) * SP_L - 0.04
                    y_hi = (n - 1 - min(rows)) * SP_L + SP_L * 0.85 + 0.04
                    ax.axhspan(y_lo, y_hi, color=col, alpha=0.10, zorder=0)
                    ax.axhspan(y_lo, y_hi, xmin=0.990, xmax=1.0,
                               color=col, alpha=1.0, clip_on=False, zorder=6)

            # Traces — normalized to 85% of spacing to avoid harsh overlap
            h = SP_L * 0.85
            ytick_step = max(1, n // 20)
            ytick_pos, ytick_lbl = [], []
            for row_i, cidx in enumerate(displayed):
                tn     = normed(F[cidx], height=h)
                offset = (n - 1 - row_i) * SP_L
                col    = cell_color.get(cidx, C_UNSEL)
                sel    = cidx in cell_color
                ax.plot(t, tn + offset,
                        lw=0.75 if sel else 0.30,
                        color=col,
                        alpha=A_SEL if sel else A_UNSEL,
                        zorder=3 if sel else 1)
                if row_i % ytick_step == 0:
                    ytick_pos.append(offset + h * 0.5)
                    ytick_lbl.append(f"{metric_vals[cidx]:.1f}")

            ax.set_yticks(ytick_pos)
            ax.set_yticklabels(ytick_lbl, fontsize=6.5, color=TEXT_MAIN)
            ax.set_ylabel(metric_label, color=TEXT_MAIN, fontsize=9, labelpad=2)
            ax.set_xlim(0, n_frames - 1)
            ax.set_ylim(-SP_L * 0.5, n * SP_L)
            ax.set_xlabel("Frame", color=TEXT_DIM, fontsize=8)

            tag = "◀ groups defined here" if is_source else "◀ colors projected here"
            ax.set_title(
                f"Sorted by {metric_label}  ({n} / {n_cells})  {tag}",
                color=TEXT_BRIGHT, fontsize=9, pad=5, fontweight="bold")

        l_is_src = (mode == "kurtosis")
        draw_large(ax_l, kurt_disp, kurt, "κ",   is_source=l_is_src)
        draw_large(ax_r, snr_disp,  snr,  "SNR", is_source=not l_is_src)

        # ── zoom panels (dual y-axes) ─────────────────────────────────────────

        src_lbl = "κ" if mode == "kurtosis" else "SNR"
        zoom_meta = {
            "top": (ax_zt, f"Top {N_ZOOM}  {src_lbl} ↑",    C_TOP),
            "mid": (ax_zm, f"Middle {N_ZOOM}",                C_MID),
            "bot": (ax_zb, f"Bottom {N_ZOOM}  {src_lbl} ↓",  C_BOT),
        }

        for g, (ax, title, col) in zoom_meta.items():
            style(ax, right_spine=True)
            gcells   = groups[g]           # best → worst
            ordered  = gcells[::-1]        # worst → best (bottom → top in plot)
            h_z      = SP_Z * 0.85

            ytick_pos   = []
            kurt_labels = []
            snr_labels  = []

            for row_i, cidx in enumerate(ordered):
                tn     = normed(F[cidx], height=h_z)
                offset = row_i * SP_Z
                ax.plot(t, tn + offset, lw=0.85, color=col, alpha=0.92)
                ytick_pos.append(offset + h_z * 0.5)
                kurt_labels.append(f"{kurt[cidx]:.1f}")
                snr_labels.append(f"{snr[cidx]:.1f}")

            y_lo = -SP_Z * 0.3
            y_hi = N_ZOOM * SP_Z

            # Left y-axis = κ
            ax.set_yticks(ytick_pos)
            ax.set_yticklabels(kurt_labels, fontsize=6.5, color=TEXT_MAIN)
            ax.set_ylabel("κ", color=TEXT_MAIN, fontsize=8, labelpad=2)
            ax.set_ylim(y_lo, y_hi)

            # Right y-axis = SNR (twin)
            ax2 = ax.twinx()
            ax2.set_facecolor("none")
            ax2.spines["right"].set_color("#555")
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.tick_params(colors=TEXT_DIM, labelsize=6.5)
            ax2.set_yticks(ytick_pos)
            ax2.set_yticklabels(snr_labels, fontsize=6.5, color=TEXT_DIM)
            ax2.set_ylabel("SNR", color=TEXT_DIM, fontsize=8, labelpad=2)
            ax2.set_ylim(y_lo, y_hi)

            ax.set_xlim(0, n_frames - 1)
            ax.set_title(title, color=col, fontsize=9, pad=3, fontweight="bold")
            ax.tick_params(axis="x",
                           colors=TEXT_DIM if g != "bot" else TEXT_MAIN,
                           labelsize=6 if g != "bot" else 6.5)
            if g == "bot":
                ax.set_xlabel("Frame", color=TEXT_DIM, fontsize=8)
            else:
                ax.set_xticklabels([])

        self.canvas.draw()

    # ── gain-mode: loading (never touches the registered movie) ─────────

    def _load_gain_folder(self):
        folder = filedialog.askdirectory(title="Select Suite2p output folder")
        if not folder:
            return
        plane_dir = find_suite2p_plane(folder)
        if plane_dir is None:
            self._load_gain_manual_tiff(folder)
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

        self.g_plane_dir, self.g_ops, self.g_F, self.g_Fneu, self.g_stat = \
            plane_dir, ops, F, Fneu, stat
        self.g_raw_tiffs = None   # clear any previous manual-TIFF selection
        self.g_raw_shape = None
        self.g_results = None
        fs = ops.get("fs")
        if fs:
            self.fs_var.set(round(float(fs), 4))
        n_cells = F.shape[0]
        reg_dir = os.path.join(plane_dir, "reg_tif")
        # Use the exact same file-matching logic Run Analysis will use, so this
        # hint can never say "found" while Run Analysis then falls back anyway.
        n_reg_files = len(_find_reg_tif_files(reg_dir)) if os.path.isdir(reg_dir) else 0
        src_hint = (f"reg_tif found ({n_reg_files} file(s))" if n_reg_files else
                    "no usable reg_tif — will need NormCorre on raw TIFFs")
        self.status_var.set(f"Loaded: {plane_dir}  ·  {n_cells} cells  ·  "
                              f"fs={self.fs_var.get()} Hz  ·  {src_hint}  ·  click Run Analysis")
        self._draw_gain_splash()

    def _load_gain_manual_tiff(self, folder):
        """No Suite2p ops.npy found — offer to run the PTC gain estimate directly
        on user-selected TIFF(s) instead. No F.npy means no per-cell photon-flux
        panel, but the hockey-stick gain fit itself doesn't need Suite2p at all."""
        proceed = messagebox.askyesno(
            "No Suite2p output found",
            f"No ops.npy found in that folder (checked root, plane0/1/2/3, combined).\n\n"
            f"Select TIFF file(s) to run the PTC gain estimate on directly?\n"
            f"(Without Suite2p's F.npy, only the gain estimate is available — "
            f"per-cell photon flux needs cell traces.)")
        if not proceed:
            return
        paths = filedialog.askopenfilenames(
            title="Select TIFF(s) to analyze", initialdir=folder,
            filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")])
        if not paths:
            return
        try:
            Ly, Lx, total_frames = probe_tiff_shape(paths)
        except ImportError:
            messagebox.showerror("Missing package", "tifffile is required:\n  pip install tifffile")
            return
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        self.g_plane_dir, self.g_ops, self.g_F, self.g_Fneu, self.g_stat = None, None, None, None, None
        self.g_raw_tiffs = list(paths)
        self.g_raw_shape = (Ly, Lx, total_frames)
        self.g_results = None
        self.status_var.set(
            f"Loaded {len(paths)} TIFF(s)  ·  {Ly}x{Lx} px  ·  {total_frames} frames  ·  "
            f"no Suite2p F.npy — gain estimate only  ·  verify fs in Settings  ·  click Run Analysis")
        self._draw_gain_splash()

    def _ask_raw_dir(self):
        return filedialog.askdirectory(
            title="Locate the raw acquisition TIFF folder (ops paths didn't resolve)")

    # ── gain-mode: analysis (threaded; this is the only place the movie loads) ──

    def run_gain_analysis(self):
        if self.g_plane_dir is None and not self.g_raw_tiffs:
            messagebox.showwarning("No data", "Load a Suite2p folder (or select TIFFs) first.")
            return

        max_frames = max(50, int(self.max_frames_var.get()))
        if self.g_plane_dir is not None:
            Ly, Lx = self.g_ops.get("Ly"), self.g_ops.get("Lx")
        else:
            Ly, Lx, _ = self.g_raw_shape
        frame_bytes = Ly * Lx * 4  # worst case: float32
        est_gb = (frame_bytes * max_frames) / 1e9
        if est_gb > 1.5:
            proceed = messagebox.askyesno(
                "Memory check",
                f"Loading up to {max_frames} frames at {Lx}x{Ly} "
                f"px may use roughly {est_gb:.1f} GB of RAM (more if NormCorre motion "
                f"correction is needed). The movie is only loaded now, not before.\n\n"
                f"Continue? (Lower 'Max frames' or raise 'Spatial bin' in Settings to reduce this.)")
            if not proceed:
                return

        self.status_var.set("Locating registered movie...")
        self.root.update_idletasks()
        t = threading.Thread(target=self._run_gain_worker, daemon=True)
        t.start()

    def _gain_progress(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _ask_yesno_blocking(self, title, message):
        """Show a yes/no dialog from this background worker thread and block
        until answered. tkinter dialogs must run on the main thread, so this
        marshals the call via root.after and waits on an Event for the result."""
        result = {}
        event = threading.Event()
        def _ask():
            result["answer"] = messagebox.askyesno(title, message)
            event.set()
        self.root.after(0, _ask)
        event.wait()
        return result.get("answer", False)

    def _run_gain_worker(self):
        try:
            max_frames = max(50, int(self.max_frames_var.get()))
            pw_rigid = bool(self.pw_rigid_var.get())
            save_mc = bool(self.save_mc_var.get())

            if self.g_plane_dir is not None:
                movie = load_registered_movie(
                    self.g_plane_dir, self.g_ops, max_frames, pw_rigid,
                    progress_cb=self._gain_progress, ask_raw_dir=self._ask_raw_dir,
                    save_mc=save_mc)
            else:
                movie = load_registered_movie_manual(
                    self.g_raw_tiffs, max_frames, pw_rigid,
                    skip_mc=bool(self.skip_mc_var.get()), save_mc=save_mc,
                    progress_cb=self._gain_progress)

            self._gain_progress(f"Computing PTC from {movie.source} ({movie.shape[0]} frames)...")

            exclude_mask = None
            if self.exclude_roi_var.get() and self.g_stat is not None:
                Ly, Lx = movie.shape[1], movie.shape[2]
                exclude_mask = roi_exclusion_mask(self.g_stat, Ly, Lx)

            spatial_bin = max(1, int(self.spatial_bin_var.get()))
            mu_flat, var_flat, mu_bin, var_bin, n_bin = compute_ptc(
                movie.array, spatial_bin=spatial_bin, exclude_mask=exclude_mask)

            fit = fit_gain(mu_bin, var_bin, n_bin,
                            self.fit_lo_var.get(), self.fit_hi_var.get(), self.enf_var.get())

            # Cell traces: existing Suite2p F.npy, or -- only when motion
            # correction just genuinely ran and there's no F.npy at all, so the
            # flux panel would otherwise be empty -- offer CaImAn CNMF.
            F_for_flux = self.g_F
            cnmf_used = False
            if F_for_flux is None and movie.source == "normcorre":
                want_cnmf = self._ask_yesno_blocking(
                    "Run cell segmentation?",
                    "Motion correction just ran and there's no Suite2p F.npy, so the "
                    "photon-flux panel would otherwise stay empty.\n\n"
                    "Run CaImAn CNMF on the registered movie to detect cells and "
                    "extract traces? (First-pass integration — sanity-check the "
                    "resulting footprints/traces against your data before trusting "
                    "the photon flux numbers.)")
                if want_cnmf:
                    gsig = max(2, int(self.cnmf_gsig_var.get()))
                    fs_for_cnmf = max(0.1, float(self.fs_var.get()))
                    F_cnmf, npix_cnmf = run_cnmf_segmentation(
                        movie.array, fs_for_cnmf, gsig, progress_cb=self._gain_progress)
                    if F_cnmf.shape[0] > 0:
                        F_for_flux = F_cnmf
                        cnmf_used = True
                    else:
                        self._gain_progress("CNMF found no usable components — "
                                             "continuing without a flux panel.")

            # capture lightweight metadata, then drop the (possibly huge) movie array
            movie_source, movie_note, movie_shape = movie.source, movie.note, movie.shape
            del movie
            gc.collect()

            has_cells = F_for_flux is not None and len(F_for_flux) > 0
            if has_cells:
                fs = max(0.1, float(self.fs_var.get()))
                flux = photon_flux_per_cell(F_for_flux, fit["gain_true"], fs)
                med, sem = bootstrap_median_sem(flux)
            else:
                flux, med, sem = None, np.nan, np.nan

            self.g_results = dict(movie_source=movie_source, movie_note=movie_note,
                                   movie_shape=movie_shape, has_cells=has_cells,
                                   cnmf_used=cnmf_used, n_cells=(len(F_for_flux) if has_cells else 0),
                                   mu_flat=mu_flat, var_flat=var_flat,
                                   mu_bin=mu_bin, var_bin=var_bin, n_bin=n_bin,
                                   fit=fit, flux=flux, flux_med=med, flux_sem=sem)

            self.root.after(0, self._render_gain_results)
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Analysis failed", msg))
            self.root.after(0, lambda: self.status_var.set("Analysis failed — see error dialog"))

    # ── gain-mode: plotting ──────────────────────────────────────────────

    def _render_gain_results(self):
        r = self.g_results
        fit = r["fit"]

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
            # Only draw a "read-noise floor" line when bins were actually excluded
            # at the LOW end (fit_pct_lo > 0) -- otherwise there's nothing to draw
            # a floor from, and conflating it with high-end saturation-excluded
            # bins would mislabel them.
            low_mask = fit.get("low_mask")
            if low_mask is not None and low_mask.sum():
                floor = np.median(r["var_bin"][low_mask])
                ax_ptc.axhline(floor, color=C_FLOOR, lw=1.2, ls="--", alpha=0.8,
                                zorder=2, label="read-noise floor (excluded low bins)")

        ax_ptc.set_xlabel("Mean (ADU)", color=TEXT_DIM, fontsize=9)
        ax_ptc.set_ylabel("Variance (ADU²)", color=TEXT_DIM, fontsize=9)
        ax_ptc.legend(fontsize=7, facecolor=BG_MID, edgecolor="#555", labelcolor=TEXT_MAIN, loc="upper left")

        gain_true = fit["gain_true"]
        title = (f"Hockey stick PTC  ·  {r['movie_source']}  ·  gain_apparent={fit['slope']:.2f} ADU/ph"
                  if np.isfinite(fit["slope"]) else f"Hockey stick PTC  ·  {r['movie_source']}  ·  fit failed")
        ax_ptc.set_title(title, color=TEXT_BRIGHT, fontsize=10, pad=8)
        if np.isfinite(gain_true):
            ax_ptc.text(0.98, 0.04,
                        f"ENF={self.enf_var.get():.2f}\ngain_true={gain_true:.2f} ADU/photon\nR²={fit['r2']:.3f}",
                        transform=ax_ptc.transAxes, ha="right", va="bottom",
                        color=TEXT_BRIGHT, fontsize=8,
                        bbox=dict(facecolor=BG_MID, edgecolor="#555", boxstyle="round,pad=0.4"))

        # ── photon-flux histogram panel (needs Suite2p F.npy) ─────────────
        style(ax_hist)
        if not r.get("has_cells", True):
            ax_hist.axis("off")
            ax_hist.text(0.5, 0.5, "No Suite2p F.npy loaded\n(manual TIFF mode)\n\n"
                                    "PTC gain estimate only —\nper-cell photon flux unavailable",
                        transform=ax_hist.transAxes, ha="center", va="center",
                        color=TEXT_DIM, fontsize=9)
        else:
            flux = r["flux"][np.isfinite(r["flux"])]
            cell_src = "CaImAn CNMF (unverified — sanity-check!)" if r.get("cnmf_used") else "Suite2p F.npy"
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
            ax_hist.set_title(f"Per-cell photon flux  ·  cells via {cell_src}",
                               color=(C_HIST if r.get("cnmf_used") else TEXT_BRIGHT), fontsize=10, pad=8)

        self.canvas.draw()

        if r.get("has_cells", True):
            cell_src = "CNMF" if r.get("cnmf_used") else "Suite2p"
            self.status_var.set(
                f"Done  ·  source={r['movie_source']}  ({r['movie_note']})  ·  "
                f"gain_true={gain_true:.2f} ADU/ph  ·  cells via {cell_src}  ·  "
                f"median flux={r['flux_med']:.1f}±{r['flux_sem']:.1f} ph/cell/s")
        else:
            self.status_var.set(
                f"Done  ·  source={r['movie_source']}  ({r['movie_note']})  ·  "
                f"gain_true={gain_true:.2f} ADU/ph  ·  no F.npy — flux not computed")


# ── entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    KurtosisChecker(root)
    root.mainloop()
