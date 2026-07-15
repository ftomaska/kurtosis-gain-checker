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
APP_VERSION = "2026-07-15.8"

import os
import gc
import glob
import logging
import random
import time
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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


def contiguous_window(n, max_frames, start=None):
    """Pick a contiguous [start, end) frame range of length min(n, max_frames).
    PTC's frame-differencing needs genuinely adjacent frames, so subsampling
    must never pick non-consecutive frames.

    If `start` is None (default), the window is centered automatically --
    the original behavior. If `start` is given (an explicit frame index
    from the user's "Start frame" setting), that's used instead, clamped
    so the window still fits within [0, n)."""
    length = min(n, max_frames)
    if start is None:
        s = max(0, (n - length) // 2)
    else:
        s = max(0, min(int(start), max(0, n - length)))
    return s, s + length


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


def load_reg_tif(plane_dir, max_frames, frame_start=None):
    """Priority 1: Suite2p's exported registered TIFF stack. Suite2p usually
    splits this into many smaller chunk files (file000_chan0.tif,
    file001_chan0.tif, ... one per registration batch), so this only decodes
    the chunk files that actually overlap the contiguous frame window we
    need — it never loads the whole recording into memory just to discard
    most of it. frame_start (optional): explicit starting frame index from
    the user's "Start frame" setting; None means auto-centered (default)."""
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

    start, end = contiguous_window(total, max_frames, start=frame_start)

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


class _ProgressLogHandler(logging.Handler):
    """Forwards CaImAn's own logging output to the app's status bar/progress
    callback. CaImAn reports its internal progress via the standard `logging`
    module (its own demo notebooks tell you to set logging.INFO to watch
    progress) rather than exposing a progress-percent callback, so listening
    on the logger is the documented, stable way to surface real sub-step
    text (e.g. per-chunk/per-file messages) instead of a single static
    "running..." string. Throttled so a chatty logger can't flood the Tk
    event queue with after() calls."""

    def __init__(self, progress_cb, min_interval=0.2):
        super().__init__(level=logging.INFO)
        self.progress_cb = progress_cb
        self.min_interval = min_interval
        self._last = 0.0

    def emit(self, record):
        now = time.monotonic()
        if now - self._last < self.min_interval:
            return
        self._last = now
        try:
            msg = record.getMessage()
        except Exception:
            return
        if msg:
            self.progress_cb(f"NormCorre: {msg}")


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

    log_handler = None
    caiman_logger = logging.getLogger("caiman")
    prev_level = caiman_logger.level
    if progress_cb:
        log_handler = _ProgressLogHandler(progress_cb)
        caiman_logger.addHandler(log_handler)
        caiman_logger.setLevel(logging.INFO)

    try:
        mc = MotionCorrect(
            tiff_paths, dview=None,
            max_shifts=(6, 6), niter_rig=1,
            strides=(48, 48), overlaps=(24, 24),
            max_deviation_rigid=3, shifts_opencv=True,
            nonneg_movie=True, border_nan="copy",
            pw_rigid=pw_rigid,
        )
        mc.motion_correct(save_movie=True)
    finally:
        if log_handler is not None:
            caiman_logger.removeHandler(log_handler)
            caiman_logger.setLevel(prev_level)

    # mc.fname_tot_rig / mc.fname_tot_els point at CaImAn's per-tiff-file
    # mmaps from the motion-correction step, not a single joined array --
    # loading those directly with cm.load() is what produces CaImAn's
    # "The file is in F order, it should be in C order (see save_memmap
    # function)" error. The fix is the same explicit re-join CaImAn's own
    # demo pipelines use: cm.save_memmap(..., order='C') merges them into
    # one guaranteed C-order file before loading.
    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    if progress_cb:
        progress_cb("NormCorre: joining motion-corrected chunks into one file...")
    fname_joined = cm.save_memmap(fname_mc, base_name="memmap_gain_", order="C")
    m = cm.load(fname_joined)
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


def _normcorre_then_maybe_save(tiff_paths, max_frames, pw_rigid, save_mc, save_dir, progress_cb=None,
                                on_normcorre_start=None, on_normcorre_done=None, frame_start=None):
    """Run NormCorre, optionally save the full result as reg_tif chunks
    (saved BEFORE the max_frames window is applied, so the complete
    registered movie is preserved for next time even if this run only
    analyzed a subset of it), then trim to the analysis window.

    on_normcorre_start/on_normcorre_done bracket the whole "NormCorre is
    genuinely needed" window (this function is only called when reg_tif
    wasn't found) -- the GUI uses these to show/hide the big motion-
    correction overlay. on_normcorre_done always fires, even on error, so
    the overlay never gets stuck open.

    frame_start (optional): explicit starting frame index for the trim
    step, from the user's "Start frame" setting; None means auto-centered."""
    if on_normcorre_start:
        on_normcorre_start()
    try:
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
            start, end = contiguous_window(arr.shape[0], max_frames, start=frame_start)
            arr = arr[start:end]
        return arr, note_suffix
    finally:
        if on_normcorre_done:
            on_normcorre_done()


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


def load_tiffs_direct(paths, max_frames, frame_start=None):
    """Read raw/already-registered TIFF frames as-is, no motion correction.
    Like load_reg_tif, only decodes the chunk files overlapping a contiguous
    frame window rather than reading everything then subsampling.
    frame_start (optional): explicit starting frame index; None = auto-centered."""
    import tifffile
    counts = []
    for p in paths:
        with tifffile.TiffFile(p) as tf:
            counts.append(len(tf.pages))
    total = int(sum(counts))
    offsets = np.concatenate([[0], np.cumsum(counts)])

    start, end = contiguous_window(total, max_frames, start=frame_start)
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
                           ask_raw_dir=None, save_mc=False,
                           on_normcorre_start=None, on_normcorre_done=None, frame_start=None):
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

    frame_start (optional): explicit starting frame index for the analysis
    window, from the user's "Start frame" setting; None (default) keeps
    the original auto-centered window.
    """
    mov = load_reg_tif(plane_dir, max_frames, frame_start=frame_start)
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
        tiff_paths, max_frames, pw_rigid, save_mc, plane_dir, progress_cb=progress_cb,
        on_normcorre_start=on_normcorre_start, on_normcorre_done=on_normcorre_done,
        frame_start=frame_start)
    return RegisteredMovie(arr, "normcorre",
                            f"{len(tiff_paths)} raw TIFF(s), {arr.shape[0]} frames after "
                            f"motion correction{note_suffix}")


def load_registered_movie_manual(raw_tiffs, max_frames, pw_rigid, skip_mc, save_mc, progress_cb=None,
                                  on_normcorre_start=None, on_normcorre_done=None, frame_start=None):
    """Manual-TIFF-mode equivalent of load_registered_movie (no Suite2p ops.npy
    available). Checks for a reg_tif/ folder next to the source TIFFs first —
    including one saved by an earlier run of this same tool — before running
    NormCorre. frame_start (optional): explicit starting frame index; None
    means auto-centered (default)."""
    source_dir = os.path.dirname(raw_tiffs[0])
    mov = load_reg_tif(source_dir, max_frames, frame_start=frame_start)
    if mov is not None:
        return mov

    if skip_mc:
        return load_tiffs_direct(raw_tiffs, max_frames, frame_start=frame_start)

    arr, note_suffix = _normcorre_then_maybe_save(
        raw_tiffs, max_frames, pw_rigid, save_mc, source_dir, progress_cb=progress_cb,
        on_normcorre_start=on_normcorre_start, on_normcorre_done=on_normcorre_done,
        frame_start=frame_start)
    return RegisteredMovie(arr, "normcorre",
                            f"{len(raw_tiffs)} raw TIFF(s), {arr.shape[0]} frames after "
                            f"motion correction{note_suffix}")


def _load_mat_movie_array(path):
    """Load a 3D numeric movie array out of a .mat file, for the "load an
    already motion-corrected movie" bypass (skips NormCorre entirely).
    Handles both classic (pre-v7.3) .mat files via scipy.io.loadmat and
    v7.3/HDF5 .mat files via h5py -- MATLAB switches a variable to v7.3
    format once it exceeds the classic format's 2GB limit, which a
    multi-thousand-frame movie routinely does.

    Picks the largest 3D numeric array in the file as "the movie" rather
    than requiring an exact variable name, since different pipelines save
    it under different names (mov, Y, data, Ym, registered, ...).

    Axis order: MATLAB's own convention for a movie is (x, y, time)
    (`size(mov)` -> [Nx Ny Nt]). scipy.io.loadmat preserves that shape
    as-is in the returned numpy array, so it needs an explicit transpose
    to (time, y, x). h5py reading a v7.3/HDF5 .mat file already reports
    the shape reversed relative to MATLAB's own dimension order (an HDF5
    storage-order quirk from MATLAB writing column-major data into a
    row-major container) -- so an h5py-loaded array comes back as
    (time, y, x) already, with no transpose needed. Returns
    (array_as_T_Y_X, variable_name)."""
    try:
        mat = scipy.io.loadmat(path)
        candidates = {k: v for k, v in mat.items()
                      if not str(k).startswith("__")
                      and isinstance(v, np.ndarray) and v.ndim == 3
                      and np.issubdtype(v.dtype, np.number)}
        if not candidates:
            raise RuntimeError(
                f"No 3D numeric array found in {os.path.basename(path)} -- expected "
                f"a motion-corrected movie matrix (x, y, time). Variables found: "
                f"{[k for k in mat if not str(k).startswith('__')]}")
        varname = max(candidates, key=lambda k: candidates[k].size)
        arr = np.transpose(candidates[varname], (2, 1, 0))  # (x,y,t) -> (t,y,x)
        return arr, varname
    except NotImplementedError:
        # scipy.io.loadmat can't read v7.3 (HDF5-backed) .mat files
        import h5py
        candidates = {}

        def _visit(name, obj):
            if (isinstance(obj, h5py.Dataset) and obj.ndim == 3
                    and np.issubdtype(obj.dtype, np.number)):
                candidates[name] = obj

        with h5py.File(path, "r") as f:
            f.visititems(_visit)
            if not candidates:
                raise RuntimeError(
                    f"No 3D numeric dataset found in {os.path.basename(path)} -- "
                    f"expected a motion-corrected movie matrix (x, y, time).")
            varname = max(candidates, key=lambda k: candidates[k].size)
            arr = np.asarray(candidates[varname])  # already (t,y,x) -- see docstring
        return arr, varname.rsplit("/", 1)[-1]


def roi_exclusion_mask(stat, Ly, Lx):
    """Boolean mask, True = background/non-cell pixel."""
    mask = np.ones((Ly, Lx), dtype=bool)
    for cell in stat:
        yp = np.asarray(cell["ypix"]); xp = np.asarray(cell["xpix"])
        valid = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
        mask[yp[valid], xp[valid]] = False
    return mask


def footprints_to_raw_traces(movie_arr, A, dims, threshold=0.2, min_pixels=9,
                              return_indices=False):
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
    If return_indices=True, also returns kept_idx -- the column indices into
    the *input* A that ended up in F/npix (row i of F/npix corresponds to
    kept_idx[i]) -- so a caller juggling multiple component-quality arrays
    (e.g. for a mask-overlay viewer) can line them back up.
    """
    Ly, Lx = dims
    A_dense = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
    n_components = A_dense.shape[1]
    T = movie_arr.shape[0]
    # Fortran-order per-frame flatten, to match A's column layout
    flat_movie = movie_arr.transpose(0, 2, 1).reshape(T, Ly * Lx).astype(np.float64)

    F_list, npix_list, idx_list = [], [], []
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
        idx_list.append(i)

    if not F_list:
        F, npix = np.zeros((0, T)), np.zeros((0,), dtype=int)
    else:
        F, npix = np.array(F_list), np.array(npix_list, dtype=int)
    if return_indices:
        return F, npix, np.array(idx_list, dtype=int)
    return F, npix


def rank_and_keep_top_fraction(scores, keep_fraction=0.6):
    """Given a 1D array of per-component quality scores (higher = better),
    return the indices of the top `keep_fraction` of them, sorted best-first.
    Always keeps at least 1 component if scores is non-empty."""
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)
    n_keep = max(1, math.ceil(n * keep_fraction))
    order = np.argsort(-np.asarray(scores))  # best (highest) first
    return order[:n_keep]


def run_cnmf_segmentation(movie_arr, fs, gsig, progress_cb=None, keep_fraction=0.6):
    """Run CaImAn's CNMF purely for ROI/footprint detection on an already
    motion-corrected movie, then re-derive raw pixel-sum traces from those
    footprints via footprints_to_raw_traces (see its docstring for why).

    After fitting, attempts CaImAn's own component-quality evaluation
    (estimates.evaluate_components) and keeps only the top `keep_fraction`
    of components by whatever quality score that produced -- CaImAn's CNN
    classifier probability (cnn_preds, a real 0-1 probability) if available,
    otherwise the r_value spatial-consistency score as a fallback (NOT a
    true probability, just a ranking proxy -- flagged as such in mask_info).
    If evaluation fails entirely (e.g. no CNN model installed and an older
    CaImAn without r_values), all components are kept and mask_info notes
    that no quality filtering was applied.

    Returns (F, npix, mask_info). mask_info is a dict with everything needed
    to render a footprint-overlay sanity-check view: dims, A_kept (dense,
    Ly*Lx x n_kept -- the ones that survived quality filtering, before the
    footprints_to_raw_traces pixel-count threshold), A_rejected (dense, the
    quality-filtered-out ones), quality_metric ("CNN probability" /
    "r_value (not a true probability)" / None), n_total, n_kept.

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
    try:
        fitted = cnm.fit(images)
    except Exception as e:
        raise RuntimeError(f"CaImAn CNMF.fit() raised an error: {e}") from e

    # Some CaImAn versions/code paths return None from fit() while mutating
    # the CNMF object in place instead of returning the fitted object --
    # fall back to the original `cnm` (still usable) rather than crashing
    # on "'NoneType' object has no attribute 'estimates'".
    if fitted is not None:
        cnm = fitted
    if cnm is None or getattr(cnm, "estimates", None) is None or getattr(cnm.estimates, "A", None) is None:
        raise RuntimeError(
            "CaImAn CNMF ran but produced no usable estimates (cnm.estimates.A is missing "
            "or empty). This can happen if CNMF found zero components for this movie/gSig "
            "combination, or if this installed CaImAn version's fit() behaves differently "
            "than expected here. Try a different 'CNMF cell radius' setting or more frames, "
            "and check the console/terminal for CaImAn's own log output for the real cause."
        )

    A_full = cnm.estimates.A
    A_full_dense = np.asarray(A_full.todense()) if hasattr(A_full, "todense") else np.asarray(A_full)
    n_found = A_full_dense.shape[1]
    if progress_cb:
        progress_cb(f"CNMF found {n_found} candidate component(s); evaluating quality...")

    # --- quality-based filtering: keep only the top `keep_fraction` ---
    quality_metric = None
    quality_scores_kept = None
    quality_idx = np.arange(n_found)  # default: no filtering
    try:
        cnm.estimates.evaluate_components(images, cnm.params, dview=None)
        cnn_preds = getattr(cnm.estimates, "cnn_preds", None)
        r_values = getattr(cnm.estimates, "r_values", None)
        scores = None
        if cnn_preds is not None and len(cnn_preds) == n_found and np.all(np.isfinite(cnn_preds)) \
                and not np.all(np.asarray(cnn_preds) < 0):
            scores = np.asarray(cnn_preds, dtype=float)
            quality_metric = "CNN probability"
        elif r_values is not None and len(r_values) == n_found:
            scores = np.asarray(r_values, dtype=float)
            quality_metric = "r_value (spatial consistency -- not a true probability)"
        if scores is not None:
            quality_idx = rank_and_keep_top_fraction(scores, keep_fraction)
            quality_idx = np.sort(quality_idx)  # keep stable column order
            quality_scores_kept = scores[quality_idx]
            if progress_cb:
                progress_cb(f"Keeping top {int(round(keep_fraction * 100))}% by {quality_metric}: "
                            f"{len(quality_idx)}/{n_found} component(s)")
    except Exception as e:
        # CNMF quality evaluation is optional (needs a CNN model file for the
        # cnn_preds path, and isn't available on every CaImAn version) -- if
        # it fails for any reason, fall back to keeping everything rather
        # than losing the whole run over a metrics-only step.
        if progress_cb:
            progress_cb(f"Component-quality evaluation unavailable ({e}); keeping all components.")

    rejected_idx = np.setdiff1d(np.arange(n_found), quality_idx)
    A_quality_kept = A_full_dense[:, quality_idx]

    F, npix, kept_of_quality = footprints_to_raw_traces(
        movie_arr, A_quality_kept, (Ly, Lx),
        min_pixels=max(4, int(np.pi * (gsig / 2) ** 2 * 0.3)),
        return_indices=True)

    mask_info = dict(
        dims=(Ly, Lx),
        A_kept=A_quality_kept[:, kept_of_quality] if len(kept_of_quality) else A_quality_kept[:, :0],
        A_rejected=A_full_dense[:, rejected_idx],
        quality_metric=quality_metric,
        quality_scores_kept=(quality_scores_kept[kept_of_quality]
                              if quality_scores_kept is not None and len(kept_of_quality) else None),
        n_total=n_found,
        n_kept=int(F.shape[0]),
    )
    return F, npix, mask_info


def compute_ptc(movie_arr, spatial_bin=1, exclude_mask=None, n_mean_bins=40, chunk=200,
                 margin=0):
    """Frame-differencing mean/variance PTC estimate, streamed in chunks so
    peak memory stays ~O(chunk x Y x X) instead of O(T x Y x X).

    For consecutive frame pairs (i, i+1): mean = (f_i+f_{i+1})/2,
    var = (f_i - f_{i+1})^2 / 2  (unbiased shot-noise variance estimator;
    slow biological/structural signal cancels in the difference). This
    matches the Lees et al. 2025 (Nature Protocols, Procedure 7 / Box 7-8)
    reference method exactly.

    `margin` crops this many pixels off each edge of every frame before
    anything else (mean/var computation, binning) -- matches that same
    reference method's practice of excluding blanking-artifact edges
    (common on resonant-scan two-photon systems) from the PTC estimate.
    Default 0 preserves the original no-cropping behavior.

    Returns per-pixel arrays (mu_px, var_px) and binned arrays
    (mu_bin, var_bin, n_bin) for plotting/fitting.
    """
    T = movie_arr.shape[0]
    if T < 2:
        raise RuntimeError("Need at least 2 frames for a PTC estimate.")

    Y0, X0 = movie_arr.shape[1], movie_arr.shape[2]
    margin = max(0, int(margin))
    if margin > 0:
        if margin * 2 >= min(Y0, X0):
            raise RuntimeError(
                f"Edge margin ({margin}px) is too large for a {Y0}x{X0} frame -- "
                f"nothing would be left to analyze.")
        if exclude_mask is not None:
            exclude_mask = exclude_mask[margin:Y0 - margin, margin:X0 - margin]
    Y, X = Y0 - 2 * margin, X0 - 2 * margin

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
        if margin > 0:
            block = block[:, margin:Y0 - margin, margin:X0 - margin]
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
# Loading-animation artwork: Filip's actual sketches, embedded directly as
# images (PNG, extracted from the SVG files he uploaded and trimmed to
# their content bbox) rather than hand-traced vector line art -- hand-
# tracing kept not looking right across several rounds, so this draws the
# real drawings themselves. Files live in an "art/" folder next to this
# script; each is loaded once, cached, then resized/rotated/distorted per
# frame as needed. Used for the Gain Estimation progress indicators -- the
# busy-state "photon hits neuron" strip, and the bigger motion-correction
# distortion/reassembly overlay.
# ═════════════════════════════════════════════════════════════════════════

_ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "art")

_ART_FILES = {
    "neuron_neutral": "neuron_neutral.png",
    "neuron_annoyed": "neuron_annoyed.png",
    "neuron_surprised": "neuron_surprised.png",
    "chameleon": "chameleon.png",
    "photon": "photon.png",
}

_ART_RAW_CACHE = {}  # name -> PIL.Image (RGBA, loaded once from disk)


def _load_art_image(name):
    """Load (and cache) the source PIL image for one art asset. Raises a
    clear error if the art/ folder didn't ship alongside the script, rather
    than a confusing Tk/PIL traceback."""
    if name in _ART_RAW_CACHE:
        return _ART_RAW_CACHE[name]
    fname = _ART_FILES[name]
    path = os.path.join(_ART_DIR, fname)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing art asset '{fname}' -- expected it at {path}. "
            f"The art/ folder needs to ship alongside kurtosis_checker.py.")
    im = Image.open(path).convert("RGBA")
    _ART_RAW_CACHE[name] = im
    return im


_ART_INVERTED_CACHE = {}  # name -> PIL.Image, RGB channels inverted (black lines -> white)


def _selective_invert_lines(im):
    """Invert only the near-black, low-saturation ink strokes to white --
    leaves colored fill (the green highlighter tint on the neuron/
    chameleon, the red on the photon) untouched. A blind full-RGB invert
    turns the green tint into magenta/purple, which isn't what was wanted;
    this only flips pixels that are dark AND close to gray (true black
    line art), since colored fill pixels have real channel separation
    (green: G >> R,B; red: R >> G,B) that a grayscale ink stroke doesn't."""
    arr = np.array(im).astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = maxc - minc
    is_black_ink = (maxc < 110) & (sat < 45)
    out = arr.copy()
    for ch in range(3):
        out[..., ch] = np.where(is_black_ink, 255 - arr[..., ch], arr[..., ch])
    return Image.fromarray(out.astype(np.uint8), mode="RGBA")


def _load_art_image_inverted(name):
    """Same artwork as _load_art_image, but with just the black ink lines
    flipped to white (see _selective_invert_lines) -- reads against a dark
    background instead of needing a white canvas behind it, while keeping
    the original green/red fill colors intact."""
    if name in _ART_INVERTED_CACHE:
        return _ART_INVERTED_CACHE[name]
    im = _load_art_image(name)
    inv = _selective_invert_lines(im)
    _ART_INVERTED_CACHE[name] = inv
    return inv


def _apply_row_distort(im, distort, phase):
    """Row-wise horizontal pixel shift (a 'scanline glitch'), vectorized
    over numpy so it's cheap enough to redo every animation frame. Mirrors
    the shape of the original hand-drawn jitter formula (two summed sines
    at different row-frequencies/phase-rates) but applied to pixels instead
    of vector point coordinates -- used by the motion-correction overlay to
    visually 'scramble' the neuron and let it resolve back to clean as
    NormCorre progress comes in."""
    if distort <= 0.002:
        return im
    arr = np.array(im)
    h, w = arr.shape[0], arr.shape[1]
    if w == 0 or h == 0:
        return im
    max_shift = distort * w * 0.10
    rows = np.arange(h)
    shifts = (max_shift * np.sin(rows * 0.15 + phase)
              + max_shift * 0.4 * np.sin(rows * 0.4 - phase * 1.7))
    shifts = shifts.astype(np.int64)
    col_idx = (np.arange(w)[None, :] - shifts[:, None]) % w
    out = arr[rows[:, None], col_idx]
    return Image.fromarray(out, mode="RGBA")


def _render_art_image(canvas, name, cx, cy, target_h, tag,
                       rotate_deg=0.0, distort=0.0, phase=0.0, inverted=False):
    """Load, resize (to target_h px tall, aspect-preserved), optionally
    rotate and/or scanline-distort, then draw the given art asset centered
    at (cx, cy) on `canvas` via create_image. Keeps a strong reference to
    the resulting PhotoImage on the canvas itself (Tk requires this --
    otherwise it gets garbage-collected and the image silently vanishes).
    `inverted=True` uses the selectively-line-inverted (white ink strokes,
    original fill colors kept) version, for display on a dark background.
    Returns (disp_w, disp_h), the actual on-screen pixel size, so callers
    can compute anchor points (eye, snout, ...) relative to it.
    """
    canvas.delete(tag)
    im = _load_art_image_inverted(name) if inverted else _load_art_image(name)
    target_h = max(4, int(target_h))
    scale = target_h / im.height
    target_w = max(4, int(im.width * scale))
    im = im.resize((target_w, target_h), Image.LANCZOS)
    im = _apply_row_distort(im, distort, phase)
    if rotate_deg:
        im = im.rotate(-rotate_deg, expand=True, resample=Image.BICUBIC)
    photo = ImageTk.PhotoImage(im)
    if not hasattr(canvas, "_art_photo_refs"):
        canvas._art_photo_refs = {}
    canvas._art_photo_refs[tag] = photo  # prevent GC
    canvas.create_image(cx, cy, image=photo, anchor="center", tags=tag)
    return im.width, im.height


# Fractional anchor offsets (relative to each character's own displayed
# width/height, measured from its center) used to aim the photon at
# roughly the right spot -- the neuron's eye, the chameleon's snout tip --
# rather than just its bounding-box center. Approximate (eyeballed from
# the source images' own proportions), not pixel-exact.
NRN_EYE_OFFSET_FRAC = (0.0, -0.22)      # eye sits a bit above center (legs take up the lower half)
CHAM_SNOUT_OFFSET_FRAC = (-0.48, -0.22)  # snout is at the far-left edge, upper third (measured
                                          # directly off chameleon.png's alpha bbox: leftmost
                                          # content column sits at y-fraction -0.217 of height)


def draw_neuron(canvas, cx, cy, scale, expr="neutral", distort=0.0, phase=0.0,
                 tag="neuron", outline="#111111"):
    """Draw Filip's actual neuron sketch (one of 3 expressions) centered at
    (cx, cy). `scale` sets the display size (kept as the same "px per local
    60-unit" convention the old vector version used, so callers didn't need
    to change their numbers -- multiplied by a fixed constant to get a
    pixel height). `distort` in [0, 1] applies a scanline glitch that the
    motion-correction overlay decays to zero as NormCorre progresses.
    Returns (disp_w, disp_h) for anchor-point math."""
    name = {"neutral": "neuron_neutral", "surprised": "neuron_surprised",
            "annoyed": "neuron_annoyed"}.get(expr, "neuron_neutral")
    target_h = scale * 4.2
    return _render_art_image(canvas, name, cx, cy, target_h, tag,
                              distort=distort, phase=phase, inverted=True)


def draw_chameleon(canvas, cx, cy, scale, tag="chameleon", outline="#111111"):
    """Draw Filip's actual chameleon sketch (the animation's 'light
    source') centered at (cx, cy). Returns (disp_w, disp_h)."""
    target_h = scale * 3.4
    return _render_art_image(canvas, "chameleon", cx, cy, target_h, tag, inverted=True)


def _load_photon_variant(tint="red"):
    """The photon squiggle, black ink lines flipped to white (like the
    other art assets) and, for the 'green' variant, its red stroke/wash
    recolored to green by swapping the R and G channels -- a cheap, exact
    hue swap since the source art is a pure red-on-white sketch (no other
    color mixed in to distort). Used for the neuron's return volleys, to
    visually distinguish them from the chameleon's outbound red shots."""
    cache_key = f"photon_{tint}"
    if cache_key in _ART_INVERTED_CACHE:
        return _ART_INVERTED_CACHE[cache_key]
    im = _load_art_image_inverted("photon")  # black ink -> white, red stroke untouched
    if tint == "green":
        arr = np.array(im).copy()
        arr[..., [0, 1]] = arr[..., [1, 0]]  # swap R/G: red stroke -> green
        im = Image.fromarray(arr, mode="RGBA")
    _ART_INVERTED_CACHE[cache_key] = im
    return im


def draw_photon(canvas, cx, cy, length, amplitude, angle_deg, phase, tag="photon",
                 tint="red"):
    """Draw Filip's actual photon squiggle sketch centered at (cx, cy),
    sized from `length`, rotated to point along its direction of travel
    (`angle_deg`). `amplitude`/`phase` no longer reshape the squiggle itself
    (it's a fixed drawing now) but `phase` still adds a small extra wobble
    rotation so the in-flight animation keeps some life to it. `tint`
    selects "red" (the chameleon's outbound shot) or "green" (the neuron's
    return volley)."""
    canvas.delete(tag)
    im = _load_photon_variant(tint)
    target_w = max(4, int(length * 2.0))
    scale = target_w / im.width
    target_h = max(4, int(im.height * scale))
    im = im.resize((target_w, target_h), Image.LANCZOS)
    wobble = 6.0 * math.sin(phase)
    im = im.rotate(-(angle_deg + wobble), expand=True, resample=Image.BICUBIC)
    photo = ImageTk.PhotoImage(im)
    if not hasattr(canvas, "_art_photo_refs"):
        canvas._art_photo_refs = {}
    canvas._art_photo_refs[tag] = photo
    canvas.create_image(cx, cy, image=photo, anchor="center", tags=tag)


def draw_tally(canvas, x, y, count, tag="tally", stroke=2, line_h=16, line_w=6,
                group_gap=9, color="#111111", max_groups=6):
    """Draw `count` as classic 'gate' tally marks (four verticals per group
    of five, with a diagonal fifth stroke crossing them), left to right
    starting at (x, y) as the top-left anchor. Caps at `max_groups` full
    groups to avoid overflowing the strip; anything beyond that collapses
    to a '+N' text suffix instead of drawing hundreds of tiny lines.
    Returns the x-coordinate just past the last mark drawn (for layout)."""
    canvas.delete(tag)
    full_groups, remainder = divmod(max(0, int(count)), 5)
    cx = x
    drawn_groups = min(full_groups, max_groups)
    for _ in range(drawn_groups):
        for i in range(4):
            lx = cx + i * line_w
            canvas.create_line(lx, y, lx, y + line_h, fill=color, width=stroke, tags=tag)
        canvas.create_line(cx - 2, y + line_h + 2, cx + 3 * line_w + 2, y - 2,
                            fill=color, width=stroke, tags=tag)
        cx += 4 * line_w + group_gap

    if full_groups > max_groups:
        leftover = count - drawn_groups * 5
        canvas.create_text(cx, y + line_h / 2, text=f"+{leftover}", fill=color,
                            font=("Helvetica", 10, "bold"), anchor="w", tags=tag)
        return cx + 28

    for i in range(remainder):
        lx = cx + i * line_w
        canvas.create_line(lx, y, lx, y + line_h, fill=color, width=stroke, tags=tag)
    return cx + remainder * line_w


class PhotonNeuronBar(tk.Canvas):
    """Busy-state animation replacing the plain progress strip: a chameleon
    (light source) launches red photon squiggles at the neuron; each impact
    increments a running counter and swaps the neuron's expression for a
    moment. Purely decorative -- ticks continuously while Run Analysis is
    doing anything (reg_tif read / PTC compute / CNMF), not tied to a real
    progress metric, since most of those steps don't expose one either.

    Dark background matching the rest of the app, neuron on the left,
    chameleon on the right -- both drawn as white-inverted line art (so
    they read against the dark canvas instead of needing a white one)."""

    HEIGHT = 190
    NRN_SCALE = 42
    CHAM_SCALE = 46

    def __init__(self, parent, **kw):
        super().__init__(parent, height=self.HEIGHT, bg=BG_DARK, highlightthickness=0, **kw)
        self._width = 400
        self._running = False
        self._job = None
        self._hits = 0
        self._expr = "neutral"
        self._expr_until = 0.0
        self._flight_frac = None
        self._flight_t0 = 0.0
        self._flight_dir = "out"  # "out" = chameleon->neuron, "back" = neuron->chameleon
        self._next_launch = 0.0
        self._t0 = None
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self._width = event.width
        if not self._running:
            self._render(0.0)

    def start(self):
        if self._running:
            return
        self._running = True
        self._hits = 0
        self._expr = "neutral"
        self._t0 = time.monotonic()
        self._next_launch = 0.4
        self._flight_frac = None
        self._flight_dir = "out"
        self._tick()

    def stop(self):
        self._running = False
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        self.delete("all")

    def _tick(self):
        if not self._running:
            return
        t = time.monotonic() - self._t0
        if self._flight_frac is None and t >= self._next_launch:
            self._flight_frac = 0.0
            self._flight_t0 = t
            self._flight_dir = "out"
        if self._flight_frac is not None:
            dur = 0.55
            frac = (t - self._flight_t0) / dur
            if frac >= 1.0:
                if self._flight_dir == "out":
                    # chameleon's photon just landed -- neuron reacts, and
                    # only fires one back every 2nd hit it takes
                    self._hits += 1
                    self._expr = random.choice(["surprised", "annoyed"])
                    self._expr_until = t + 0.5
                    if self._hits % 2 == 0:
                        self._flight_dir = "back"
                        self._flight_t0 = t
                        self._flight_frac = 0.0
                    else:
                        self._flight_frac = None
                        self._next_launch = t + random.uniform(0.5, 1.1)
                else:
                    # return shot landed -- volley's done, go idle until the
                    # next launch
                    self._flight_frac = None
                    self._next_launch = t + random.uniform(0.5, 1.1)
            else:
                self._flight_frac = frac
        if self._expr != "neutral" and t >= self._expr_until:
            self._expr = "neutral"
        try:
            self._render(t)
        except Exception:
            # Never let a rendering hiccup (e.g. a missing art/ asset) kill
            # the recurring .after() schedule -- that would silently stop
            # the whole loader with no visible error. Log once via the
            # canvas's own error flag instead of spamming every 33ms.
            if not getattr(self, "_render_error_shown", False):
                self._render_error_shown = True
                traceback.print_exc()
        self._job = self.after(33, self._tick)

    def _render(self, t):
        w = max(self._width, 60)
        h = self.HEIGHT
        nrn_cx, nrn_cy = w * 0.16, h * 0.68
        cham_cx, cham_cy = w * 0.86, h * 0.46

        nrn_w, nrn_h = draw_neuron(self, nrn_cx, nrn_cy, scale=self.NRN_SCALE,
                                    expr=self._expr, tag="nrn")
        cham_w, cham_h = draw_chameleon(self, cham_cx, cham_cy, scale=self.CHAM_SCALE,
                                         tag="cham")

        if self._flight_frac is not None:
            cham_pt = (cham_cx + CHAM_SNOUT_OFFSET_FRAC[0] * cham_w,
                       cham_cy + CHAM_SNOUT_OFFSET_FRAC[1] * cham_h)
            nrn_pt = (nrn_cx + NRN_EYE_OFFSET_FRAC[0] * nrn_w,
                      nrn_cy + NRN_EYE_OFFSET_FRAC[1] * nrn_h)
            if self._flight_dir == "out":
                (x0, y0), (x1, y1) = cham_pt, nrn_pt
                tint = "red"
            else:
                (x0, y0), (x1, y1) = nrn_pt, cham_pt
                tint = "green"
            fx = x0 + (x1 - x0) * self._flight_frac
            fy = y0 + (y1 - y0) * self._flight_frac
            angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
            draw_photon(self, fx, fy, length=30, amplitude=8,
                        angle_deg=angle, phase=t * 14, tag="photon", tint=tint)
        else:
            self.delete("photon")

        self.delete("counter_label")
        label = self.create_text(14, h - 12, text="photons:", anchor="w",
                                  fill=TEXT_MAIN, font=("Helvetica", 9), tags="counter_label")
        bbox = self.bbox(label)
        tally_x = (bbox[2] + 8) if bbox else 70
        draw_tally(self, tally_x, h - 12, self._hits, tag="counter_tally", color=TEXT_MAIN)


class MotionCorrectionOverlay:
    """Big, centered popup shown specifically while NormCorre motion
    correction is running. Shows a progress bar driven by CaImAn's own
    log-line events (via the existing progress_cb channel) and a neuron
    drawing whose scanline/phase distortion decays to zero as those events
    accumulate, visually 'resolving' into the clean drawing -- a metaphor
    for what motion correction does to the movie.

    NOTE: CaImAn's motion_correct() is one blocking call with no true
    frame-level percentage exposed, so the bar is a step-count heuristic
    (each distinct log line nudges it forward, capped short of 100% until
    the call actually returns) rather than an exact percentage. The UI
    says so plainly rather than implying more precision than it has."""

    W, H = 600, 620
    NRN_SCALE = 95

    def __init__(self, root):
        self.top = tk.Toplevel(root)
        self.top.title("Motion correction running...")
        self.top.configure(bg=BG_DARK)
        try:
            self.top.resizable(False, False)
        except Exception:
            pass
        self.top.geometry(f"{self.W}x{self.H}")
        try:
            root.update_idletasks()
            rx, ry = root.winfo_rootx(), root.winfo_rooty()
            rw, rh = root.winfo_width(), root.winfo_height()
            x = rx + (rw - self.W) // 2
            y = ry + (rh - self.H) // 2
            self.top.geometry(f"{self.W}x{self.H}+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

        tk.Label(self.top, text="Registering movie with NoRMCorre...",
                 bg=BG_DARK, fg=TEXT_BRIGHT, font=("Helvetica", 13, "bold")
                 ).pack(pady=(18, 4))

        self._bar_w = self.W - 60
        self.bar_canvas = tk.Canvas(self.top, width=self._bar_w, height=18,
                                     bg=BG_DARK, highlightthickness=0)
        self.bar_canvas.pack(pady=(6, 2))

        self.pct_label = tk.Label(self.top, text="0%", bg=BG_DARK, fg=TEXT_DIM,
                                   font=("Helvetica", 10))
        self.pct_label.pack()

        self.msg_var = tk.StringVar(value="Starting NoRMCorre...")
        tk.Label(self.top, textvariable=self.msg_var, bg=BG_DARK, fg=TEXT_DIM,
                 font=("Helvetica", 9), wraplength=self.W - 40
                 ).pack(pady=(2, 8))

        self._art_w = self.W - 40
        self._art_h = self.H - 190
        self.art_canvas = tk.Canvas(self.top, width=self._art_w, height=self._art_h,
                                     bg=BG_DARK, highlightthickness=0)
        self.art_canvas.pack()

        self._progress = 0.0
        self._phase = 0.0
        self._closing = False
        self._render_bar()
        self._render_art()

    def _render_bar(self):
        c = self.bar_canvas
        c.delete("all")
        w, h = self._bar_w, 18
        c.create_rectangle(0, 2, w, h - 2, fill=BG_PANEL, outline="")
        fill_w = int(w * min(1.0, max(0.0, self._progress)))
        if fill_w > 0:
            c.create_rectangle(0, 2, fill_w, h - 2, fill=ACCENT, outline="")
        try:
            self.pct_label.config(text=f"{int(self._progress * 100)}%")
        except Exception:
            pass

    def _render_art(self):
        cx = self._art_w / 2
        cy = self._art_h / 2
        distort = max(0.0, 1.0 - self._progress)
        try:
            draw_neuron(self.art_canvas, cx, cy, scale=self.NRN_SCALE, expr="neutral",
                        distort=distort, phase=self._phase, tag="neuron")
        except Exception:
            # A missing/broken art asset shouldn't take down the progress
            # bar itself -- that's the one part of this popup that's not
            # just decorative.
            if not getattr(self, "_render_error_shown", False):
                self._render_error_shown = True
                traceback.print_exc()

    def tick(self, msg=None):
        """Called (via the app's progress_cb channel) each time a new
        CaImAn log line / status message arrives during motion correction."""
        if self._closing:
            return
        if msg:
            try:
                self.msg_var.set(msg)
            except Exception:
                pass
        self._progress += (0.95 - self._progress) * 0.18
        self._progress = min(self._progress, 0.95)
        self._phase += 0.35
        self._render_bar()
        self._render_art()

    def close(self, success=True):
        if self._closing:
            return
        self._closing = True
        self._progress = 1.0
        self._phase = 0.0
        try:
            self.msg_var.set("Done." if success else "Stopped.")
        except Exception:
            pass
        self._render_bar()
        self._render_art()
        try:
            self.top.after(450, self.top.destroy)
        except Exception:
            pass


class MeanProjectionViewer:
    """Zoomable/pannable viewer for a registered movie's mean projection,
    shown right before the CNMF prompt so the user can eyeball how many
    pixels a cell spans before picking the 'CNMF cell radius' setting.
    Uses a real embedded matplotlib canvas with its navigation toolbar
    (zoom/pan + a live cursor-position readout at the bottom-left) --
    unlike the main app's plot canvas, the toolbar is kept here on
    purpose since that readout is the whole point of this window.

    Rather than just showing the image and trusting the user to go set
    "CNMF cell radius (px)" themselves afterward, this window asks
    directly for the cell diameter (in px) they measured and hands that
    back via on_continue -- the caller converts it to a radius and
    updates the setting in the same flow."""

    W, H = 640, 720

    def __init__(self, root, mean_img, default_diameter, on_continue):
        self.on_continue = on_continue
        self.top = tk.Toplevel(root)
        self.top.title("Mean projection — measure cell diameter")
        self.top.configure(bg=BG_DARK)
        self.top.geometry(f"{self.W}x{self.H}")
        try:
            root.update_idletasks()
            rx, ry = root.winfo_rootx(), root.winfo_rooty()
            rw, rh = root.winfo_width(), root.winfo_height()
            x = rx + (rw - self.W) // 2
            y = ry + (rh - self.H) // 2
            self.top.geometry(f"{self.W}x{self.H}+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass
        self.top.protocol("WM_DELETE_WINDOW", self._continue)

        tk.Label(self.top, text="Mean projection of the registered movie",
                 bg=BG_DARK, fg=TEXT_BRIGHT, font=("Helvetica", 12, "bold")
                 ).pack(pady=(12, 2))
        tk.Label(self.top,
                 text="Zoom/pan below (magnifying-glass tool, then scroll or "
                      "drag a box) to count how many pixels a cell spans "
                      "across. Cursor position in pixels shows at the "
                      "bottom-left of the toolbar as you hover.",
                 bg=BG_DARK, fg=TEXT_DIM, font=("Helvetica", 9),
                 wraplength=self.W - 40, justify="left").pack(pady=(0, 8))

        self.fig = Figure(facecolor=BG_DARK)
        ax = self.fig.add_axes([0.09, 0.07, 0.87, 0.87])
        ax.set_facecolor(BG_DARK)
        vmin, vmax = np.percentile(mean_img, [1, 99.5])
        if vmax <= vmin:
            vmax = vmin + 1.0
        ax.imshow(mean_img, cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_xlabel("x (px)", color=TEXT_MAIN, fontsize=8)
        ax.set_ylabel("y (px)", color=TEXT_MAIN, fontsize=8)
        ax.tick_params(colors=TEXT_MAIN, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(TEXT_DIM)
        self.ax = ax

        canvas_frame = tk.Frame(self.top, bg=BG_DARK)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = tk.Frame(self.top, bg=BG_MID)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.draw()

        entry_row = tk.Frame(self.top, bg=BG_DARK)
        entry_row.pack(pady=(10, 2))
        tk.Label(entry_row, text="Cell diameter you measured (px):",
                 bg=BG_DARK, fg=TEXT_MAIN, font=("Helvetica", 10)
                 ).pack(side=tk.LEFT, padx=(0, 8))
        self.diam_var = tk.StringVar(value=str(default_diameter))
        entry = tk.Entry(entry_row, textvariable=self.diam_var, width=8,
                          bg=BG_PANEL, fg=TEXT_BRIGHT, insertbackground=TEXT_BRIGHT,
                          relief=tk.FLAT)
        entry.pack(side=tk.LEFT)
        tk.Label(entry_row, text="→ sets CNMF cell radius to half of this",
                 bg=BG_DARK, fg=TEXT_DIM, font=("Helvetica", 9)
                 ).pack(side=tk.LEFT, padx=(8, 0))

        btn_row = tk.Frame(self.top, bg=BG_DARK)
        btn_row.pack(pady=10)
        tk.Button(btn_row, text="Continue →", command=self._continue,
                  bg=ACCENT, fg="white", font=("Helvetica", 10, "bold"),
                  relief=tk.FLAT, padx=16, pady=4).pack()

    def _continue(self):
        diameter = None
        try:
            raw = self.diam_var.get().strip()
            if raw:
                v = float(raw)
                if v > 0:
                    diameter = v
        except Exception:
            diameter = None
        try:
            self.top.destroy()
        except Exception:
            pass
        if self.on_continue:
            self.on_continue(diameter)


class CNMFMaskViewer:
    """Non-blocking sanity-check popup shown right after CNMF finishes: the
    mean projection with footprint contours overlaid so you can actually
    see the masks CNMF found, not just trust the flux histogram. Green =
    kept (survived component-quality filtering), red = filtered out. Same
    zoomable/pannable matplotlib canvas as MeanProjectionViewer -- doesn't
    block the pipeline, it's purely for visual inspection."""

    W, H = 640, 700
    THRESHOLD = 0.2  # must match footprints_to_raw_traces' own default

    def __init__(self, root, mean_img, mask_info):
        Ly, Lx = mask_info["dims"]
        self.top = tk.Toplevel(root)
        self.top.title("CNMF footprints — sanity check")
        self.top.configure(bg=BG_DARK)
        self.top.geometry(f"{self.W}x{self.H}")
        try:
            root.update_idletasks()
            rx, ry = root.winfo_rootx(), root.winfo_rooty()
            rw, rh = root.winfo_width(), root.winfo_height()
            x = rx + (rw - self.W) // 2
            y = ry + (rh - self.H) // 2
            self.top.geometry(f"{self.W}x{self.H}+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

        tk.Label(self.top, text="CNMF footprints over mean projection",
                 bg=BG_DARK, fg=TEXT_BRIGHT, font=("Helvetica", 12, "bold")
                 ).pack(pady=(12, 2))
        n_total = mask_info.get("n_total", 0)
        n_kept = mask_info.get("n_kept", 0)
        metric = mask_info.get("quality_metric") or "no quality filtering available"
        tk.Label(self.top,
                 text=f"Green = kept ({n_kept}/{n_total}) via {metric}   ·   "
                      f"Red = filtered out. Zoom/pan to check whether the "
                      f"green outlines actually look like cells before "
                      f"trusting the flux numbers.",
                 bg=BG_DARK, fg=TEXT_DIM, font=("Helvetica", 9),
                 wraplength=self.W - 40, justify="left").pack(pady=(0, 8))

        self.fig = Figure(facecolor=BG_DARK)
        ax = self.fig.add_axes([0.09, 0.07, 0.87, 0.87])
        ax.set_facecolor(BG_DARK)
        vmin, vmax = np.percentile(mean_img, [1, 99.5])
        if vmax <= vmin:
            vmax = vmin + 1.0
        ax.imshow(mean_img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

        def _overlay(A_dense, color):
            if A_dense is None:
                return
            for i in range(A_dense.shape[1]):
                comp = A_dense[:, i]
                peak = comp.max()
                if peak <= 0:
                    continue
                # A columns are flattened with x-major/y-minor order (matches
                # footprints_to_raw_traces' flat_movie convention) -- undo
                # that back into a (Ly, Lx) image for plotting.
                comp2d = comp.reshape((Lx, Ly)).T
                try:
                    ax.contour(comp2d, levels=[self.THRESHOLD * peak],
                               colors=[color], linewidths=1.0)
                except Exception:
                    pass

        _overlay(mask_info.get("A_rejected"), "#e74c3c")
        _overlay(mask_info.get("A_kept"), "#2ecc71")

        ax.set_xlabel("x (px)", color=TEXT_MAIN, fontsize=8)
        ax.set_ylabel("y (px)", color=TEXT_MAIN, fontsize=8)
        ax.tick_params(colors=TEXT_MAIN, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(TEXT_DIM)

        canvas_frame = tk.Frame(self.top, bg=BG_DARK)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = tk.Frame(self.top, bg=BG_MID)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.draw()

        tk.Button(self.top, text="Close", command=self.top.destroy,
                  bg=BG_PANEL, fg=TEXT_BRIGHT, font=("Helvetica", 10),
                  relief=tk.FLAT, padx=16, pady=4).pack(pady=10)


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

        # gain-mode state: pre-motion-corrected movie loaded directly from a
        # .mat file (bypasses NormCorre entirely -- e.g. when motion
        # correction was already done in MATLAB, or when NormCorre's memmap
        # write keeps failing with "No space left on device")
        self.g_mat_movie = None       # (T, Ly, Lx) array, or None
        self.g_mat_movie_note = ""

        # big centered popup shown only while NormCorre is genuinely running
        self._motion_overlay = None

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

        # gain-tab-only: load an already motion-corrected movie straight from
        # a .mat file, skipping NormCorre entirely (e.g. when it was already
        # motion-corrected in MATLAB, or NormCorre's memmap write keeps
        # hitting "No space left on device")
        self._mat_movie_btn = _lbtn(left, "🎬  Load .mat Movie", self._load_gain_mat_movie,
                                     bg="#8e44ad", bold=False)
        self._mat_movie_btn.pack_forget()  # hidden until the Gain Estimation tab is active

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

        # ── progress bar row — Gain Estimation tab only (never shown in
        # Kurtosis mode -- toggled in _switch_tab), covers the NormCorre /
        # CNMF / reg_tif-read steps that can take a while with no other
        # feedback than the status text above. Shows the photon-hits-neuron
        # animation instead of a generic striped bar. Default tab is
        # "kurtosis", so this starts unpacked. ─────────────────────────────
        self._progress_row = tk.Frame(self.root, bg=BG_MID)
        self.gain_progress_bar = PhotonNeuronBar(self._progress_row)
        self.gain_progress_bar.pack(fill=tk.X, padx=14, pady=(0, 6))
        # idle by default; _progress_start()/_progress_stop() start/stop
        # the animation while Run Analysis is doing real work

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
        # Two stacked rows instead of one long horizontal strip -- a single
        # row of this many items (especially the longer checkbox labels)
        # overflows past the window edge and gets silently cropped rather
        # than wrapping, which is why "CNMF cell radius" was getting cut off.
        row1 = tk.Frame(sf, bg="#141414")
        row1.pack(fill=tk.X, anchor="w")
        row2 = tk.Frame(sf, bg="#141414")
        row2.pack(fill=tk.X, anchor="w", pady=(4, 0))

        def lbl(t, dim=False, parent=row1):
            tk.Label(parent, text=t, bg="#141414",
                     fg=TEXT_DIM if dim else TEXT_MAIN,
                     font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 2))

        def ent(var, w=6, parent=row1):
            e = tk.Entry(parent, textvariable=var, width=w,
                         bg=BG_PANEL, fg="white", insertbackground="white",
                         relief=tk.FLAT, font=("Helvetica", 10))
            e.pack(side=tk.LEFT)
            return e

        # ── row 1: numeric fit/analysis parameters ──────────────────────
        lbl("   ENF:")
        self.enf_var = tk.DoubleVar(value=1.2)
        ent(self.enf_var, 5)

        lbl("   |", dim=True); lbl("Fit range (pct of mean):")
        self.fit_lo_var = tk.DoubleVar(value=0.0)
        ent(self.fit_lo_var, 4)
        lbl("–")
        self.fit_hi_var = tk.DoubleVar(value=50.0)
        ent(self.fit_hi_var, 4)

        lbl("   |", dim=True); lbl("Spatial bin (px):")
        self.spatial_bin_var = tk.IntVar(value=1)
        ent(self.spatial_bin_var, 3)

        lbl("   |", dim=True); lbl("Edge margin (px):")
        self.margin_var = tk.IntVar(value=0)
        ent(self.margin_var, 3)

        lbl("   |", dim=True); lbl("Max frames:")
        self.max_frames_var = tk.IntVar(value=2000)
        ent(self.max_frames_var, 6)

        lbl("   |", dim=True); lbl("Start frame (blank = auto-center):")
        self.frame_start_var = tk.StringVar(value="")
        ent(self.frame_start_var, 7)

        lbl("   |", dim=True); lbl("CNMF cell radius (px):")
        self.cnmf_gsig_var = tk.IntVar(value=6)
        ent(self.cnmf_gsig_var, 3)

        # ── row 2: checkboxes (these have the longest labels) ───────────
        def lbl2(t, dim=False):
            lbl(t, dim=dim, parent=row2)

        self.exclude_roi_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2, text="Exclude cell ROIs",
                        variable=self.exclude_roi_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 0))

        lbl2("   |", dim=True)
        self.pw_rigid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2, text="NormCorre: piecewise-rigid",
                        variable=self.pw_rigid_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 0))

        lbl2("   |", dim=True)
        self.skip_mc_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2, text="Skip motion correction (manual TIFF already registered)",
                        variable=self.skip_mc_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 0))

        lbl2("   |", dim=True)
        self.save_mc_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row2, text="Save motion-corrected TIFF (reg_tif)",
                        variable=self.save_mc_var, bg="#141414", fg=TEXT_MAIN,
                        selectcolor="#141414", activebackground="#141414",
                        activeforeground="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(4, 0))

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
            self._progress_row.pack_forget()
            self._mat_movie_btn.pack_forget()
        else:
            self._tab_kurtosis_btn.config(bg=TAB_INACTIVE_BG)
            self._tab_gain_btn.config(bg=TAB_ACTIVE_BG)
            self._action_btn.config(text="▶  Run Analysis")
            self._sort_frame.pack_forget()
            self._progress_row.pack(fill=tk.X)
            self._mat_movie_btn.pack(side=tk.LEFT, padx=6, after=self._settings_btn)

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
            if self.g_plane_dir is None and not self.g_raw_tiffs and self.g_mat_movie is None:
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
        self.g_mat_movie = None  # clear any previous .mat-movie selection
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
        self.g_mat_movie = None  # clear any previous .mat-movie selection
        self.g_results = None
        self.status_var.set(
            f"Loaded {len(paths)} TIFF(s)  ·  {Ly}x{Lx} px  ·  {total_frames} frames  ·  "
            f"no Suite2p F.npy — gain estimate only  ·  verify fs in Settings  ·  click Run Analysis")
        self._draw_gain_splash()

    def _load_gain_mat_movie(self):
        """Load an already motion-corrected movie straight from a .mat file,
        bypassing NormCorre entirely. For when motion correction was already
        done elsewhere (e.g. MATLAB), or when NormCorre's own memmap write
        keeps failing (e.g. 'OSError: No space left on device' -- that error
        comes from np.memmap() writing a multi-GB scratch file to disk during
        motion_correction_piecewise, so it means the disk NormCorre's scratch
        dir lives on is full, not that anything is wrong with the movie).

        Expects a 3D numeric array shaped (x, y, time); axis-order handling
        for scipy (classic .mat) vs h5py (v7.3/HDF5 .mat) is in
        _load_mat_movie_array."""
        path = filedialog.askopenfilename(
            title="Select a motion-corrected movie (.mat, matrix x/y/time)",
            filetypes=[("MATLAB", "*.mat"), ("All", "*.*")])
        if not path:
            return
        try:
            arr, varname = _load_mat_movie_array(path)
        except ImportError:
            messagebox.showerror(
                "Missing package",
                "This .mat file needs h5py to read (MATLAB v7.3 / HDF5 format):\n"
                "  pip install h5py")
            return
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        T, Ly, Lx = arr.shape
        proceed = messagebox.askyesno(
            "Confirm movie shape",
            f"Loaded variable '{varname}' from {os.path.basename(path)}.\n\n"
            f"Interpreted as {T} frames of {Lx}x{Ly} px (x, y, time order).\n\n"
            f"If that looks wrong (e.g. frame count and pixel dimensions "
            f"look swapped), cancel and double check the axis order the "
            f"movie was saved with.\n\nProceed?")
        if not proceed:
            return

        self.g_plane_dir, self.g_ops, self.g_F, self.g_Fneu, self.g_stat = None, None, None, None, None
        self.g_raw_tiffs = None
        self.g_raw_shape = None
        self.g_mat_movie = arr
        self.g_mat_movie_note = f"{os.path.basename(path)} (var '{varname}')"
        self.g_results = None
        self.status_var.set(
            f"Loaded motion-corrected movie from .mat  ·  {T} frames  ·  {Lx}x{Ly} px  ·  "
            f"{os.path.basename(path)}  ·  NormCorre will be skipped  ·  "
            f"no Suite2p F.npy — gain estimate only  ·  verify fs in Settings  ·  click Run Analysis")
        self._draw_gain_splash()

    def _ask_raw_dir(self):
        return filedialog.askdirectory(
            title="Locate the raw acquisition TIFF folder (ops paths didn't resolve)")

    # ── gain-mode: analysis (threaded; this is the only place the movie loads) ──

    def run_gain_analysis(self):
        if self.g_plane_dir is None and not self.g_raw_tiffs and self.g_mat_movie is None:
            messagebox.showwarning("No data", "Load a Suite2p folder, select TIFFs, or load a .mat movie first.")
            return

        max_frames = max(50, int(self.max_frames_var.get()))
        if self.g_plane_dir is not None:
            Ly, Lx = self.g_ops.get("Ly"), self.g_ops.get("Lx")
        elif self.g_raw_tiffs:
            Ly, Lx, _ = self.g_raw_shape
        else:
            _, Ly, Lx = self.g_mat_movie.shape
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
        self._progress_start()
        self.root.update_idletasks()
        t = threading.Thread(target=self._run_gain_worker, daemon=True)
        t.start()

    def _gain_progress(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))
        self.root.after(0, lambda: self._motion_overlay_tick(msg))

    def _progress_start(self):
        """Start the photon-hits-neuron busy animation.
        Safe to call from any thread (marshals onto the main loop)."""
        self.root.after(0, self.gain_progress_bar.start)

    def _progress_stop(self):
        """Stop the busy animation and clear the strip.
        Safe to call from any thread (marshals onto the main loop)."""
        self.root.after(0, self.gain_progress_bar.stop)

    def _motion_overlay_start(self):
        """Open the big centered motion-correction overlay. Called via
        on_normcorre_start, from the worker thread -- marshals onto the
        main loop. Safe to call even if one is already open (no-op)."""
        def _do():
            if self._motion_overlay is None:
                self._motion_overlay = MotionCorrectionOverlay(self.root)
        self.root.after(0, _do)

    def _motion_overlay_tick(self, msg=None):
        """No-op unless the motion-correction overlay is currently open."""
        if self._motion_overlay is not None:
            try:
                self._motion_overlay.tick(msg)
            except Exception:
                pass

    def _motion_overlay_stop(self):
        """Close the motion-correction overlay if one is open. Called via
        on_normcorre_done, from the worker thread -- marshals onto the
        main loop. Always fires (even on NormCorre failure) so the overlay
        never gets stuck open."""
        def _do():
            if self._motion_overlay is not None:
                ov = self._motion_overlay
                self._motion_overlay = None
                try:
                    ov.close()
                except Exception:
                    pass
        self.root.after(0, _do)

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

    def _show_mean_projection_blocking(self, movie_array):
        """Shows the zoomable mean-projection viewer and blocks this
        (worker) thread until the user closes it, so it's seen and
        dismissed before the CNMF prompt fires. Same main-thread-marshal
        pattern as _ask_yesno_blocking, since Tk windows must be created
        on the main thread.

        The viewer asks the user directly for the cell diameter (px) they
        measured; if they gave one, it's converted to a radius (diameter/2,
        rounded, floor of 2) and written into cnmf_gsig_var right here, so
        the CNMF call that follows uses it automatically instead of relying
        on whatever was left in the Settings panel."""
        event = threading.Event()
        result = {}
        mean_img = np.asarray(movie_array).astype(np.float32).mean(axis=0)
        try:
            default_diameter = max(2, int(self.cnmf_gsig_var.get())) * 2
        except Exception:
            default_diameter = 12

        def _show():
            def _on_continue(diameter):
                result["diameter"] = diameter
                event.set()
            MeanProjectionViewer(self.root, mean_img, default_diameter,
                                  on_continue=_on_continue)

        self.root.after(0, _show)
        event.wait()

        diameter = result.get("diameter")
        if diameter is not None:
            new_radius = max(2, round(diameter / 2))
            self.cnmf_gsig_var.set(new_radius)
            self._gain_progress(
                f"Cell diameter set to {diameter:.0f}px -> CNMF radius {new_radius}px")

    def _parse_frame_start(self):
        """Parse the "Start frame" setting: blank -> None (auto-centered,
        the original default behavior), otherwise an explicit non-negative
        frame index. Invalid/negative text is treated as blank rather than
        raising, so a stray typo just falls back to auto-centering."""
        raw = self.frame_start_var.get().strip()
        if not raw:
            return None
        try:
            v = int(raw)
        except ValueError:
            return None
        return v if v >= 0 else None

    def _run_gain_worker(self):
        try:
            max_frames = max(50, int(self.max_frames_var.get()))
            pw_rigid = bool(self.pw_rigid_var.get())
            save_mc = bool(self.save_mc_var.get())
            frame_start = self._parse_frame_start()

            mat_movie = getattr(self, "g_mat_movie", None)
            if mat_movie is not None:
                # Already motion-corrected -- NormCorre is skipped entirely,
                # so this never touches disk the way run_normcorre's memmap
                # write does.
                total = mat_movie.shape[0]
                start, end = contiguous_window(total, max_frames, start=frame_start)
                mat_note = getattr(self, "g_mat_movie_note", "")
                self._gain_progress(
                    f"Using pre-loaded .mat movie ({mat_note}), "
                    f"frames [{start}:{end}) of {total} -- NormCorre skipped")
                movie = RegisteredMovie(
                    mat_movie[start:end], "mat_file",
                    f"{mat_note}, frames [{start}:{end}) of {total}, "
                    f"already motion-corrected (NormCorre skipped)")
            elif self.g_plane_dir is not None:
                movie = load_registered_movie(
                    self.g_plane_dir, self.g_ops, max_frames, pw_rigid,
                    progress_cb=self._gain_progress, ask_raw_dir=self._ask_raw_dir,
                    save_mc=save_mc,
                    on_normcorre_start=self._motion_overlay_start,
                    on_normcorre_done=self._motion_overlay_stop,
                    frame_start=frame_start)
            else:
                movie = load_registered_movie_manual(
                    self.g_raw_tiffs, max_frames, pw_rigid,
                    skip_mc=bool(self.skip_mc_var.get()), save_mc=save_mc,
                    progress_cb=self._gain_progress,
                    on_normcorre_start=self._motion_overlay_start,
                    on_normcorre_done=self._motion_overlay_stop,
                    frame_start=frame_start)

            self._gain_progress(f"Computing PTC from {movie.source} ({movie.shape[0]} frames)...")

            exclude_mask = None
            if self.exclude_roi_var.get() and self.g_stat is not None:
                Ly, Lx = movie.shape[1], movie.shape[2]
                exclude_mask = roi_exclusion_mask(self.g_stat, Ly, Lx)

            spatial_bin = max(1, int(self.spatial_bin_var.get()))
            margin = max(0, int(self.margin_var.get()))
            mu_flat, var_flat, mu_bin, var_bin, n_bin = compute_ptc(
                movie.array, spatial_bin=spatial_bin, exclude_mask=exclude_mask,
                margin=margin)

            fit = fit_gain(mu_bin, var_bin, n_bin,
                            self.fit_lo_var.get(), self.fit_hi_var.get(), self.enf_var.get())

            # Cell traces: existing Suite2p F.npy, or -- any time there's no
            # usable mask at all (no F.npy), regardless of whether the movie
            # came from reg_tif or a fresh NormCorre run -- offer CaImAn
            # CNMF so the flux panel doesn't just stay empty.
            F_for_flux = self.g_F
            cnmf_used = False
            if F_for_flux is None:
                self._gain_progress("Movie loaded — showing mean projection "
                                     "for a cell-size check...")
                self._show_mean_projection_blocking(movie.array)
                movie_desc = ("the registered movie" if movie.source == "reg_tif"
                               else "the motion-corrected movie")
                want_cnmf = self._ask_yesno_blocking(
                    "Run cell segmentation?",
                    f"There's no Suite2p F.npy for this dataset, so the photon-flux "
                    f"panel would otherwise stay empty.\n\n"
                    f"Run CaImAn CNMF on {movie_desc} to detect cells and "
                    f"extract traces? (First-pass integration — sanity-check the "
                    f"resulting footprints/traces against your data before trusting "
                    f"the photon flux numbers.)")
                if want_cnmf:
                    gsig = max(2, int(self.cnmf_gsig_var.get()))
                    fs_for_cnmf = max(0.1, float(self.fs_var.get()))
                    F_cnmf, npix_cnmf, mask_info = run_cnmf_segmentation(
                        movie.array, fs_for_cnmf, gsig, progress_cb=self._gain_progress)
                    mean_img_for_mask = movie.array.mean(axis=0)
                    self.root.after(0, lambda: CNMFMaskViewer(
                        self.root, mean_img_for_mask, mask_info))
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

            self._progress_stop()
            self.root.after(0, self._render_gain_results)
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            self._progress_stop()
            self._motion_overlay_stop()  # safety net -- should already be closed via the finally in _normcorre_then_maybe_save
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
