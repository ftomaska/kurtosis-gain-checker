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
APP_VERSION = "2026-07-15.18"

import os
import gc
import base64
import io
import webbrowser
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
from scipy.ndimage import uniform_filter1d, binary_dilation
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

# ── greyscale GUI-chrome palette (buttons, cards, sidebar) ──────────────────
# Deliberately separate from the plot-color constants above (C_FIT, C_HIST,
# etc.) -- those are data-visualization color coding and stay as-is; this
# palette is only for buttons/panels/chrome, per Filip's "ditch the colors,
# greyscale for now" request for the redesigned Gain Estimation layout.
GREY_BTN        = "#2b2b2b"   # default/inactive button fill
GREY_BTN_HOVER  = "#3a3a3a"
GREY_BTN_ACTIVE = "#e8e8e8"   # primary/selected action -- light fill, dark text
GREY_BTN_TEXT_ACTIVE = "#111111"
GREY_BORDER     = "#333333"
CARD_BG         = "#161616"   # grouped-settings card background (vs BG_MID chrome)
CARD_BORDER     = "#2a2a2a"

SLAB_LAB_URL = "https://slslab.org"


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


def photon_flux_per_cell(F, gain_true, fs, baseline_pct=None):
    """Photons/cell/s from raw F -- no neuropil/background subtraction by
    default. Matches the Wilt convention: F0 is defined to include all
    detected photons (signal + background *light*), so subtracting Fneu
    here would be inconsistent with that definition.

    `baseline_pct`, if given, is a DIFFERENT correction from Fneu
    subtraction: the low-percentile value of each cell's own raw trace,
    subtracted before the photon conversion. This targets the digitizer's
    constant black-level/dark offset (present even with zero real photons
    -- visible as the nonzero floor on the PTC plot's Mean-ADU axis), not
    optical background light. Left uncorrected, that offset gets
    miscounted as detected photons, and since F is a per-ROI pixel SUM
    (matching Suite2p's own convention, see footprints_to_raw_traces), the
    offset's contribution scales with ROI size (offset_ADU * n_pixels) --
    exactly the kind of thing that produces implausibly large flux numbers
    for bigger cells/masks. None (the default) keeps the original raw-F
    behavior for backward compatibility."""
    if not np.isfinite(gain_true) or gain_true <= 0:
        return np.full(F.shape[0], np.nan)
    Fc = F.astype(float)
    if baseline_pct is not None:
        f0 = np.percentile(Fc, baseline_pct, axis=1, keepdims=True)
        Fc = Fc - f0
    photons_per_frame = Fc.mean(axis=1) / gain_true
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

_ART_FILES = {
    "neuron_neutral": "neuron_neutral.png",
    "neuron_annoyed": "neuron_annoyed.png",
    "neuron_surprised": "neuron_surprised.png",
    "neuron_sleepy": "neuron_sleepy.png",
    "chameleon": "chameleon.png",
    "photon_red": "photon_red.png",     # chameleon's outbound shot
    "photon_green": "photon_green.png",  # neuron's return volley -- real
                                          # green-inked art (not a
                                          # programmatic recolor of the red
                                          # squiggle)
}  # kept for reference/documentation only -- the actual bytes are embedded
   # below, not read from these filenames on disk (see _ART_DATA_B64)

# Base64-encoded PNG bytes for every art asset, embedded directly in this
# file instead of living in a separate art/ folder. That folder had to be
# copied alongside kurtosis_checker.py on every deployment, and in practice
# that step kept getting missed (copying just the .py file to a new
# machine) -- which raised "RuntimeError: Missing art asset '...'" and
# silently killed the progress animation's redraw loop. Embedding the
# bytes means this single .py file is fully self-contained: there is
# nothing else to ship, symlink, or forget.
_ART_DATA_B64 = {
    "neuron_neutral": (
        "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"
    ),
    "neuron_annoyed": (
        "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7727NjQ0WDvttJP697//zW3bNgHIveGGG5xddtmlxHEcbdt2m40kMMaQE/hYt27dCm+99dbok08+me7Tp0/aZIGmTYAxxo866ih+3nnnfSOEcF3XbdWI588jYsiQIcVnnnlmbluEea9NBjAMXvCj9x8h7YjecB2f2ezVBjCTMW/+bth220AISyjdvERBuq4OFYSRjCca/nLGxXVvP/VCCcBCwrJiB15wcnzB6YtKMskUd9IZcNuCkhIFxYX4/vOv1V/OuEQn6mOCC6GlK/H999/bSqlMPB7Hrrvu6r755ps8J+jI3//+9/KEE04ok1KKtgx2uUzg81dnEbvvvnvh4sWLMWHChJV+BSW01szfVDZy5ZVXDnz44Yelbdustef0zB55Z599NoqKihr8tgjze7QAHAEKeIS0O3rDdXxm/m4EgCHe2pbIDJk0JqikZIyxjWYoypWIlhSz2tVr5E0nXIAv/vthIQBd2q0iftqtVwXGzd4hWreqGoxzMM69JcUiYfz49bfJv5xxsUzWxzjnfG0LAWPMBaDmzZun3njjDeHPo2nXdeWee+65/Oyzz9amaKS9cc7XtiZUVFQE33777dJjjz027bquzMlAYVmWtXDhQv3qq6+uyskAW+05KKVQVFQUuuCCC5JKKeW/FgLe7/JIAFXwAiC9BwlpJ/Rm6/jM72g2Y4xrrVVhRWmqqndP5mQy2UUpN0BKiWhpMb5458OVNx5/obNq+Y+FAPiA8SNqT7rxcrdr/96BRH2MCcubY9Naww4EkEmmUzef+ttYoj5ma8BRSqX9QhA9e/Zs98ADDwy9/PLL3LIsZoYT58yZk3jkkUe6K6VCQogttn0PsLYpHZzzwM0332yfcMIJDY7juEIIbVoItNahgw8+OFhfX59mjK23J3BT+UGPnX766ZUjRoyQ/lJnJssrB/BreMGPCscIaScU8Do+cxbehfkZ05BJY0VBSVGBm3GBXwgq0nV1UVkJPnjpzdgtp/0uXPvz6hCA1DY771B77B9/WxIpihYn6htggh3g7yxrW+n7L7muIV5bXwlA77777msikUjazIPde++9waeeeso2GZyUEttvv33qH//4R5RzbgPYosHOMEOLUkpxww03FJ1wwglJKSUzwVAIgeXLlxeddtppcc6525oFLCaACiHYmWeeqQDkZnkKwDEABiK7yDQhpI3RG61j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"
    ),
    "neuron_surprised": (
        "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+5bICSEdFgVeQkhrxuCFHskYu08pdZAwRN0JN15qVXTrIvKZnC6e/aqkRDSZwAd/e8NetuDbMGPMue222xwAhv+y+QY3HixMKxQK7mmnnZY78cQTk9XV1ea+++6beuedd+ScOXOSSikz2KUtUBwelVLqV7/6lSwpKanfRri5/JYIBIvagoVqQb9uc5imCcdxMHz48PCf/vQnG4CjtWaGYeCbb76JPvnkkzJog9ieigP39OnTjZdfftkNh8OFhtDLxT8f/Uv4pd8+tiZeWuKoJu5fui6iiZgxYeYPNADNOAsq+RfDW5y4fQ+aENJhUeAlhLRm3iI1xq5nnJ1mhEznxJ9eHhk4ao9O2bo0hCE2SLBCGMil0uqtJ583GWNi5syZetq0aZVBu0IgqKYahoH333+/bq+99rIffPDBhBBCX3nllZnXXnstNHjw4JjrutofS7bBQcXjcdswjBwAjBw50p43b15cKcWaG3aD2buc82BBmlyyZEn2o48+wscffyxt23aFEPUL6DYnCL1TpkyJXn311Y6Usv64b7/9dheACtogtjfTNOG6LiZPnhx9+eWXdSQSsbzQC805N//+0J9K//Py67KkshzS3TC/Ms5h5Qps3CFTE7GShKOU5mBMARgO4CBQlZcQsp1Q4CWEtFbB+LHTOefXKamys84/OTtyyt7hdE2tFsaGOUhJiUgyjo9efdtev2K1yQVX11133fdeyg/6eIUQ6s4776zbd999I5988kmsT58++RdeeKFw8803lxiGEVFKwfDn+dbfhz/q67PPPnPy+bwJQF9wwQUOAB5szbs5wfU45+6XX36Zu/XWWzMTJkzIDxgwgI8ZMwajR4/GiBEj7Ndff70qGC/WHH5AZldddZXu1KlTneM4YIzh008/NT///HMnaENoCUGld7/99ou+9NJLiEQiLvz+Wy5E5Imf3R367+vv5hPlJRuEXsYYXNtGWadKPmzS+Bw2HN92mf+WRpQRQrYZBV5CSGsUhN2DuRD3KSnV9BOPxMTDD0zWrauu31gioAEIw0A+nXHeeuoFxRhjo0aNrhs1ahSK2xGKZutmjzrqqOzFF19cYtt2+LDDDku/8847esaMGfHiFoOmMMb0Pffcox3HMSORSGbmzJkAmjeRIajqptNp67TTTivstttu7Iorroi/++67CaVU1A/C4quvvoodeuihoZqa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"
    ),
    "neuron_sleepy": (
        "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1qX+TXZScAMmFVopCaiWOCIopbySkpLswIEDM0cddZS79957y6OOOqqosrLS8bdNAPDyyy+vO+qoo8o555GAVQVEhFAolJ03b54YM2ZMNJPJnGLFSu+iWwzKFksHYKwpDwG4nBjzlJTOuP33WXnJHd/pm25OhLti59n2MEGinuem7jjnulSysbkCfpyF36kX+++/f+Pbb79dJITg3TFOxYiH/fff35s5c6bTllAJh8PehAkT1syZM2eQSdM2GLGS11OmBc45lixZgsGDB2M7hAo453jrrbdS06dPLzDvlVJ6s2bNyk6dOjXW1eODgmIjbz9lOp1Or1271nvhhRcys2fPZi+99BKvra0FgDKzUlDsbGM9mmRhYWGob9++G84888zKiRMnNhx44IEVgwYNkt/61rcyP//5z4tNo0d/35QQgiZNmrTmo48+qvI8jzuOc7KNWek9dP+0AIvlqzG1GQ4AcAkIQknphCLhhtNuuqxKeF5kmytZ7WKICJ7roqisNLL7tL1TAFqsQUIIxRjDO++8E5o9e3adSU/dxbu8vahwOLy+vRdd13Xmz59fCyCdL0a2JFIALYS2NdsnuG0AeOqppzYBkGaij0Qi60eNGpXwP7tL/+aMMeKcG5GipJTwPE95nsei0Whs2LBhJddee23VI488Urpy5Ur26quvNt95552rDzrooNWlpaUbpZRpIYTcTKS00feQdHNFcM4RCoViiUQitHTp0v533XVX+LzzzqsYOnSomDJlSsZve+AFtymEIMdxMG/evJp//etfGcdxkM1mn2lubj6OiGw2UC/AChVLT8eMmCEA9wLgfnCjPOS8kzeU9+0TySTTftnx7gURKeF6bJ9jDwWA5kCDOoJOW45997vfdaFTlbvfF9SoaDS6xvyd/5pSColEYgTasQ63I0QUAHDO15aUlDRv8w75Ash1XffTTz+tAMCICFJKXHLJJaWVlZV9/Ays7vSbE2MMjuOQcfEY4SKldEKhUMlhhx028Dvf+U6/N998s9/y5cvVBx98IL/xjW+sOP300zNElPC3o4zbhzEGxjmIUUtfHyEEXNc1Lh0VCoWIcx6SUkbff//9khkzZhQACOUfN99lxS644IJUOp2Oh8PhUGFh4fNKqZP23ntv98EHHzR9nTj8GjI788ezdC72YFp6OsblcwOA+4kxoaTkNcMGrf/mn+6p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"
    ),
    "chameleon": (
        "iVBORw0KGgoAAAANSUhEUgAAArwAAAIaCAYAAADC/56wAAEAAElEQVR4nOydd3wc1bn+v2dmtmhVbEnu3cYVG3ebZrApIfQSAiQhBFIgBQgp9ya5aZBGyv0lNwkhBEhCJ3TTmyk2xrh33HuTiyyra9vMnN8fM2e1krVFsiytzDyfz1re2dnZM+2cZ97zvM8r8ODBg4eWIZq9l53SitQQgAbYNG3bQGAacDLQ0122B5gDrHbf64B1nNpDs237gGHAdGA0MBQoBvKBBmAHsAh4FahI2lauHW8PHjx48ODBg4cTAhpgcDTZJc3y5hA0kj896aVl+f1s25iM4cAdwNtANQ5ZbP4KA4/iEE7cNh1rewSN+5eMYuAK4K/ASqA+RZuSXzuA7zbbtgcPHjx48ODBg4d2gkZjdFIhBHQDCltYtzkU8WtORFuCWre1hE4RaIU84PPAG0CUowmkCcTdl5m0fD9wbdJ2DFpHxlPtazfgs8AjwMEW2mO77TBxIsDqZdIYpZbAizjRX/XgkCsQLbw8ePDgwYMHDx6OO5pHUlsio+nQnOiOAP4beBkn4liFQxDfBb6JMz1P0neak1CFEmAicDFwDXAVcAbQq4Xfz6aNyb8xFPg5sImjCW5z8tgS4VTv/wkMaPZbhvtSxzL51RLJDQIXAP8A9rbwe3EcUpuqTc3Xj7n/fyZp3zuKWDaPzBs45zvdw4mKtrdHtNyDBw8ePHjw4OEotEQ0ITsSqaKUCmfhkKww6UnZYmBUC78TAs4D/hf4ADiCQ/Saf78aR6t6XZb7kfwbJwN/o6lkoaUIaaaXiqxKoAyHPA9P0YaWEMQ5Xv8LbORo0hpvZXta+r4Ebs5wfI4FitwqYpvtbyQ/ELQEj/h68OAhJ+F1TB48dD0osmK778cAk9z/LwG20nhvt5T4ZOCQRIAJwE9xpuIVTI6etlYEzgB2AWcDu4HTcWQFFwMntfBbyb/fvL+ZC3wDJ1KbnESmoprq/VTgNhySHExqY0syjNbAopG4RXDI/FK3PXuBSne5DQSAQcA5OMR+TNJ2ZNK22qNPtd3tlAHjcKLsx5rElhy9hcbz33ydbkA/nHN5EtAfJyrfEyjAebDRcEh5HU70fztO0t2HQI27reORFOjBgwcPHjx4yEEcD51jMsG7AHiLxoigxEmO+k3SuqLZd9X7bu56dUnfzSYyqX5rhfvbLX2uIq7Nt6WWKemBxCGWE9w2NU/+mgw8SVMZwrFET7ORObT0eUuRatWWVJ8d60u16evuschGG50MRW7TRWN7ArOA7wEPAPOB8gzHI91rN3A3jc4Yx/Iw4sGDBw8ePHjIcTR3EWguH2gr1Da6AX8n9TS9BP6Y1JbmGtjP0XQqvrUEpznJayvxU3rV7UDfpPaNAf5F00S09ia6qYhvOs1tsia3zb8lhJBCiGwIrw287x6TTA9OzbW3zREERgJfxpGFzMeJHGdzPNRDTEvJdurz5OO1DScKDsdHjuHBgwcPHjx46EQkTxkrBJt93lYoEjMBx0tWEc/mZDVZA3phs9+cRtOobGu1r81Jb1sjgS2R3keBwcA9NI06t6mNQgip67o0DENqmnasRPiYibYQQmqaJg3DaM3vSqAuEAgMlVIad955pyGlTD6f6ZwxAjgPDt8A/g2sS3O+FHFNFZlvTZvVtRfBSVQEj/R68ODBgwcPJwyaT8X/ESd5ay1OktY17mdtmeZVhOYC4DAOoUiWMaSKDi50vzcMJ6qnIqbNo8Gd/VKygZoW9qFV29I0Teq6ftRyXdeziaq2+6slkpufn2/37du3Nov9UwT1CpoilUxhMHA9cB+whpbPsUVTcns89lu1uw6n2AYp2uvBgwcPHjx0KSQnOCVPoTcvOqBxfHStnQ01mPcGHiQ1Gf19s/WzgSK7n6fRQSHbyKoFvITjmNCcjOTqq12Ibo8ePeR1110nb775Zjls2LAm6x3vfVDR5eTfMgxDnnbaafJ3v/ud3LJli1y3bp2liHAqIq5pmgXICy64YJGU8nYp5VQpZfKMQTec5MFfA/No2WFDEdxsrdHa8zxKHLlKH1qe/fDgwYMHDx5yGs31gm0hsMkJNR3pNdqeSNbmnoejXVQDvoqiJUfUJHCTu342g78iu1+hMVrX1shsq4hke0VDNU1rDclss3RBvR8zZoz8v//7P7l//36pUF1dLX/60582IZ/tsW8ttaN5dHnYsGHyBz/4gVyyZIlMRjwel6NHj04co3TnoLS0VP74xz+WDz74oPzd73639vzzz/8r8BhOueTm30smuB1BbDNdcxKnMIjqL7rife7BgwcPHj5BSM76bgn5OJGcMTgepZ8GLgMuwSGDU3EKBvTA0Re2hK5mYK+Oxc00alFTRXeVjGAPTtnZTJHu5G0rMtiWCF2rI6aKtLWV9Cryl/z99pYTCCGaENcRI0bIf/zjH7K+vj5BKi3LkqZpJt7Pnj1blpSUJPbxWKO9yTrh5OXFxcXyuuuuk7Nnz5YNDQ1HEd1YLCallPLb3/62hKwIeBO9tN/vb/758ZQoqOtOtaG53jdb0qtmN9r6gOzBgwcPHjwcVwiOJrkGTsb3DcBfgTdxvEuTk41SDdzlwCrgeeB3OP6vJ9FYwUtBSSFydXBUx+QHNBKDTFE1Nfjf4X43lbRBbfurNB634z4drWlagpgqMpi8LN0rVYRzzJgx8owzzmi3NjYnuj179pS/+c1vZFVVVRNSadt24r1t2zIej0sppVy3bp0877zzmuyzIr/p9lM5KyhNbvP9DIVC8vzzz5f333+/3LNnTxOSa5qmtCyryXsppZw7d25i29kcW/ecqAju8XKtSCa38eQ2qGNlGEbilcVDQ7Lt23fc6zqX72sPHjx48PAJQ3MrKx9Oadg/4BQ3yKQFtZu90q0bB5bhJFZdCpQ2a4tBbun/mpPdbAmpWm8RjYN+84FfbftGGsnHcZ2ebi4LuPjii+VHH30kf/jDHzYhmYoUJpO/liKcpaWl8qabbpKvvPKKjMfj8tChQ7Jnz56J77RHG0OhkLzjjjvk7t27mxDJZKLbHMnR3ocfflhOmDAhJaH2+XzS5/OlJXXdu3eX5557rvzjH/8o161b1+S3VHQ5VXts25amacrTTjvtqH3r6FfyeVSEXp1rXddtTdMO4RTleAL4P+Au4CHcUsqtJL0/aHate8TXgwcPHjx0Cpp7xvYFvo9DclsicMla1WxIn03TKFVLZG4/8AiOJKIgqS3NSXhnQBHSr+O0tbVTyTaOW8I4dzvJ+6O2fQNNPU+PG9lJJlqnnHKKfOmll5qQsp/85CdZkbGioiJ54YUXyn/+85+yrKysCfGTUsp7771XAtLn8x0T0fX5fPLLX/5yE4KZieg2J6IK8XhcvvLKK/Kb3/ymHDFihMzPz0/ZjkAgIHv16iVPO+00eccdd8inn366CdlWx6t5dDkVFPmePXv2UefheL+SI9UtkNVynIppfwW+BIwGfFLKYinlWCnlZVLKr0kpv/fRRx89ee6559aSPelV98kDOAl3Cl7E14MHDx48dBiaSxeG40RzDtF00DoeCTHJJLg5edyIk4U+ullbO2OQVOT0ctqum1SRrluStpl87G9M2u5xI7vJyWTBYFD+7Gc/k9XV1QlSmEwiFy5cKL/yla/IoUOHyuLiYllUVCR79+4tx44dK2+88Ub5z3/+U27duvUoYqm2oab0r7nmGgmOZjWTfKC5nZdhGPLaa6+VS5cubRPRTUU4FWKxmNywYYN8/fXX5aOPPirvv/9++Y9//EM+9thj8tVXX5UrVqyQFRUVR21HkdxkIp0t1Hcuv/zyxD4er/Odzge4X79+8rOf/ax84IEH7GXLlm2UUt4lpRwqpRRSyqullM9ZlnUwxXG0brzxxmzbnxzpXY/jPJL8wJfs6OIRYA8ePHjw0O5IlguUAr+laRUmFW08bgMyLQ+MyWQyDDwNnJPU1vaqYJYN1O9MpPHYtOWYqAH/CXd7yZ6qN9MYKc+47Swr"
        "dh1FdJOjiVdeeaVcs2ZNSiKYTORisZjcvXu33Lp1a1ry15yE2rYtbduWdXV18rLLLmtCYpXuN1kbmtzeoqIi+aUvfUkuWrSoSZtMy5SWbUvLtpJezu9kS4GTyXhrYJpm1pHcdLAsS9q2Lfft2ycHDx7crqQ3neRE0zQ5ZswY+b3vfU++9tprTfTPSftoxuPx/S202bacA2bG43Hbtm0Zi8Xk+eefLyHrSHWyHOojHBeSXinuOXV/tGRt2PzlwYMHDx48tIjkyKIGfBPHNzN5YOpI786WXi1V9HoHx/1BQQ2ExwtqkO0JbElqV1v3RwIraOpW8SMayX7GY55MdDMVVWgpujd27Fj57LPPNiFxqQicZVkpSaFlWVlFONW24/G4/OUvfyl79OiRsr15eXlyypQp8u6775abN29u+ltmPO3vJNZtJRm1bTsRlY7H401eihQfK8FtsZ3ucVuxYoXs379/gvRmmyzY/Bwrgtv8u4WFhfLMM8+Ud999t1y4cOFR50udR9M0bSll8lOP5b5vcefVdsrKyhKkPUuddnO5zgHgcRzbvhFAXpb3pkJzD3CDo8myR4o9ePgEw+sAPrnQaBxsJuPIF852P7PIvQFCkcHkdr2Pk0T3pvtekV67HX9XDaQW8DKOrtgktT1bJkh3m+U47hRR4F7ga+5n6jdTN0gIpJQUFhbi8/k4cuQIQgh0XUdKiZQysZ4QAtM0E98dPHgwt99+O7fccguFhYXYtnOoNC3z80JL227VjkuZ+E5ZWRlvvfUWK1as4NChQ+i6zoABAxg3bhwTJ05k/Pjxie9ZtoWUEkNvPOQV9TXsr67gcH01DbEIAIWBEP26lTKguBcBwzH9kEhETl3GR8OyLHRdZ8uWLdxyyy3MnTsXoMk5BRJ/1WfJf5PPscKgQYM444wzOP/88znrrLMYOXLkUb8Lzrlv4VxmdS0mt/+DDz7gggsuIB6PN7lWMkDdq8kXoIkjZdri/t0K7MJJkqvDKVscwbl3bFp3vxs09iVZNdCDBw8nBnJ7JPBwvKDjEDiBE1n8CY6Prlp2rNHSlgaS9rzWmrfzRRyLs8Xue4PG6NGxwsAZgP8HuNvdbnvIKCzghzglh08ly4cMTdMSxPGVV15h1KhR3HHHHbz22mspv6PrOlOmTOHGG2/k85//PMXFxU4DXKLS0cj2dxUhU+uG4zE+LtvOhgM7KauuIGrGsaWNLd05bSHQNY2S/CKmDhrF9EGjmxyvXIY6JqZp8tBDD/H3v/+dVatWZf19wzDo168fEyZM4KyzzmLKlCmceuqp5OfnJ9aRUiYecNr7vJumiWEY3Hvvvdx22234fD7i8XhrNqFIaKb+pxanBHUNUE9j8mvyKwJU45Th3o1DmLcBZe5nCnrS73rw4OEER26PAh6OBxSBGwzcj1MgAtpG5JpPU0LjgCVaWM9OWke08P/WIpkkWjjODr8CdrqfaxzbYKYeDGYA75Ld1GjWkbEkqMh1WijyBvDoo4/yxS9+MfHZ/PnzeeONN1i6dCkVFRUEAgGGDBnCtGnTmDFjBtOnT0+sa1lWqqheh0FKmWhH8jIpZaJtqn2ReIwVezazbPcmKuprEELg1w1nneRtun9NyyJmmZzUox9XTJhBcV5BTpBeKWWTJzBF0hVs204cD9u2WbBgAYsXL2b16tXs3r2byspKbNsmGAxSUlJCv379GDZsGCNHjmT48OGMHTuWvLymSgDbthPbzSaKfyxQpPemm27ikUceaQvpVUjuL9pLpxvDiRavwKn89h4OIQbnvvYivh48nODwCO8nB2rAsHGm5e8D+uOQ39Y4HiQPDOkIcvOxPd16KmLb1uhyMlk/DPwSRyZg00jwWwvVju7AQpxCG5mIqdQ0TRzjdG7LjdG0BHF5+OGHueGGG7AsK0EMsyFzuUB0s4EtJZrbxjVl25i3eTXl9VX4dAOfpjtsKM3xFTjHIxyP0rOgG1+Ydj4loaJOI72ZfteWEiGcdqsobFsjsLZtJ45NR59rdd2Hw2Euuugi5s+ffyykN+XPpPjb0jrpyPI+4Fng7zjSCTj2B2QPHjzkMHJ75PPQXkjW6/4AZ/pfRUWzGlmbR91cRIE9Qoh1wEZN07ZLKQ9ZllVtWVby1KEfx0+3Bw7JHg6cDAwE+jX7KSupza29PpP35z2cyk5rabr/2UJFdx/F8cXNdKxsTdM027bn45Ds08kycpuxIbqOZVkEg0Eee+wxPvvZzx4lC0jWYyokE+/OkC60BUpzW1Ffw5yNy9h4cDe60PDpOlI6n2cLTWiE41H6d+/Bl6ZfQMDwg0ssOwLNiW55XRX7qg5T2VBHUTCPAd170qOgO7p7zpqvr85pqgcadX5biop3FtQ+VFZWctVVVzFv3ryEDlnJKTqraUmv5AfrGhzP4btxnGDUfe/Bg4cTDB7hPfGhOnAfTiWzW2js+LOeRneJk8Txz3wXJ1FsOY5P71EwDCMxyKWJxpUCY3Gsvi4EJgF9kj43ab0Dg4oYGzg6vh/iSDcg+wiOOmbXAU+RmexKN5Gs6he/+MX1Tz/99K3r16+/JIvvZYRhGJimSa9evXjyySc577zzElPHJxKSyd7afdt5e+NSaiINBH3+xOdtgSY06mMRpg8ezWWnnNFhUd7kKPW6/TtZvmcTeyvLiZnxBOPy6Qa9i0oY23cIkwaMIOjzd4kku0xQMxH19fV885vf5LHHHgOca1lJLHIASjKh7s9lOA+2G/FIrwcPJyS6ds/qIRNUx10CPImj181KwqAyxN3s7+jEiRMPX3XVVSu/8pWv7BowYIAJBIEQIE3TPGJZ1l7TNNdXVlZuHThw4DaSiOVf/vKXwIcffiifffbZluy2ktEbmAlcDZzvthtaMfWfhGSy+RBwO06SS6bBTEWDe+Po/fqQ4eFA13VpWZb4+c9/fvgXv/iF9Zvf/Kb3z372M6lpmlBRutZCCIGmaViWxYQJE/jPf/7DmDFjcorsNtekAkfparOBIoembfHG+iUs370JQ9MwNB27jUS3eZtipskXpp7HiF4DjjvpVdvfX13Bu5tWsO3wPsAhuMm/K6UkblnYtkXPwu6cN2oKY/oMzgm98bEiWY/80EMP8dOf/pSysjKgcbYhw8NwRyH5AfkAjtxrGR7p9eDhhEPX7lU9pIPSrg4AXgCmkaWdVnJUd/r06fzhD38Iz5w5UyVsZfp+Aw5R/BAnOWSpECIMIKVUMgXb9QlNnlpsni09BPgMji/nKe6y1hLf5CjOIuCLONna6QYz9dmTOJWg0kZpldzgiiuu4MUXXwTg1VdflZdddplQutvWQm0T4POf/zx/+9vfKCkp6TRXhebIRMhaQ9hsaaMJjSP1Nby05kN2VBwgzx+ADPIFaUs3GgpkmMpXhLd/9x7cdOqF6Lp+XDq+5P1etnsj72xaQTgWTRulVg8IccvEkpJzR07i7OETTgjSmyy32LdvH/fccw+PPvoo+/fvT6yjaZrTC0j36VdZsCX9v4Og+sZ9wFnADjxNrwcPJxS6do/qIRUU2R0NzHb/ZkV2k8nWHXfcwW9+8xvy8/OTZQ12krG8QrLjQnMy+jHwGvCUEGKVWiil1GkkvsnbURFWNdAYwFXAd3F0sdD6RDu172U4JHoxLSezKbJ7DfAMGciuIrSDBg1i8eLF9OnTRwIcPHhQjBs3joqKCoQQWZNepcO0LItQKMRvfvMbvvOd7wBNI2atQfMomooctxWKiEkp2VlxgO2Hy6iK1OHTfQwu7sWYvkPw60ZWhE2ts/1wGS+u+ZCacD1Bnz9tVFfaNkLT8AX8zoMZYEZjmKaZdr80IYjEY1w9aSan9BvWRHLQHlD7Ytk2r69fzNJdGwkYBrrQsopSCyFAQkM8woUnn8qZw8adEKQXmtrQHTh4kNdefYWXZ7/EwiWLKS8vT/k9IQRCS7H/sqW3Dms+hqix6icW4FR0TC6M4cGDhy6Ort+bemgOReTG4RDNQWSpJVV60T59+vC3v/2Nq6++Gmi1"
        "X2syWU1OPLOAOcCDwCtCiDikJL7J301OYrsW+DGNEd/WaGTVYFYFfBZHh5xMelUmdzFOhHogaaQMijjats2bb77JBRdc0MQ39stf/jIPP/xw4pimQzP5CDNnzuSvf/0r48ePx7btNhd5SJXtr9wdWkN8k7Wl2w7v44Mta9hTeQhL2k2IWZ+iUi4ddxoDi3ulJGzSJSZCCFbt2cpr6xdh2xY+3chIdv15eVhmnLKtu6g8UE4gFGTAqGEUdu9OpCGckiA5Ud44g0v68KVTL0AIrd06P7Wf9dEIL6z+gC2H9hLyB1qUfKSDOr5x2+SG6RcwtLTvCUN6lXZXyXHWle/m+XlvsXXDRg5s282BHXs4cuAQ9VU1ROobiEVi2G2UAwFoupYgv60kwKqf+BHwezxpgwcPJwy6fk/qIRmqcx6HIycYQBakMFkvevrpp/P4448zbNgwTNN0pn+PbcBVut3kNqzASaB7RghRDwniK4UQLYVDkwedfOA2nEIQ3WhdVTh1LOqBK3FKFKttK/L7F+DbZCll+OEPf8jvfve7xEOBIqgbNmxg6tSpRCKRxLFNRjLhVJ/16dOHH/3oR3zrW9/C5/O1WcLQvKLZqlWrOHToEH379uWUU06hX79+id/NZvuJ6KW0eW/TChbtWIclJQHdR9NLQxA1YwR9Ab4w9bwWSa90LbhAMG/LKt7fsgqfrqMJLS0xsW2bYCjErnWbeOOf/2HbyvWowFuP/n249Js3MHbGNKINYUQKIi9wdMI3nvZpBhX3bhcyqSQZ5XVVPL9yHvtrjhDy+bHanGQniJpx+nfvyZdPuxBd63wJy7Ei+TjvrSpnzvqlbK/YTygvj0AwmCCltmlRV1VNfXUt0XCEaH0D0XCUeDRKLBwhFolixuLEY3Hi0SjRcIRYOEKkPkxdVTWR+gYidQ3UHqlq8vuapmE7P5LNyVYR3SqcB+v9NNo5evDgoQvDI7wnDhRxOwV4nVaQXTXtfsMNN3DvvfdSWFh4vPSizSukrceJ+D4qhDgCTXS+EocAJzOHZOJ7EvBH4Ar3fbYWYMmk90IcrbEPp0LTacAHNMolWrw/FNmdOnUqH374IYZhNPE8VfKDf//733z1q18FaGL8r4ouKPTp04cbb7yRb3/72wky2lYJgyKNlmXxq1/9ivvuu6/JtHGvXr249tpr+fGPf0zfvn0znmdFViLxGC+s/oANB3YT8vnBlTUcdWyEIGrF6Z5XyJdPu4iCoFMIQXnMKjnE6+sXs3jHeoJ+P0KmnzOWtk0gL4+tq9bx8I9/TzQccXW7zpalbaMbBt/4vzsZPG4ksXCkRdKrud68pw8dx4UnTz8mWYPSmAoh2FGxn9mr51MbaSBg+I450U4IQTQe47op5zKmz+B2l190FJKPUcyMs2D7xyzauZ6oGSdg+JC2nThWSout6zqaoSPc+yURsU+47iZFzdX/XcIcC0epr66hqryCyv3l7Nqw2V6/aEWk+sBhHQi4157KHUgH1UfcjVOF0ovyevBwAqDr9aIeWoLqkEfgyAYGkwXZTU5O+/Wvf81PfvIToO1kqxVoHvXdgaOZfUQIsSF5RTfym3h711138Ytf/EKjUYpwG87UY4jsJQ6KHB8EzsUh3gFgLg7pTUme1QNCMBhkwYIFTJw4scXjpZY999xz3Hnnnaxfv77J54WFhYwbN45rrrmGz33uc/Tt2xc49nK/6nzedNNNCTso5YOq2gXQr18/7r//fi699NKUv5k8Vf/U8nfZVXmIkC+ALdMHuzSh0RCLMHnQSK4cP6OJhMG0LV5cPZ81+7aT5077Z9ofTdcxo1Hu+dZPKd9T1kRnrvbPsixGTpvAV3//P8SjsRYjt0IIYpZJv6ISbjrtYsfbl9Z3gk2T0zbx9oalWLaNT28/V4loPMbYfkO5ZtKsNrWxs5F8jLaV72POxuXsrzmM3/CjpXhYUt9z/0PCCBGOOgCi+TvhSoMMHd0wELqGJoSsr6sLL5y7YPfbjz0bMfdVjtYgaAuRKdqrDvkhHNvEwzQ+hHvw4KGLoqv1ox6OhiK7A4G3cRLUsiK7tm3j8/n417/+xQ033NBmvegxoDnxjQLzcFwl3gd2CiFizb8kpdQfeOABbeTIkfK8884zNU2bbprmQzjFLLJKzqPxGK0HJmua9kXbtv9JhmOnNLl/+tOf+O53v5uWoCrSG4lEWLRoEbt378ayLHr37s3w4cMZOXJkY2PaoQqaasuTTz7J9ddfj9/vxzTNJklzyXphn8/HQw89xPXXX3+U3Zka8RtiUf6z7B12Vx4iLwuym/gdwLRtvnTqpxlU3CsRJX5u1Tw2H9xDyB/Malu2bZNXEGL1+wt57M4/ITQN2SwJUAiBBLr1KOG2v/+agu7dsEwz5bHUENw841JK87u1Wtagoq1xy+TN9UtYtnsTfsPIKMloDQRgSZt8fx7fmHG582BA1+isk6O6UTPOvC2rWLxzAxKJX/dlff20+felbEKWDZ+BPxigIly35aVnn9+y7uFXRmMyTAghZXrSq/qBrwH/ou0VGz148JAjyA1DTw9thYbTMfcAXiRLsqsiYgUFBTz55JNcdtllneXvqsKiivgGgAvcVxjYLKXcDqx0XzuBXUKIWpKmGG3bXtK3b9+Z5eXlj0opLwKkZVmZ+IHu/u7JvXr1mh+NRodUV1dn8tvFNE0uvPBC7rjjjgRJTblz7kNFMBhk1qxZR32uEss0TWsX+Uiy76lye2juECGlTDgaxONxbrrpJkpLS7nwwguPIu+mZfHCqnlOZNcfaJXFmormfrT9YwZPPY/qcD3PrZzL7spDWZPd5G0d2L4nYeHVIq2UjvRS2ulJpxCCiBnnSEMtpfndsm4DNNXrvrRmAbsrD7oR71YnRqWFpDFKfqC2kqGlfZz9y3FZQ+LhQQi2V5Tx9vqllFVXEPT5HdnUcSa7QOL3E5mylkVDXT3FgbwRX/rSl/oumzZp8XM//n8RWdFwskgTaVa7BHwJ+DeepMGDhy4Pj/B2XSj7riCOHGAyWUQ3Fdnt3bs3zz//PGeeeWYuFDNoyYs3D5jgvq5yl1nANinlJhxz+OXAzn379h0YMGDAYeBi4E/A7W6Z30y6DA2wDx06NC1TA9Xg2KtXL+67776EHCRTdFCt15wsqqS19tRJq7Zs3bo1Y3a6ItqmafKlL32JJUuWMHjw4CZR/lc//ojNh/aS7w9gtdJP2JYSv26w+8hB1pZtY9GODeytKifP52898ZHgC/ic/UlzvG1bOrZlmXihlFQ21KlNZ4ycJgyjhcb6/Tt5ff1i6iPhBNk9HtCEoCEeo6qhFkr75HwFNhX5jlmmG9Vdjy1l4uGmswpMqGs5Eo1IPa4VnDZ6/Hn97//D5n/ecdeBun2Hewsn1NvSV1XfcSowBljPzJkGc+dZCE/a4MFDV8RxFWp6OG5ITqh6DMczMmuy269fP954441cIbvJEDiRV51G8mslvXRgJE41pF8ArwKr+vfvv8SyrA8sy3pASrn5yiuvXGXbtub6BWeCMpdPu66K1v6///f/GDJkSMLaK6udcmUEya/jqZEuKSnJavvKtqy8vJxbbrnFWeaS+A+2rmHFni2E/IE2Ow4oZ4cXVy/gQO2RjB67qSCBguLumVdMeA5n8P8FIvFodr8tneIWtpS8t2kFz66cRyQeJeA79uS09BDYSOpi4cT7XIQii5oQ7Kk8xCOL3mD+1jXoQnNt5nLD3EDTNGGDbKirY1DvviO/9qsfdQ91K5RAqvtYOTMEREC/FikF8+aZCbJ7pzd2evDQ1eDdtF0PihRaONZen6UVZHfAgAG89tprTJo0KdfIbnMoNwed9CTYAIZpmnaWpmk3A/fNnj176ve+9z1s2xZZ7l9aWzN17K666ipuuOGGnKl41hwqkevaa69NeJ5mIuVqX+bMmcNDDz+Moets2L+TuVtWutHYYyd1hqZhCP2Yonzx6FFS7qOgvF6z4YYZlA/uOg75b4hFeHrF+8zdsgq/YaCL"
        "9klOSw+JhlMhDtqP7kpkow3YMUox1PGxbJsPtq7msSVvU1Zd4XgQc0wFII4LBAhN16mtqpZDTx4RnDjrDE1KmdLGDnd8FD7fzSX/c/mtpf912RW9vnPJMEDwi8RMVG4+iXjw4OEoeIS360HHIbg/AL5FFmRX+cAOGDCAV199lYkTJ+Y62U2FjCTYsizbsiz++Mc/cvPNNye8hNv8g0lShj//+c+tTnJSA78iGMkvpzBB+5ECJZ/49re/zTnnnEMsFkNKmXH/lYzhF3fexZY9O5mzeUW75qO76tpj2kY8kjkiK21Hw5vp7AjAr6e/9h29rmB/dQWPLH6LjQd2JRWT6Dgi116kUeJccwKRmObXkhJUW/M7yVHdgzVHeGzp27yzcTkA/nawZTve0HRdxCMxhk08GeCoJMgkCEDa9dE+5say6SLkP83UxddLf3jZN3r912Xj3XWkF+314KFrwLtRuxYU2f0cjhVX1m4MvXr14tVXX2XChAldleymQhMSrOu6pvb5vvvu4+KLLz6miKwqJvHrX/+aQYMGZbRsSya4ajo8mWAkv5wkrEbCcazkRpGXvLw8Xn75ZW6//Xby8/OPKnpxVJtdEr97z26+/6ufEtMlusgdoaIQEI85hDetl5Rtu+QlA+UVgsKEP3AL25ESTWis27+Tx5a+zaHaynaLdmcPgUTiM5zr9lh+WbpEVxOC2kgDW8v3sWrvFtaV7aSs+jC2tBuJb4ZfspMe+BbtXM/Di99kZ8UBQv5A4rdyHUIITDPOwDHDCRaE0mvDHQszLfLxvqGa36gT2KaUDLU1vtjzB5d+uff3P9XLjfZ6Y6kHDzmOE4b1fAKgZAwzgIdoLHubcnRXxK+oqIjZs2czYcIELMs6kchui1BRWV3XeeSRR5gxYwabNm1KHI9soVwZzj33XL761a+mJc5qoFdZ4gqReIzaaAPheIyoGSNumWhCw6/7CPr8FASCFAXzW6hG1raZUrXvBQUF/PWvf+XWW2/lueee46677kpb4lj95vvPvcLET59FqFshlpm9Tvn4QmQnaZAu4U3LiiW60OjRgkODOgaOhnk1czevQtO0dikm0VaEfME2fzf5mqyPRViwbS3r9u+gOlKP2h2frtO7sITJA0cyaeCIhEdu8/OulmlCcKShhrfWL2Xjwd34daNTj09bIITAilt061HCwFHD2bJ8DZoQLe+Da10W31sxzjxUtUAE/XFhE5VCSDROjut5g7r91xUvVf+/l1a5eYVd50B48PAJw4nNfE4cKLI7BHgKx5khbVQh2RLr+eef54wzzuj0yG6iKhIZOIn6XNDmzHQl4+jRowdPPPEEZ555JvF4PEEIM0Gtl5eXx5///OcWybKK5iZPDYfjUcqqK9h15AD7qg5THamnPhomEo+5CTyN++PXDfJdwjuguBcjevZnUHFv9KSKbG0hnKrttm0zatQofvKTn7B69WqeffbZo4o2JPZFSjRNo66yhlXvf8Ssz19BQ3UtorO1yu6pimUpaUiQlhYuMlVauHuogO6hArXQWd091nHL5LWPF7JizxbHTosUROg4QyLRNZ18f9sIb3J1trVl23l/8yoO11XhNwwChh8gkYxXVl3B3qoFbDy4iyvGz6AgkNfEFSJ5W0t3bWTe1tXURRoI+vyJKmddDbZtkZeXz0mTxrJl+Zp0qzpFAiPx7g0rdowouHDCKnmkPg9dt6UtwwIR0LG/0OO/L+12WLw6zyO9HjzkLrxpmNyHchHIxyG7/clAdhVJ0nWdp556ivPPP79dyW5zTWomXWPz6f3EdH6KV/J0f7JEoLVQEdopU6bw61//ulUV5BTB/e53v8spp5zSxHNXtcmxqnJI1NbD+3hpzQL+ueA1Hl/yNvO2rGZb+T4q62swbQu/4SPPFyDP5yfP5094k9ZFw+ypPMSCbWt5bMnb/GvhayzeuZ5IPJYgrm0ZPZUzRDwex7Zt/vu//zshz0hFoiUSBCx7cx6R2nr0HJoJMONpotPuX9uVNKR6SBICTNuiX7cebhEN2aTkcU2kgSeXvsOKvVsap+g7ibtIKQkYPrrnOcQ81WOPerBR90eyvrayoY7nVs7lhVUfUB2uI+QPogutyf0L4DcM8vwBNh3cwxNL36EuGm5C9DUhOFxXzZNL3+HVjxcSiccI+vxtvjZzAULTiEdjjD51ErphZDXzE1uz91RZF/Gj67baCkJaQiOK4LLS/770PISn6fXgIVeROyOah5ag7MdsnGo/p5JBt6v8XS3L4oEHHuCKK644ZrLrZHbTRI+qGtccSuMnaDpFDM70fnldFYfqqqioq6Y22kAkHsOyLXRNJ+gL0C0vn+55BZTmF1Ga342iYKiJRMBOIs7ZQEU0v//97/Pyyy8zf/78lFFOBXX8Ro0axQ9+8IMmRDnZXL8uGmZt2XbW7tvOgZojWNLG0HT8hq9JgYSEprel9gkN3dAT6++vPsK+qsMs272Js06awPj+w5r+bivh8/mwbZtp06ZxySWX8PLLL6eO8trOb+zftottq9Zz8plTidTVp8ti7zDYaXXILtlzk9ZSsUOnoINgXJ8hjdt1i0nsqz7MC6s+4HBd9XH1180GiUprvjxKCxzpRfNzr2zxkqvzqevUsiyW793MB1vWUBt1I7GQ0iJMXZ/5gSD7qsp5ec0CPjf1XDShEbNMluxcz8Id66iPRtyoLl1KwtAShBCYsTi9B/dn0Jjh7Fi7scUqfgAqamvXNPQPf7hlTMGFp6y23Civ+kzaIiyQF5V897L6I794ZRF3oiU5OXjw4CEH4BHe3IZKUvsFcB1Z2o+Zpsmdd96ZcCloK9lN1gAmj7fV4TrqY1HiVhwpwacbFAbzKArmJ6Y+1ffilsWOijLW79/J3qpyKhvqiNtmIgqXPIzLpP9pQqMoL5+SvEKG9ujHST370a9baWL7TTSzaZAcOf7zn//MzJkzqa+vTyttUNHdm266iW7dujVxelBR2eW7N7Ny7xYqG2rRNQ2fYeDHKYjVmshXsswDnGibQFBRX8MLqz9ga/leLhp7aiIiqbWB9Kr9/N73vscrr7ySiPK2tP9C05CWxdoPFnPymVNzpqRt2gicbFzHTiStNW25kiv0LOjO0J59kyKhGhsO7OLltQuImvFOSE5rAUJgWTY9CroRMHxHfZzsvFFTU0NNTQ2h/HxKiouptMK89fFi1u/Zjk/XW7U/lm0T8gfZeHA3q/ZuozS/kDfWL2F/dQV+3Wizj3KuwrYs8grzGX36ZIfwkiY50P0gsnz7OcFJQ7aJgmCUuKk7SW0INKSUIiIMcXnRf11ZXvOLF7d58gYPHnILHuHNXSiyez3wc7IsGWyaJrfccgt33XVXm90JmkdmayMN7K08xNbDZRyoOZKIzMYtEwkYrtawJL+I0b0HMqH/cHRNY+WeLazcu4UDNUewpcTQdAxNx6frKUcBRVEkUB8NUxOuZ9vhMj7Y6qNvUQkn9x3M6N6DKQ4VJtoK6Ymv+mz8+PEMHTqUtWvXJiy8Uu0/wLvvvssPf/jDRBQtblms3LOZhTvXU1FXjV83yPP5G2UXrTnIKaDkIT7dQACr9m3lYG0ln5l4Nr0Li9tEenXd8cA9++yzmTp1KkuXLk0T5XWI5cbFK6kuryBUWOBEVzszeU2AbWUOlimXhpaaKnBKJU8aOAK/7kscx6W7NvLm+iUIIdxCCZ3PTwSOt+2Q0j5AUw2tujdnz57NP//5TzZu3EhdXR3BYJA+fftQNHwAQ6aOZdTEUwCINITR9Owj9La0Cfh8vLNpGaZlEbdM5xpPkkCcKBCaRiwc5ZSzpvP+47OJNIRJw3oFQkg7HCupmb3k091vOe95GTN1kKKR9ApbStvwSevq3t//1L0HxZwG0m3RgwcPHQqP8OYmVJLaacCDZOHIoAjMBRdcwN///vfE9GarPGObEd0dFftZvW8bOw7vpyrslGLVhEDXdDQh8LvRJykl9bEw1ZF6tpXvZfnuzRiazt7q8qOm+J3p0zRt"
        "SN4noSem+20p2VtVzq7KQyzY9jGj+wxm6qBR9CkqSbQB0hPfQ4cOsWXLlibrtwQVAf3www/ZtnUrw0eMYGv5PuZuWcWeykP4ND3hyXq8SIAi0CFfkEO1lTyxZA7XTJ7FwOJebZI3KHeOL3/5yyxdujTt7wohqD1SxfZV65l8wdk01Na1KbLcnpCZCK97gbUUCRY40d1ehcVMGjACcK7jdzct54OtawgYvoSWNxdgSxu/YTCouDdAQh6kXrfffjv33XffUd/bvXs3LF6C/vTLjDltMud+8UoGjx1FpK7eiXdneQ4FgrhpItx7/EQjugpCCOKxGL0G9WfMGVNZ+c58Z3Yn1bUmpUAI29x7ZELNfxZUdf/i2e9YVfVBLKmhIbHRBFpMaHavuB24EHg+xytCe/DwiULni/M8NIeGQ3YHAk8DebSYc94IRXbHjh3Lk08+2WT6PRuoqKIqBbu2bDsPL3qDx5e8zco9W6iPhROJVn7Dh54UbVIkQRc6AcNHyB+kor6aA7VHCPkC+HSjzVWdVFWoxuQaHyGfn6gZZ8mujfx74eu8sPoDDtQcSRD1dL+hdLiZjovjWKATiUR44ZWX+HD3Oh5f/Bb7qsrJ8/nRNafSVkfQAFvaBAwfdbEwTy1/j/3VFYhUFkppoK6Jz372s5SWlqYtjaw0u1uWr80dEpjBSzjhKGDbR0ejBZi2zYyTxhH0+YlbJrNXfcC8LatdJ4bOS05rDhXdLc3vRv9uPZxlQiQeYO+9917uu+8+DMNIlKhWWl5d19F0Dcs0+fjDJdx3x128+/gL+AKBtDMaLbajmXToRIV6mJh24SwQjg48LaTUEMjYhrKZVQ/PvYiAYYnueWF8hnOBSluXlowgmV70rQuGI/DKEHvwkCPwbsTcggqEBoAngUE45Det/ZhlWZSWlvL0009TWlqatRtBovqS64iwfv9OHl74Bs+v+oBdRw5iaM6UvS60RjeGFEQvmZwaupGYHm7PAVNtXxOCkJuIs3rvNv698HVeXbuQyoa6lAO1lJIePXowefLkrKqPSTfB599PP8EHW9fgM3yJaFdHkyNbSvy6j3A8yjMr36cq7ERcW9MORZp69uzJdddcC5DaU9iNkm5d+TH1VTXoht7pk7LZXkeOS0MjhBBE4jFG9RrIxAEjqI9F+M+yd1m5b2tS5bTcgRAQt0xG9R6I4UpR1PVaV1fHH//4xwR5tSwr4dBg27bz3o1Oaq4Lwev3P8Gzf7gPoYlWk95PAoSmEQ2HOWnyWEZNm5Cw50sLV5sb23LgjKp/zPlCZMnWYeaByiIEUhTmRfWi/IheEooGBhbPQBMyqTBF8yqR3vjrwUMHwpM05BaUbvdenAITGR0ZAAzD4IknnmDs2LFZ63Yb5Quwp/IQ87Y6NlogCBq+rOQH6bZ9PKHaBiSScpbs2sDGQ7s5fejJTB88JkG4VeRXTen/8pe/5NOf/jSmO2WbSce7b/N2wkdqyO9ehOV+pzNgSxu/7qOyvpaX1yzgC9PORxdaqzLK1D4NP3UC/CN1Ipha78j+csq27mT45HFEG8IIkcPjsxAgpUP4hHA1QE60NN8f5JJTTqc6XM9/lr/L/uoKQv5gq4qQdBRsKQn6/JzcZ3DjMvcBdv26dY5sIZvtuLIcoWksfXMulmVx3f/cioxlLuDxiYN0HhDO+fyVbF6W5ayGS3qtw3XD615dOVz4jTph6GFREKjSi0KHteL8w0ZJwcjAlGEHoku3LQBq02xNydUkeM4OHjwcL+TwCPaJg4FDdr8HfNX9f8YkNdu2+fOf/5wgcRkjl24kVrkNvLFuMY8sfout5XvxGz78RvtHZo8n1NR+yB8gGo/x1oalPLzoTXZU7Hf8fGnMardtm/POO49nn32WwYMHp9XCqs/qq2vZv2M3voCPNrH/doQtbYL+AFvK9zF386pWSRvUMdh2uIwj3QTde5am9eTVdB2Q7Fq3Gd1ndP71kMmNw/3bRH8pIG7GuXrSTOKmyb8Xvc7BmiOO40UOkl1NCKJWnOE9+9OnqDQReRY4D22r1n+ckC9kAykltmWhGzor5szn7YeeIZgfysl970wITSPSEOGkyeOYcM7pTpQ3m0S/JBcGGTML7IZoT+tQzYjY1gOnR5Zuu6zurdXXRZdvnw2sAZbiSNR+A3wJmIkzgwcOybVo9Ffv5GovHjycmPAIb25ARXYvBv4frXBk+M53vsOtt96alf2YUyzBGTzX79/Jvxe+zsId69CEIGj421zgIRfQKHUIUFZ9mCeWzuGN9YsJx6MJYqjkH1dddRXvv/8+xcXFQGpNr3CngA9s3+3+vyP3qGXYtk3I52fhjnXsrDiQKAWbDcLxGK+vXUhBSXdGnToJILXHrrvN3Ru2EI/GOt2LN9vAum05BSo0IaiPhvnUmKkEDB//XvgatZEGtwxubhI+KR1f5mmDRgONUVobm4V7N/LBjrZpqi3LiRC/9/hsNi5aSTA/1LLf7CcYQoAVN7no5i9QWNId20r9MNgEMslYRqiXsBHCBiS2DOJUyJwKXAv8GHgEmAusAFbjJCZ/ERhMI/kVeOOzBw/tCu+G6nyoJLUxwKPuMlVwokWoJLVLLrmEP/3pT9i2nYUm1Y1YRiO8svYjnls5l+pwPSG3dOmJkIktcfWuhg9d6CzcsY6HF73BtsNlCZcBoWmYpsnQoUOZMWNGes2ee0wO7NiDbVqd6szVBC7JfXvjUswsLMPUuX9/8woO1lYS0H2Mmj7BTdJJL2vYvWEr9VU1TsS3U6+R7A6+ZVou2Y0wfcgYSkJFPLLoTaKWmTO2Yy1BCEHUjDGiZ3+GlPZxIvKaxoGaIzy25G3e/ngxQyee7JAx20ZzE9Y0XXMS19I9kLg6ZSklr/7jcaL1YTeC70FBOTaU9uvNVd/5aqK4TCskTKKR8koNKZvLFKykl+kuLwXGA18DHgOW41TTPCfpe96J8uChneAR3s6FhtOxFeF0dKU0WpC1CEV2x40bx2OPPZZYnm5qXn2+6dAe/r3wdZbu3ohPNzB0PWejXccC5ToR8gUpd0uivrtpBabtkCEl2Thzxpnpt+P+Ld+9r+Xs/06ClBK/YbgV2TYmbNtSrSuEYGv5Ppbt3kQoECQSiTBo7EjyCwtTyjrUdVNXWc3BnXsxfL5O5bvZEg/bsgnHY4zqM4ge+d14YfUHCa/oXJ69kNLxXj5r+ARHWw58tP1jHl70BrsrD+JHp1tpMdf84BsUlRZjuwlrtmUnSiqn3b6rA96/bRcr3plPIJTnSRuaQdM0wrX1TDjnDC7++vWJY3qMun0VqU1OVjM4mgybOP3/dcB7wEvAJBqjvbnR+Xjw0IXhEd7Og0h6PYrzpJ+VI0NxcTFPPfUUxcXFaSOUKmnLtC3mbFzG08vfozJcS8gXbJL4daLCljY+3UDXNOZtWcXjS96mvK4Kv8+HEIIzTjvDWS/VwO8en5ojVUTqGrLWTnYEpASfrrNwx3rqomHXtaEFCEfK8OaGJeottmlR0L2IAaOHueuk0PG6+7t/+y4Mvy/hXJHLCEfCDOjekx75Rby3eQUg0HPcnUAXGuF4lMkDRzKge0+ONNTyn2Xv8NaGpVjSJmD4kQKi4QijT53EbX//DZ/5/i2ceul5jD1zKidNGkv/EUMyRm3VEVj08hwi9bl1PecKNF2jobaOc75wJRff8oWEzKs1xTtagWQybOCcIsv9ezmwEPgpjeTYi/Z68HAM8FwaOg9Kt/sn4AoylA1WbgO6rvPoo49mdGSQrqb1SIOT1b+9oow8I+BUreoCxKW9oIhOyB9g15GD/HvhG3x69FTG9R/GPurQ/T6sWDxtPaRYOEJ9VQ0l/XtjxuKd5tSQDIlTua6yoZbluzcxc8TEo6K16v07G5dzqLaSkFue2LZt8gIBhowbzaYlqxMRxaPgbmv/tt3YptmpEe5sj3lJsIBuefks27YJ3Z3uz2WyqwlBOB5lQHEvLhw7nZV7t/DOxuXUxyJHVTgTQhALRygq7c4Z"
        "V1yApmtIWxINRzAMg+f++ABLXns3URq7OaSrCd6/bRc71m5g1PRJROsbOl2fnWvQNI1IfQPn3XA1pf168+JfH6L2SFXis+OY6yBoJLUWjj3lr4BTcRKZD9FYlMiDBw+thNfTdQ6UI8OtwHdxOrC0Dx9KyvD73/+eSy+9NKUjg3IVFUKwbv8OHlr4OjuPHCDkCyQcGj6JsKUkYPiIW3Fe/Xgh/170BuuO7KWkd0+gsXBBS4g2RAjX1jkRsRw6fFKC3zBYsWcL9dFIkyiviu6v37+TFXs2kWcEkoiTU3lt8MkjEJpIWdRBXStlW3cSDUe6RETQtm22lu9DCNBEbpNdgTP7UhgMMWvERF77eBEvrv6QqBknmPB8bvYdTcOMm4Tr6qmvqqGhpg7LNEHA2dde0ihVSHE5C83Rf29avCr1g44HhBA01NYx4dwz+dZff8mUC85GN4yE77EQAk3PoJ0+Njg2Kc44cSnwLjCaLBKaPXjw0DJyfwQ78aAiuxcCf6HRiiYlDMPANE2+/OUv8/3vfz/hKdscyoXBlpJ3Ni3nuZUfEI5HE4PnJx22lOjCSfY5UFVBQUE+RT1SOzUoumGZJtFI1B3ccuc4JqK84TrWlm13lrn6ZU0IqsP1vLlhCXqzQVkIgRmN0eekwYQKCxPLjv4BZ18P7ztAQ3Utut6JOtgsI7xH6mvc1VtXmKMz4JwnjSGlfZi3eRWLd65vUuglFZQ1meZWVtM0jUhDmL5DBzHmNMd9I9XDidrs9tUbMaMxtFz2Vu5kKE1vcd+efO7Ht3Hrvb/i9CsvoKhHsWv5ZruFTgQkXBqE7VqVtcfFJ2gMjowD3sGJ9nqk14OHNsCTNHQs1HTUWOBxGpPW0joymKbJWWedxX333ZeyilrChSEW4eW1C9iwfxd5/gDIE8OBob2gSJChaeiGQSAvmG7lRHGKWDSWiIh1vqChERIwhMbqfduYMmgUPt0pe2xLm5fXLqAmXE/QLc6RgFt1LZgfoufAvtRX1yQKNzTZtvs+Fo5weN8BuvcqxYybnXIAslVTiC50qTvJdBqbD+0lbprk+4NtvlfVdTp+1umseu+j1CVy3e1XHjzEkQPllPTthRnPDZlOLkLTNcxYHFPGGDByGIPGjOC8L36GnWs3sXnZavZs2mbu37rLkR84jrxHH0iBdK7MRMheOThkCwNn3OgPvIYjgVtAIxn24MFDFvAIb8dB2Y+VAs+6f9NGd1WS2qBBg3jiiScIBAItFgtQHrRl1Yd5cfWHHKytdCpJfYK0uq2FlBLcKBmQevhxtb3SsnKSFEgp8RkGB2oq2Fmxn+E9+6MJwZxNy9hSvjeh2z3qe7ZNIJRHn6GD2PnxppTT20ITSFtyeN8BRk2fiGwII3I4uNSVnAcEYLqlkH3GsVmmObZacQaOHk5B927UVVW3WElQMa5wbT3V5RX0HNQPMxbLGQeSXISyKIuFo0gk+d2KmHjuGUz+1FkyXF8vXnlvzpzFCz46EqyODYodrukpG2LdZCReJGPxwoRVWUtEmEQkOJuENBUsKQVexklq80ivBw+tgEd4OwaqswsAz+B47mYsGyylJBQK8fTTTzNw4MAWk9RUctrasu28tm4RMTPultvtOgN/50AgbRvbdPWrKbmGw3hzOXCobMk+3r+DEb0GsGz3Jj7ats6pKJbGrkzTdXoPGaCWtLxtoSGxKN+zv1ND262xJetKSFQtOMZZGCEElmlSWNqdvicNYsvytS2XzpZO1NK2bOprat0krNyatchVCE0gcI6zGY87RWBCIf28s2dO2h4KP4Chb0FKYVU3BKyahqCsiwbjh2q62xW1xXZ1Q0+zvCZs14QNYBjQh6Z2Y42F9VJDkd4SYDZwEY53r5fI5sFDFvAI7/GHsp6xgH8C55KFI4OSMjzwwAOcdtppR1VSkzRWTftg62re37wSQ9dz2lw/pyCUNjeSfj33WGo5pt9NhpQSQ9fZXXmID7et5YOtqzGM9HpbIQRWPE7PgX2dal6piKLS8e4tI5bQMecwPsHXvrRt/MEgvQb1Z8vytRlZbLi23vmPx3ZbBeWYo2maiEajdnGosP/5/cee+8zy994rye8eQmDqJYUN9Cqq843se0gTmiV8IiADvg8Ofe+xdymglDp64BDfYcA0nHLDKtqbDentiUN6LwA24pFeDx4ywiO8xxfKZsYEfodTPjIt2QWHXJmmyS9+8Quuv/76o8lukr/u6+sWsXzXJgI+f8sRHQ9Hw/UujkWi1FfXOotSkFm11PD7UusiOxkSx8u1LtrAe5tXoGt6WtcJAITAMi2Ke/fE8PuIR2Mt63jdI1Cx74BjyaZpn2hSmdsQSMuiuI/jPJLpNJlxbya8HSDi8bg8eeBJp40v37tx48Hd+wt8wYAVNwVxdCnjWLaUIPPQ6QNI6jkMHMYhqgpLgXtoHekdiCNvmAWU4ckbPHhIixwP13R5aDgd0PeAH5KF/ZhhGFiWxRe/+EV+/vOfHyVjUGS3LhrmP8veZdmuTQT9gcRnHjJD4kzrxsIRag4faVzYHAKHHOsa/kDAdcHIXQgEPt3Iqo0CJ3GtoKQbeQX57rIW4B6X6sOVhGvrXWu2Dr7OJJ7GNBsIpw8oLHWcRzKdJ6F5x/RYIYQQlmUR9AcCs0ZOusBn+DQLaSOERAiJJiSGbgtDt6WmBwDBz9FoWnDCB/wNuMXdbDYuD4r0jsCpytaNRrKr3B10vPi9Bw8JeIT3+EFl1t4E/JFW2I/NmjWLBx98MOHIoPSLylu1vLaKRxe/xdbyfYT8QY/otgGarlNVXkFDTR3Q8sOCipL6g0HyCkPOtH+OE6+srwUBUtoYPh9FiiC15EzmjrvRcIQjBw6hG3puB3hz+/QcV6hd9/l9Wa0fzA+lLbjiITsIIUQkErH7lfQceu6IiVOrw3URPdlKRzoJaxrC0UXddVRJYTXr9yCON7vmfp4N6TWBqcDzwBlAiEb/XlW1TcMjwB48eIT3OEFNLV2B04nZZKiHrjS7o0aN4j//+Q/BoGOX1Uh2bTQh2HXkAI+6JXKdpKSulaSTC5C2jW4Y7Nu8A8icEOUPBsgrLOhSDgBZwZbohkFhaXcgRfENZc1m21SXH0HT9RxP4ftkQwJ6Cx7dLSEQDJxQZNe27cT1qt53YDBAxONxOWnQyHNG9xncpz4WiWnNO5bUmihFUA3gPpzZQJ3sSK/hrnce8CGOTOJ5nIJGM3AIsE3LBNgb/z18ouBd8O0PRXbPB56g0YkhJatS9mOlpaXMnj2bPn36YFlWwm/XIbsa6w/s5Mml79IQi+A3fB7ZbTOcU7Fn4zbnXapELPeMhYoKyO9WiJ2j1mRthZSgGTqhwgJnQarqXO4+N1TXOjZuOR3i/WRDgKPHTgNFAg2f4fy/q1/S0tmnvIJ8dJ/uekULQoX5GIaB7IAHVVfaIEPBvLwLxky7WBOasKWUmpPhJgFsTUt/Yholb38A7iZ70qsiwgJH1/sZnJL1H+AQ4KeB24HTcZyCFAFWB0bHi/56+ATAS1prXyiyOxPnKTufDFKGZPux5557jjFjxjTR7Tq2YxpLd23kzfVLEEJgdGbFq64O6UQ16yqr2b1+i7so1YDozPcW9+6JYfiIxiMnlO5RItF1PaHhTcN4AairqukcRYfIXqqhtVBu+5MCCSAE0XA4/XruoTT8vhPi4UUIMPx+lr05l2VvzePQzr0UlhYz+tSJnH75pyjqUUIsHD3u964QQguHw7JfSc+hV5xy5qynlr87pyS/KJQ4xpIGta50ZA7KwSexePny5WLBggWB7373uz+RUnaTUt5KFonONBYxSn4ZOAR4IHAtzli0G5iPQ4Y/AtbT1N1B3UCe44OHEw4e4W0/KLJ7Bo5dTBFZkF31evTRR5k1a1bCkSHZdmzullXM3bwSv+HznBiOERKJz2+wZ+NWyveUOctSzDQK4YwavQb3R+gZi+J1PaiEvHTV5pLQUFOLbVmdomPONkr3SSa8ajq/9kiV874F1w21"
        "zPD7CIRCTiGbLn1NSzSfjxf/8m8WvvR2YmlNRSX7Nm9nzdxFfOmX36fX4P7E3WqJxxNCCKLRqBzbf+hZM2smlM3ftnZ9cV5RwLJNdEStlFK/a+5dQpwjTEhoeZORTDRv03W9GPiCbduWlDLTxd1cNtcSAR7ivm4AIsBm4E3gLRwCnOzTqPJQvAHHwwkBj/C2D5LJ7ktAMVmQXSVl+Mc//sHVV1/dSHbd5DRb2ry+bhFLd20kaAQA6ZHdY4S0JYbPx7oFywBHTpJJm9tn6KAT1qBfCJFFkpNbYjgSdQlSxyNb/bSmf5JVWs4DTOWBw0DjA1syVI6aPxgkEMpzHiS66EVt2zZ5BfksfOktFr70dkJuo2QamqZTvqeMZ//3fr7x5zs7RI4khBBSSimEEDNHTrr0YG1VxfaKskNFgfyCmkhDuRDCApBSBoBR7qsv0ANHb1sB7G5oaNi0YcOG9VOnTr0eqAW+ruu6tCyrNTuRjgADBIHx7usHwA4cm7MXgbk0uj5kK63w4CGn4RHeNHCnnTRAzJ07l7vuuivx2bx589R/BRAHzsGRMbSK7N599918/etfP4rsRuIxXlqzgHX7d5DnD3hEtz0gnYSe+uoaNi5eoRa1DOFU7dINgx4D+2LGzVw3aGgTHB1vdt1APBZzouGdEeHNsoLaJznCKzSNeCRK5cHyjOsG8/Mo6FaIZXbdmWvHSzvC4lffS8x8JWYCJFi2idA0dq/fzKYlqxh/9qmEa+vd2ZrjByGEME1TBv2BgkvHnf7Zx5fOeXhvxf7D0waNqVsi5WW2bV+JU4CoH+BvaRuhUCg+ceLEg1LKt4FXi4qKetbW1l6haZompRRtHA9SEWAbhwsMBe5wX0uBx3H0vwfd9T3i66FLwyO8R0M888wz2rXXXiuFEMo6JhMuxCkZXEgryO7PfvYz/ud//gfLspqQ3ZpIPc+tnMfOigOE/KnLw3poHWxpEwyE2PnxJg7t2udoQ1NEDgUCiaRbrxJ6D+6PGYvlfpWxtiITf3UvPzMW75TiG4LsI7zNS29/UiClo8euq66hYt8BZ1lL58qVNISKCskrLCAWjXbNREwJmqERqWvgSNnBlB7ZQggQgq0rPmb8zNM6TJQkhBCxWMzu2a2k92cmnHVtfSy69p5r77gRGK417UdsmsoalIuCT9f1AcBXgK9UVlZu/epXv1rzyCOPdAOErutY1jE/rCTriJuT32nu68fAQ8ADOBFg8Kq6eeii8AhvIxJV0a699lrL7/cTjUZ7AlNqampGbtq0qeeyZcv0VatW1a1Zs6Zh27Zt0fLy8jDQH/gpzvRQRq9dRXZ/8IMf8Mtf/jLhxqDIbkV9NU8tf5/y2kpC/qDnxNDOELpg1XsLAFfOkCpy6BKD/iOGkVdQQDTcgBAnHuEVbnGNbGBbdidZkomsibZufDIJLziR+tqKKioPOpKGlr2lHVbTo39vt2oeXVTSIBP/po12uhKHQ7v3uaWxO25nhRBaJBqRfbqXDi0u6DYUXQBIG9vS0DQayWZLHYvaKcu2bV3X9eEPP/wwU6dOlT/84Q9paGhAd5OX28kusSXyK4HewI9wimL8DfgrjuwieT0PHroEPMLrShZwTcCllNo777xz0dq1a2/4zne+86kdO3aU7Nmzh8rKSurr64nFYolXM6gn85QQQmBZFrfddhu///3vE4UlJKAJwe7KQzy/ch41kXqCnsduu0JKic/n40jZIT7+cKmzLA2JUsRg+MSxaIaGtEGcoFwq2xKzup5FyeLjBMvKro2+QOA4tyQ3IW2JYejs2bjNsc/TUjwkuKev16D+aLpIJMd2PQhHjqNraXXb6gGt7kg1lmkmggsdCCFtKQ9VVsh/zH2Rz005V+vbrdRQAY5033P/Gm6egZRSittuu01MmzaN//qv/+LDDz8EUPaV0rZt9fhyrCc0eRsSZ2wsAX4OfAH4H+A593Mv2uuhy+CTTHhVL6lYZY9u3bp9ZsSIEbc0NDRMqaysJOzY+6gbXqGlTkUjQyejkqNuvvlm7rnnHgDHfszQ0YTGxoN7eHH1B8Qtk4DnsdvukLaNPy/IugVLHT/ZdMlqwim76wv4GTxuFFbM7JrTvllAWjZmBt9WBc3QOj4aKMC2raxJueFvURL5yYCmsfPjTYBrd9hC8E2RvV6DB+Boejqyge0PgUhfaMPdP8s0OzMPQvgMQ9RGG3hy2TtcP+1T9CkqIQvSm4CmOaFpy7I49dRTmT9/Pk888QR/+tOfWLFihQ1ouq4LV+aQLJMQSX/bQoZVmWI1Dg4HngWexClucQiP9HroIjjx5mgzQ0kXVKcwHmeaZl1tbe39W7dunVJWVibD4bCl67p0S/sa6oVT91yVaczasFt1tm+//TY/+tGPKCsrw+fzoQmNFbs389zK9zFtG0M3PM3ucYCm60TqG1jx9geZ13XPZt9hg+k9dICjczyB/HcTEALLtgnXNWReFyfhT+tgHbNAYJkWdgbCq+4v3edk6p+AZys1pET3GdQeqWLnuk3uohbkDMKJ+vqCfnr074NlduFETEcdgNAEWhYyls7eTyklft2gPhrm6eXvURWua5PFpK7riQf166+/nkWLFsnZs2fHZ82atdKyrDW6ridXUksep1RQRlV1a23ymSK+atz8Ao6Tw6k0Fszw4CGn8UkjvDqNT6pjgH/jZKPeDvSybdsSQtjCeezWLcsStm0L6erAjiVCoL67a9cufv/73zNt+nT++eCDLN69kdc3LnHKvHb8dNsnArYb3d2+ej17Nm5zLN/S6d7c0fHkM6fgDwY6JVGrIyCEwLYswnV17pIU++kej2BBvqv77MDjoYFtWVhmNoRXYBifvAdGKZ2qaQd37uHwnv2k1jw757Fbz1JK+/fBjHbxREwJCIHhy4JrCa3TH4JsKfEbPiobapm9aj4xV6bT2j5fyTIsy8Ln83HllVcG3n///T6HDh26x7KskwOBwM1CiLtxxrfZwDycghMRGomrIsDJFdey+nn3ZeKMoe8DN7rvT1DRl4cTBZ+kpzI17ZKH4zn4fRxXBWi8WfXjTTiFEOi6Ttm+fdx8yy2MmD6Bz37vFnr060VDbf0n2lbpeEHN4y19433nfYbIim3ZGD4fo6ZPxIqfuHIGIQRW3KSushrIzGNDhQXohp4yI/54QEV4TZfwpmuibmjO9LbssplYbYKUzvW6cdFKADRdYFstRXidQ9Nv+FDyCkJE6hu6NuFF3dtpzrUb0wzkBRBa51eotKUk6AuwvWI/czev4oIxU9vUJjWOSCmFZVnSMIy+PXv2fFBK+QMhxP+qmZikbecBfXBmNE8DzsZxYVAm3MqhIdsBSBWlyAMexklu+wNeMpuHHEbX7u2yg8o8tYCzgA+Bu3DIrqp2Y9BBI6SUEtN0SJSm62xZspoHvvtLtqxYR6ioMLVrgIc2QUqJLxDgwI49bFjoeu+midiqgWLo+NH0HzGkwzO7OxJCCEzTpKaiylmQYeDNKwgd/0YlQxVgsSzsuJIItjxVD06pXMNnJAoPfFKg6ToNtXVsWrLKWZDqPLrHZNgpoxBusmxXh5RgpZmtUWQ4kB9ykttyIPpvS5uQz8+inevYUbH/mKpnCiEwDEPgJK3ZwB8ikci9tm3rZ599tnHNNdcoAhvGsRV7CSfpbAYwEcd2bBWNUr+Wqr+lQvL6vwfudv/fHolzHjy0O050wqueNm3gv4H3gMk0lkvsMKLbHFJKbMtC03Uq9h/koR//njXzFhEqKnDKt3poF0g3+Wz1+x+55FVLa62lPpl03gx0w9fpEaHjBikRmkakvoG6yqr0q7qEoqC4m3ttdswtI8HRGZsmZjyetLBl+AMBfCewBKUlSNvGFwywd/N29m/bBQjsFPtvWzaG38egsSMd/e4JwUlkViQ2kBdA03OI5Lsk971NK4jb7dLfCzexzQoEAt+SUv5z1qxZ9jPPPCOllLqUUr///vt9f/nLXwK33357QEqpaZq2Xgjx25kzZ04DrgLe"
        "pWmAKJvDpcithUOkf0UW9pwePHQGTmRJg4Zz44VwTLOvp1G/mzO6Adv14Y02hHn8F//H53/ybSadf6bjJODJG44ZumFQV1nNyncXZFzXSeqxKe7dgzGnTyYajpyw0V2J41lbeaCcSDjiLGuJOAhnudAE3XqWYpl2h0o8NE0Qj8aIRqMZ1/UFA/gCgZTFRE5ESOmcx7XzFiOlU1q4ZTmDhpQ2vQb1p8/QgcSjsa5/bbtyhbSzYu4u+oPBDk+4TAfp6nl3Vx5k7b5tTB44slWuDSmQ8JIHbrrrrrssIcTXkj5PMGvlFAQwb9480zCMF23bflEIcYVlWT8DpiR9J9NAlEySf4oTTb4bh19kZ6/iwUMHIHd6gPaFcmHoC7yOQ3aT64LnFGzbRmiOnvKZ393LpsWrySv0Ir3HCttyktU2L13N4b37E4Q2FdRgM/mCsx1yFz9RomBHQ9pO2eSKfQeQlp1Sy6n23+f3071XqXNNduAhEUIQj8WJR9IQXve8+QMBfAG/qzE+Mc9bE7juDDWHK1n/0TJ3UctBOVUzZfiUUwjkBdurWEHnQTb+ST8L414beUFndieXZmwk6JrOkl0biFnxhPNEO0ARza9KKe+TUo6SUn5aSvl5KeWNUsrrpZQXSynHSym7AZimiW3bWJb10jXXXDNL07Sf4RDXbC3Hkknvb3AqxJmc2EE1D10MJyLhVTfdIOAtYCaNN16rRkFVBljXdXRdx7UoOy4RLmk7UbRYJMoTv/wz+7ftIhA6AQamToRwSwevev8j532aCI8QwkkoyQ8x5YKZJ64VWRKEEBzcvS/x/3TI715Efrcip6hBR0V4pXPOzGiMmEt40xEWX8CPP+B37pkT+9QBYNuSQF6ATYtXUnmgPHWxCRp168Mnjs0t0ncMEAiktBvlLmmg5eC9LJH4dIP91UfYfHCvuz/ttnllIfYN4GPgTRzv3IeBx4HXgJXAVinlHCnlL2Ox2Bl33nln8Nlnn62zbfvXRUVFs4QQy2gMIGVqnSK9NvB34Hw89wYPOYQTjfCqm60v8CpwCm14ynQTARJlGy3LwrIsbNtO2JMZridpew7+0pZomkZ9dQ3P/P7vxMLRRPlID62DlBLD7+fwvgNsXbEWIG3EXLgp7FMuOJveQwY4U74nqDsDOIlOsUiU/Vt3AWmIpHsMeg8ZgD8Y6OAHMGeKNx6NpXXLUEt9eQGMwAmsu24GoQnMuMly5S2d6vhojl60tF9vBo8dSSwczanp/bZAIsG9NtITXudaiEViSGnn5HOQEILV+7YlNOvtCDUeqvHPTnopv94eOMT0Zz6fb8Fdd931oWma39u5c2ffmpqaJVLKGYZh/FM01lXPhvQCBID/AENxAlBd+4LzcELgRLoIVYJaNxzvwTaRXUUwTdMkEAgwfvx4rr76ar70pS9x5pln0qNHD6BxCkiR3/YiR7Zto+lOidDX7n8cf17wE5WE016Qto0/GGDjohVE6sNpB3gV3Q2E8jj9ygswY/ETmuyCEzkN19Wxb8sOZ0GqqfAE4R3oehJ37IyDEIJYJOK+Sb9uQfeiE/68Kajre++m7Wxfvd5JgkqhZVXHZOyZ0ygo7oZ1gkilhCaIhaMJDW+LV7C7sL6qBsu0aORtuQFVkGJ35UEO11U7suT2fWBT46L6v3qpIhQqr0VJ/qbouv7HwYMHL49Go7+WUhaYpnmzlPIONylOkNnFQc2y9gCewLEug0/EvIuHXMaJoq9JvpEewqn+0iaya1kW+fn5fPOb3+TGG2/k5JNPbkKW6urqWLFiBcuXL2f27NksXLgw4RGqyPKxRsFUItuil99h9OlTGHvGFMK1DWlrxntoCk3TiEWibHC9SdNadbra3umXnEvfkwbTUF13Qh9r27YJ5gXYsXYX9dU1icS0lqAIbu9B/dO6WxwPOBEvCNeqSnBqjG4ZhcXdc8F1qkPgJKsZrJwzH8u0Uier4UgfNF3n5BlTHUlKxze3/SGde7yhthYrUYWvhVLK7t9YLOYEDnJw54UQhONRtpXvo2dBt0Tt+vb8iQzLkyUHKvrb1+/3/8S27Rvq6+t/m5+f/1fbtnfouv6CZVlKLpGuk1TJc6cDfwS+hZfE5qGTcaKM6kpj9Gsce5U2k90ZM2awaNEi/vd//5dx48YlqtooEltQUMDZZ5/Nd7/7XT744AMWLFjAt771LUpLSxOyB6X7PdZok5SS1/7+GPVVteg+T9qQLaSUCXeGvZu2A6S0asKtupbfvYgZV19MPBLNSb1fu0KCphvsWrspIaNJuap7LPuPGoYZjXdCoQJBbVV1VmsWFnc7zm3JEUiJ4TeoPFDOalefnjLy7lbG6zt0IINGD09Y83V1SJzrtqGmrlGq1LLJCADBUJ5znedoH6oJje0VZUBmPf3xbgpJFUk1TRsUCoXui0aj70spP7Ys60xN0+pdiUOmyI4ivd/EKUXs6Xk9dCq6fs/XeFNdhmOi3WrbMUV2r7zySt58803GjRuXkCxAY/IakCjpqKK606dP595772XNmjX8+c9/ZuLEiQndr5I7NE94y6ZDc6QNGod272P+c68TCIU+UXZLxwIpQfcZHNixm4aa2sYSUy1AkdsRU8ZT2rcX8Vi8vXV0OQdnKjjClhUfOwtSPQu413zPQf0o7dc7UTClIyEE1LqFMTL9ckFp9+PdnJyAKpW9Zu5Caiur3YfyltdVx2zi+TMIhPJOnMI2EjRdo96tEpiSxLvXa7fSYnRfjpadlk5Z+fLaKuqi4eMha2gLlMWZDVh+v3+WaZqrpJQDbdv+lJSyVnMGxUwWGWqde4AheHpeD52Irn7hKT1RX+AfScuyHpU1TcOyLC666CKeeuop8vPzsSwrkZR21A+6JR0NwwkgK3Lbr18/7rjjDpYtW8ZLL73EVVddRVFREaZpHpXwpraTiTxI20naWTD7DQ7u2OOY6nd+R5j7kDa632DX+s0AWSXoDB03CqFrJ3xBTGlLfAE/5XvK2LNxG+BUfmoJ6vocNGYEwfy8jrfJk4AQ1GYojKFQ2L378WxNzkDTdcI19SxxS2WnSzi0bZtQUQHjZ55GLBo7gWYvnMIpVeVHgMxR0cLS4qSy07kFCRiaTnWknsqG2s5uTnOoiK9lGEYR8JyU8jP9+/f/sm3btbqup9cZNSbOlQD/pFE/fKJciB66ELo64VVPj38A+tHKp0dN07Btm1GjRvHkk08SCDhZ6HorCj4o+YJKdNN1ncsvv5wXXniB1atX88ADD3D99dczZMgQioqKEmUkk8lvKigj8nBtPR88+xo+v99LYMsCQtOIh2Ps3ZQ5Icu2bHSfwYDRwxwngK5+R2SAlBKf38fmpasxYzF3mjfVug4RHjp+TOdMg7tDac3hI2lXs22nGEaoWwHSzs1M/PaCbdsEQnlsXraaA9t3py1Lq8jtuBnT6TGgD2YsdsLMXgjh+JZXHTzsLklRXc6dFSvt19v1kM7d/bdsm/LaKiAnn7uVzMEE/mvv3r1fPvvss/9iWVY0i/FSzcKeB/wXXpTXQyehK190yhD7YpzCEq2SMqgIayAQ4JFHHqF79+5YbrJYW5BsZabkDEOGDOHmm2/m8ccfZ8uWLaxZs4Z58+bxxhtv8OKLL9KvXz8gfQTSdknvynfms3/7bvxBvxflTQMpJbquU19dy77N2xPLWoQ79pX06UnvwQOJx+I5l8Xd3hC6RjQcYc28RRlWdDxd8wryOWnCGMfKqiPJgpuUFKkPU1dVoxa10EynTXmF+YSKChwHghwmNccKTTjJmHOfesVZkGZXpZusNuXTM7FN+4QqxiE0jVg0SvleR/fa4i3uSpl0n0HvoQNyyn0lVdDjcH12evVOgqAx8eySefPmXXHOOefMtSxL6rqeaVBS8ohfAJPwSK+HTkBXdWlQ0yiFwP8jiymSZFKpNLnxeJyf/exnnHrqqZimmZAptBWq81K/paQMSgYxePBgBg8e3KRNl19+eaJNLRIzKR2S0hBm8SvvcOUdXyEa"
        "jiZcBHKlA88ZuAPcod17qamodBeljvBKJEPGjSKQn0e0vuGESOhJBdu2CYaC7Fi7ib1KzpBCF65pAtuSDB0/hu69erjJTh13rUkkum5QX11DQ7U7zZui9DES8rsVNhbG6LBWdixsyyZUmM+yt+aye/3mtIUm1OzVSRNOZvApo4hGTqAy2VIidJ1YOEL5nv2JZc2hBomikmKKe/XE6gQNenNIKUGC4fc52mskZiwO7n1YE6nv1PZlCcO2bUvTtFPee++9HpMmTTqwatWqfpqmSdu20zlC2EAQR887ExKGFF4Ex0OHoKuO7koX9DVgDBmeFlXnr16WZRGPx7nyyiv5n//5HyzLSiljSH4ST/6+SlxTL+VtmZyYpus6Pp8vpU/vZZddxo9//OOEs0MqqEFt1bsfUn24glBhPrrPQDecNtsusfagpuz97N6wFUgfPVdj5OBxo9D1pok/Ukpsy04cW9uyToikQU3TWfbG3MzXnHssRp82CcNvdLglmUpKCtfWJzS8LbVBRS1DRYWEigqwTfvEjPBKiWZohOvref/Jl7JY3TlWZ1716cTM04kCKSWGz6B8T5mTlErLD7Wqz+170mAnYa+T719p2/j8fvx5AarKD7Nvy3bKd+9DCIEvlIe0bcLxGJD7l7CmabrtHNC+L7/8cknv3r3jUkqRYYZUzcqeCdxEZmszDx7aFV0xwqueFEuB79H4lNjyym7ixpQpUwgGg5SVlTFkyBCuvfZabr75ZnRdb5JQ1jyxLJkIZ+uuEIvFjnqFw2EikchRfwcOHEhhYSG1tamTFaR0yg7XVlbz0YtvM/2Sc7HiJv5gkGBBKBHZioYjCd1vV0Q6PWL229CIx+LscQlvqitDee/6An4GjhruyBncilRIiS8QwPD7EJqr89V1zHicaDjSJY+vlI3Jams/WAyQMmNdHZv8okJGnzrRmVHo8H12pCm1R6qIR2Jpo5kA3Xv1QNN1JPKEmrpXsG2bUGEh8556hQM79rjnKMX50zSkbTPo5BGMPnUS0Yb0hVe6Gpwqij72btqGbdmJgMZRcC+DgaNPwh/0E4tEEK3Iz2hPSNdZ49Cuvbz98HNsXrqaSL3jrT745JFcfMv19BszjJhbNa4rXMOapmmmacqBAwcG//Wvf8lLL7004UaUph9XEd2fAs8BNXhRXg8dhK5IeFUVl28CA0ij3VUd4ZgxY3j77bcpKSlpEs1VWttMg0FDQwN1dXXU1dVx+PBhDh8+THl5eZO/VVVV1NTUUF9fT11dXeJvXV0dDQ0NaYlcNoORGtzeeeQ53n30eQKhIKHCQop6ltBrYF9GnzaFMadPBqRbUSj3O8zmiFsmhqa3mfg6nrE6tZVV7Fy3yVmWkiQ5fWyP/n3oObCvM60oJZph4A/42b9jN5uXrKZs607isRgFxd0Ze8YURkwZTzwWa/tOdhKkLfEH/Cx+7V3CdfVuoYIU7gyahrQsRp82iZI+vYnU13e41EOVWT2874BqFenGxJK+vTqgVZ0DNWtRse8g7z/5Yhb3tnOczrjqQvx5ARpq608owis0DStusnvDNndBy8dDXd+DxozATFOa+nhDug/QB3bs4cH/+jW1R6qatHHH2o3860e/5ct/+BH9pp+KaVkYnUTMWwvDMIRpmlxyySXitttu429/+1vC5jMF1Pg9BPgK8H80Rn49eDiu6GqEV0V3i4FvkCa6qzo3TdO4//77KSkpAWgSsVX/r6uro6qqirKyMrZv386+ffvYs2cPBw8e5MCBA5SXl3Pw4EEqKiraIQIpmvy/1ZXZ3O9E6sNE6sMcOXCInWs3suT19xl31nSu/cE3MQJ+ZBdL3hFC0L97Tw7UVGCaFn6j9Z6ZKvKzb/N2x7s1DXEWGkgLhowfgz8vSH11LcH8PMK19bx632Msf3Mu0XCkyXc+mv0m53zhSi762ueIRWNd5qFCSokR8HF43wGWvP5+IiEtFdT1OOGcM1BCgg7fU+lE6w/t3gdkDgH1GtyvQ5rVGZDSxhcM8P6TL1J7pCp1RJPG6G6/4YMZN2NaxrLaXRFOMmMDuzdsAVouuqH61oLu3eg7fHCnJ6QKIXjtH49Te6QK3TCwLDNxQeuGQaS+gbf+/QynnXp6l4juJkNdj3fffTdvvfUWW7ZsSXuN0ng73w48CNTjRXk9dAC6GuFVT4dXA/3JEN21LIu7776bs846i1gsRmVlJWVlZWzatImtW7eybds29uzZw969e9m9ezfhcDhjAxTJaamIhCJXqf6m+3/WUN8RbrcoBEI4U2Afz19CQfdufPa/v06krr7LEDJwptevOOVM6mJhXvt4IYdqq8jz+VtHet2qYNtXbwDcjjhFpEFtdsjJI5BSkpefx5ED5Tz+i/9LVGcTmpb0zCCQ0ub9J19k6PgxjDl9sjMl2QXIhLRt/MEA8597jfqqmqwI04CRJzF88jii4Uin7KMQIG0rQXhTwZYShKC0Xx9HztNB7esoOImGeWxftY6lr7/nSLRS+CY7cC7sWZ+/gmBBiPAJFt2Vto0vL4+9m7dzZP8hZ1kK/a6UkkFjhpPfrZB4Jz2gKinD3k3b2LriYyc6bTatrmtbzozcno3bqNx/CH2K1qWkaWqsLSws5C9/+QsXX3xxxq/gBK6G4lRGfQyv7LCHDkBXIrwquusDbkla1iIsy6J79+5Eo1FuvPFGNm3axPbt2ykvL0/9A0kkVnU2zbW9LZHYToF0U3jcv0IIhCZY+e6HzLzuUor79MKM544NTzpIJLrQ8RkGQwr6cOOpF/Lqxx+x/sAuQr5AdqRXOob8kfoGtq1c5yxKU1BBDUT9RgzB5/exdcUmnvjln6k8eBhN17FtJ0kt+Zc1XUMKWP/RMk4+Y0qXIFfSlvjz8ti7aTuLX3k3C8Lk4LTLziMQyqOhtq7DCZOUjp1WXVVNwmc1HakpKO5GUWkxVvwEq5InHRsyM2by6j8ed6blNZGyoKt6WBl88khOOfvUEzK6KyUYPoOtKz5G2rZzr6YpiDJs4sn4An5i4c7R7zpVH32smbsIyzRdZ4bm6zjkNl4fxmxwZpVa6wff2VAyhosuuojrrruOp59+OpO0Qbqvr+EQXk/S4OG4oysRXvVUeCowhSwyPGtqavjFL35x9IaSyvymIrRdDU50SxBtCHN43wF6DupPPJb70+4CJ0oX8Bn4dSebvCCQx7WTz+HFNR+yeu9W8rIgvVJK/IEAezdta/TfTTVt7/pz9h48gIGjR7D6/Y/4z2/uIeLakqUcQN3NVR8+4k6j5vaxVdA0wZv/fMq1FtMgVbKTez8U9+7B2LNOJRaOoHXGNLBLEmrKDlB5wHlAbfG+dM9jUY9iikqLO1WneTxg2xb53Qp578kX2bVuc6Zp4gRmfeEKfIEAVl09nGCEV2ga0UiEzcvWOAtaenhzE5U1Q2fYpLHEo/FOsRuUrkVi7ZGqRKJouusYDQry8ju4le0H1X/cfffdvPHGG9TW1qbLx1An5HRgIrAKT8vr4Tijq/SGyvBaAtfQWGEtLZT1kmEY6LqeGAyVtVjzsr9dHVJK15HY6UC7xtDvdIhBnz+RsGZLiSY0rhw/g9G9BxOOxzK6BCj97vqFKzJabqlNTZh1OmvmLuSxu/6UILsZrcekpFtpsetpmtvXjG1Z5BXms/ztD9iwcLkTXUq3f+51c+ZnLqKoR3fMuNkpnF4i0Q2N8t1lzlR0Cv9YtbS0by/8obwT4h5WcCLzQcq27eKdR55P3BepoM7tmNMnM/bMKYnr+USCk7xncHjv/sRDbUvPbqqf7zN4IL0G9cPspAd/KSX+YIDtKz+mouxgSvKnehIjL0jfHk7yZVd8cFMPZMOGDeO2227LlBAucMitD7gyaZkHD8cNud4jChpLGkaBHsBlZLAiS4Zt2wlieyINiM2hItZ5+SF69O/jkpUu0H8IFeH1o2vOFJ7mDgya0Lhywgz6FpUQy2Aar+ka4bp6NixcAaSnorblaFoPlx3gsbv+lKjAlJEMCuc4"
        "TzzvTOwcL/EsbYkvGKCi7CCv3/9ERucLZclW0q83Uy+cRbShc7S7QKLK2p6NjrVcpiSevicNTVFqqwtDAFLy8j0PE6lvaIwCtrSqe27zCvK55BtfPOEOhYKUjo3ghoUriEdVWezU/rsnTR5LXn5+p/rvSilZ+e6CJu06Cu7ybj1LOGno0CbLuhrUtXj77bfTo0cPLCutY5DqYC6jkQB78HDckMuEVxFdC+gL/AiYjyN0hxxue3MtsKZpR710XU/5asUPNf6GriOlZMzpU12brdyXMyhIKQkagSZRXBXRyvMFuHz8DHyGkZKQStvGHwiwb/MOyrbuSCxLB9uyWPzqO4lklkwPQ7pr43Xqpedx0qRxxBrCuR1BEw6Jnf3nfzkV5zLuo0Oozr7mUgpLumN1UnQXnHabMZN9m3ekXU/tzYCRQ9Ou19VgWxZ5BQV8+MIbbF62JmNkXl2/593wGfqeNIRYONpl7v3WQBM60YYIa+e58oAU6ymCO3LaBKRtd8plLG3HiuzQrn1sXu7IL1L6Xrt/TxozisG9+6Mm6roiVJS3T58+fOELXwBIN6ap3RwHnIJzSruOcNlDl0OujthKy1MC/ApH3/NbYDStiO62J5qT12SCahhG4qVp2lGa4OQqb82rtbX0yhpJv2GZJn2GDOSimz+PGe06yTtOJqIkz+cHmmrcNJf09utWyrkjJxO1Wk7Ck0g0Q2flux8mEp4ywYybjsY3ExEUTjKcZVqMnDaBS7/1JeKRaKdaHGWCbVmEigp4//EXs5IyKClH/5FDmXbhTCINDYnS1R0NVUWr6tBh9m/fnVjWHCoiHyosoMeAPl0mQTMTpCUJhELs3bSVt//9TEYpg3AJxuCxIznzqguJuB7LJxqkbeML+tmzeTt7Exr9lu3IkJLSfr0ZNGZ4o269gyGR+AM+1i9YRlQlD2Z4qB5/+lT86E4wvwtfyuoB7PrrrwdIN6YJHGcGP/Apui7P99BFkGtJa4LGqY1rgT8Ag93PLPezDum9mie2qZs2W1lEMBjE7/cTCAQSf1WZYZ/PRygUwufzOTovvz/xFByJRPjggw+y+p1QUQHB/BCarjNq2gTO+cIVFJYUd4lktUY0aniBoyplKdI7fchoth8uY+PB3U3syhztrp+KsoOsmbcIoHVTmOnIhCtjsC2LsWdO5XM/vg3DZ+R0cpRDdgtZ/f5C5jz8bIIQZYMLv/p5Avl5RGobEJ1FmlxruUO7y6irrE4dfXeJTUm/Xl3KkSQtpETogng0yvN/+mdjVb80UgZwXAsu+cYXMfw+ork+83AM0H0Ga+YuTOvOoIqmjJo+gYLu3Wioqev4BwBXuxqua2D1+46cIWX5GzfBrqC4GyXDB/Lsyvf5zCkzyPMFOiey0w5Q1+WkSZMYP348a9asSZdwqXbxfOCPpPQg8eDh2JGLhNfGIbr/7S6zcEhuu011JNuOpSIDzZdrmkZRUVHi1bNnT3r37k2vXr3o3bs3PXv2pLi4mMLCQgoLCykoKCAvL49QKJT46/f7M+oiw+EwQ4cO5eDB1EkOqvMYOXU8n/3vbyBtm7zCAsxYrIuRXcAluEEjkHIN4XoOf/rk6eytKidqxtCFY+8jbZtAMMgH73xIfVVNxhK02SKRwCZh5nWXcdHNn0dKiRkzUyZRdTZsyyZYEGL3hi088/v7nIe0DNeCupYmnT+DMadPIlzXedFdcJPVDYOtKz8GSHkPqESfficNwRcMYNbGc/a8ZAvbtgkVFfLiX//N7vWZXRkUWTrtMxdy0qSxDrk7Acmu43bgo+rgYVa/95GzLEO/PfrUyY6coTOSLqUkkBdk64qPKdu6K21+gLq+R0w5hT4D+7Nh7w5mW5LPTT3XefDvUn25AyEEpmni8/m45JJLWLt2bbprWV2wk3AKSlXiFaHwcJyQS4RXE0LYUspf4ZBdFdE9JqLbXE+rJAXpIqg+n4/zzz+fadOm0bdvXwYNGkSPHj0oLS2lR48edOvWrc3tSeXjm2w03qtXr7SEVy2pPHgYfzBAPBIl7Baa6Fpkl4SHsIrwthTTUNZlJaFCPjV6Ci+u/hDdp4Nto/sMqg8fYdErcxo3eIxQZNcfDHLVd7/K9IvPJVLvlIfOVVLllC/1c3jvAR6/68+EXf/cjIRJ2hT1KOaiW76AGe3kKKmbrBZrCLN99Xq1KC0GnTwCcQJkaVmuDGX1+x/x4XOvZz53mtOXlfbrzfk3XE2sIZLRyaSrQhVOWfL6e9QeqUr5UKv6y5K+vRg8dqQjZ+gE6ZGUThLtync/TDjGpBpvVL8/6fwZ2LZNvj/AhoO7mbNhGZ8+eXqXKkCRDNXm888/n9/+9reZZA0S6AlMBebQWGDKg4d2RU4Q3pkzZxq33nqrvPbaaycDP6YxqtvqO11JEYCEfrZ5Z1NUVET//v0ZMmQIy5cv59ChQ4nOUj2dHjlyhJtuuomhKms2CakKUCR3Ts3/qv+3tDz5u8FgkO7du7e4TtLaADTU1GHGHL1uVx7sNCHI8/kyrmPbNhMHjGBbeRlry7bj0wzyQiHmP/MalQfK2yW6q4hGad9efOHn32HIKaNoqK5D03L4YUI6k59SSv7zm3uoKDuA5ibZpYMTeZJc+s0vUdq3Nw01tVnpn48XpJT4gn4O7tjL/m27nGUpdJq27WTs9x85DMuyu1w51mTYtk0gL8ihnXuZ/X//zErOJBBIJBd//XoKSroT7oyp+w6CpulEIxGWvTkXaNz35hCaQFqSk8+YSkFx58gZEmW8yw6yfsGyxLKWkKhqOGoYwyc5VQ0RgpA/wMId6xhY0ouT+wzpkqRXzTRMmDCBkpISjhw5ki452MLhItNxCK8HD8cFnU54pZS6EMKcN28e+fn5X6uvr1dPd226w5tHRUpLSxk4cCBjx45l6tSpjBs3jkGDBjFy5Ej+8pe/8PbbbzeJpqjOZfHixcyaNYtFixbRo0ePRMJaSyWF2xNCCAwjw2lx+wwzHscyTXTD6LqWa9LZ54DhT7uaLZ1Iyf7qCg7UVqILDcPvo3xPGfOff93tTI+tKYpI9RrUj5t+8wN6De5PQ3XnksBsYNk2Bd278fo/Hmf3+i1ZkV21zpRPz2Ty+TOciHAn76fE8VLeumKt67+bItnO1bX2GNCPXoP6EY9GczbynhFSous6Zszk6d/dR21ldcYHt2QZyvhZpxOpPXHJrm3b5OWHWL9wOXs3bk3co6nWFUIw7uzpiXK9HQ1pO9676z5cSl1VTVpvbxXanH7JuU2rGkrQNZ05G5cxqLgX+f68o3IbugKklJSUlDBlyhTmzJmTKEHcAtSOneb+9XS8Ho4LOo3wSik1QAohrOrq6tKioqIf3XDDDTc+/vjjtq7rWqvcCmiczho0aBATJ07kjDPOYPLkyYwaNYpBgwYdtf6ePXv46U9/imVZR+nepJT4fD52797Nfffdxy9/+UtMtyxkRyBr8ioltmWjd/pjS9shcSPbLuFtqUtXhSj2VB7imRXvUxcN4xM6voCfOY88S11ldeaiChmgBse8gny+8PPvOGS3tj7ny3valkVh924sfu1d3n1itlspLkNkVxPYlk3fYYO4/LYbE9ZsnQ0hBGY0zvqPljvvaVnSoJYPPWUUwfxQp5Q/bi/YUhLKy+O5P97PrnWbMj6sqH6uW89SLvn6FzFPtHLKzSBwCO5Hs99y7Lq0lpP4FLHsN3IoA0crd4YOPi6ulCFaH2blnPlpV1XEvVvPUsa5VQ0TM5NIfLpBRV0NC7Z9nJA2dCW+q2ZKDcNgwoQJzJkzJ10foz4YD4SABjwdr4fjgA6nSlJKAehCCNN9/1Xg56+99tqguXPnAq3MsqeRrPh8Pl5++WUmTJjQ/DcTLyWmnz9/PnV1dSnrfatowTvvvMOdd96J7vrcHm9iYJomdXV1iXanhRBdN7KVBA1BwNdyhNchu4Jth8t4ftU8ovEYhtAIFoZY9d4Clr/1QdYlV9NBDUBX3vFlBo46ifqa2i5BdvO7F7Hs"
        "rXk8+7//AGmTqb6ek/UPwfwQ1/3PreQVFhBtCHc6YVQa5IO79rJ7/RYgcz9w0qSxXXdmA0e3W9CtkPnPvcHCl95O6TyQDHWdXvKN6ynu3cMh+zl+nbYV0rYJhPLYvmYDm5auTp/8hcOOTjlrOnn5Ieqrazr8uEhpE8jLY8vytezdvCOrZLWpF86kW8+So2aSbGkT9PlZuXcLkweNpGdB9y4nbVBtHT9+PJD2flY71QfHenQFHuH1cBzQoaPcM888owshpBDCrKioGCulfK2ysvKfX/jCFwZdeuml1t69e4FWRDiTIKWktLSUPn36YNs28Xg8UTZYyRGUb66maQlNUaoORBHkqqoqYh3ofBCJRKisrEy/ktsUw2fgCwS69KAPTsfoa2FwSia7z6x4n6gZRxeak5y17yAv/+0R5zwdY7+oCPPZ11zKtIvOoaGmLrfJrntt5ncvYuGLb/Ofu+9xCkWQ2sIKSFSKk1Jy1Xe+ysDRw4nWN3Q62QXXqN/vY/38pcSiqb1TFeELFRXQb/gQrC5qR2ZbFqHCfDYuXsXL9z6cdqpeQV2nE887k8nnn+U6auTwdXqMkABCsOCFN13HhdSVypSm++QZ0xJymI6GSlZbMWd+2vaqezBUVMC0i84hHmm5vZoQNMSiLNu9ydn+cW19+0Pt/8SJE4GMhFf58Z7sLuv8TsnDCYcOuaiklEJKaVx77bXWRx99lCel/FFJSclHs2fPvvikk06y/vOf/9i6ruttHbjU90pKSigsLETTNAzDSJDbltCvX78myWctbVPTNHr16kUwGDzupFJtv6amhszE39nfYH5+oiRsV4ZA4FO6DPcSUGR3a/lenlnxPqZtoQsN4erAnv7t36gur3CnMtu+/8qnduCY4Vz4tc8Rru/8aGc6OFObgmB+iDkPP8uz//sPpGmnSwhJQNN0bNvmwq99jqkXzXKTenKAMLnFQhpq61k9d6Fa2OKq6l7vP2IYpf16E+9sZ4k2wLZt/HlByvcc4Onf3dtY1S6DJ7Ry1GiUMnRcmzsa0pYEQ3nsWruRjz9cktj/lqC5s1wjp02g79CBnWLNqJLVyvceSEhy0o0tUkomnTeDngP7pWyvlBK/4WPD/l3URcKJkutdBWqfBg0aRN++fZssSwM1Pdt1dtRDl8FxH9mllFpSVPeM008/fR7w229/+9tFn/nMZ6zKykpdaXbbejOrm6iwsJBQKJR2XRW5O+uss+jVqxe2bbcYzVPRlCuuuAIhRKaa4O2G7du3E41GgTQdpvu3W8+SNpGzXEp+kFKiaQJDS57OU2R3H8+unIflkl0kGAEfL/zxQbav3oCmt4duVxIIBbn6u1/D8PucKeXcOTxNIG3H0N4I+HnxL//ijQf/kyiOkZHs6hq2ZXH6FRdw/peuJlybOxW5HN/SANtXr2f/jj0J94h0OPmMyU6yZhcbF6UtMQyDaEOYJ3/1Z6rLj7j68/T74WhX4fJbb6S4T88uSfRbA3Ve5z3zKrZpJaQ4La7rHrvJF5ztPAB3AimUtiQQDLBu/hIaampTtkOR3UBekFMvOz9twRQJGJpGdaSeDQd3JZZ1NRQWFjJq1CiAdOOVOghT3L9e4pqHdsdxHfGklIbrrRuKRqO/KCkpeXfXrl3TJk+ebN5zzz1SRXVbm6CWCors2hmmk2zbpmfPnvz2t79NVFFTEWH1isfjTJ8+nVtuucUxPj/OkTA13bNsmWNlk/b33F0r7t3TIS2t6OCFEJh22x8u2hsSJyNZ2aolZAzl+3h25Vws20ZzvTT9oSAv/uUhlr01LysngkxQ2fDnffEzDBo7Mie0rKkgXc9hIQRP3/03Pnz+DYcokS3ZtZl20Syu/PaXiTZEOqbRrYCUkiWvvutUG0t370rHk/WkiWO7XHU1KSVoAs3Qeeb397Fn4zbn3GSSMrjn77TLP8Wk82ecsOWDFaRtE3S1u+sWLE0b3VUzXD0H9WPE5FOIhSOJ/qLjGuyco0h9AyuySFaTUjLmjCn0Gz7Y9QpOfQ07PuWw5dBeN8G3ndt+HKES1zRNSxDeLBLXRgAF0GULzXnIYRyXnkFKqUkphRDCjMVi00zT/MDv9//8nXfeCU6bNs1euXKlYRiGyDaqm+2g5nO9XDNP7TqDzFe+8hX+9re/UVxcjGmaWJaVeF1yySW89NJL5Ofnd0iygNr+4sWLm7xPsTYAvYcMQNP0rPmuJgRRM06Pgm4UBPNSDiIdBZWTbGh6YiDQhcbW8n08o8guzqCmGzrP/v7vfDT7Tef8HTPZdbYx9JTRnHXNpW7EMwem91uAU2nKwIqbPHrnn1gxZ34jUcrmWrccC6trfvBNrLiZU8kv0pb484Ls27KDjYtXgkij9XOjfANGn0TvoQNzxl0iG6g+KS8/xMt/e4SP5y/JzitZOWqcNJhLvv7FxnLDJziklLz/xIvYlp02uquOxaTzZlDQvQjLNDucJknpSFS2rVpP2dadaZPVpFt2+NRLz8e2M6WYusmcmsG+6sNUh+sdD+IcCVa0Bief7Ehz07RdHYpewMhmyzx4aBe0u0uD66trAcRise/quv4rTdPy77//fvPb3/62HovFNF3XMU0z47aUjjbbCHBrBgJV/ebWW2/lkksu4bXXXmPDhg1069aNWbNm8alPfUrtz3GP+qkIck1NDfPmzQNIvc9CYFsWmq7Tc2BfLNPM6qlf0zQaohGG9ejHlRNm8NyKuVQ31KMbnd+BCuFEdoUQbD+8PxHZFVKi+XwIAU//9l5WvrugXSK7uFIGw+fj0m/ekCCTOel4IaXzUGNLHrvr/9i0ZGVW2fzQ+GA3fNI4rvnvb2DG4onjnCuQSHTDYOFLczDjTjQoZTQPlYl/KobflzE6litQDxj+YICX7nmIBS+8kXVhECTkFYS47kffIhAKEg1HcnYWoj1g2zZ5BfmsW7CUjYtXJjT2LUI4DwOBUB4TzjkjbbLj8YRyyF3y2nuJ8aJFOYNrnTZ0wskMGz/asSLLor2aplEXDbO/+jDd8wpStIHEw2/z0Ghn3iPqt0855RQgY+KahZO4NgpYiROQ86QNHtoN7Up4XQmDKaXsa1nWfbquXwHw4x//2P7tb39rAFkTWDVYW5ZFIBBI6FrToS3evZZlMWTIEG699daW9qdDOgtVfnLevHns378/bQKSGvSLSovpPURFudJ3morsDu/Zn+umnEvA8DlT4TmiCJMSQr4Ae6vKeXbl+5iWhQYEQnnU19Ty3B/+wccfLs2a6GWCisCccfUFDD1lNPWdXGEsHSTg8/t45vf3smnJypQ2es2hiELPgf343E9vR9N14rFYTpElKR2T/n1bdrDynfludDd1oo9t2wTzQ4yaNqHzyyC3AkIIfAE/s//8Lz568a2sZyhUdPfy27/MoNEjqK/peKutjoamacSjUd597IUs1hXYlmT8rNPoPWQA4br6Dr++nWvY3zhDQWpSp/ruUy8513lgy3KGQq2xt+owY/oMSRDaxBgh3LwMd1stbdFxs3GuxY68a9T+jRo1ivz8fOrr69ONb2rhOPf/uTFAeThh0C69g+vCoLlk92LLshbrun6FbdvmLbfcIn/7299quq5nZb0Djn7Vtm2CwSC/+tWv+N73vuc0NkNnpkhxazo99VvJcgYltejIAVUIwVNPPYUQIq1+V7Vp4OiTyO9W6AycaZqpCUFDNMKYPoP5/NTzHLIrJX7DyInexJYSv26w68hBnl0xl3A0ik83CHUrZO/m7Tzw/V85ZFfT2pXsFvfpxTnXX+VGzHKTRKho16KX32bZm/OyJ7vuPoaKCrj+59+hW2kx8WhukV1wE7h8Pj58/nVikairvUzvzjBiyjh6qupqXYDwOoQoyCt/f9Qhu1lodqFRtzvzusuYdvE51Nfm7kNZe8G2nAealXPms3v9lrRVysB5ONJ0nVMvOQ9pWZ2SjCttpzrg8rc/SHuPJReaGD7llFbNTiiCe7C2skmQQtlqCgThWJTyuir2Vh1i15GD7Ko4"
        "QFl1BZUNtc5smXDKzwucPrej+n61j7169UpEedPstzp4FwI+PB2vh3bGMUd43UISuMlpP7Bt+7e6rmvhcNj64he/aLzwwgsJUpnN1LlhGJimybBhw3jooYc4++yzueeee3B/I1UbAKitrU1Ud2kNYe1MIqCiu7t27eLVV19NJNFlwshpE9BUMYwUfYImnI7wlP4n8ZkJZ6G708Wa0DA0vdOlDE7CmkZDLMJzK+bSEI/SrXs3GurqefvhZ3n/8dmJKdxjLSyhoKIL53/paopKi11rrtwiguDaHPl8VB4oZ84jzyW8RjNCOJEeTdO49offYuDokxxTeyO3yJJTVCDIzo83smLOfDcxKc316H408dwZDhEi90dC27LJK8rno+ffZP6zr2UvRXHXO+XsU7n469cTyRGv5OMNw2dQe6SKdx51o7tprgfVJ4ycNoFBJ48gGu74ympSOmS36mA5q977KLGsJah+Z/DYkXTrUdLqaLSmadRFGjAtK2HhuLeqnG3l+9hTWU5luJaGWISoGXcILmDoBnk+PwWBPHoXljCiV3+GlvYj5A8k2toRuSlqTJ40aRKLFy9O159rOLKGycD/At/B4SiZ9Y8ePGSBYyK8blTXllL6pJT3AF/XNE3W1dXZ11xzjf7mm28mCGwmqBvPNE0uvfRSHnzwQfr06YNlWfj9LVfhao6qqipqamooKSk5lt3qUCjN1/33309NTU3aKF7ytO7wyeMc/8YUnbwQgoZ4lEkDR3LF+DOTPByd9VWn2ZnlbKRtI4GYsPDnBQlosOKd+bz3xIuUbd3ptE/LkuhlATXNP/jkkUz+1FkOkchBsgtuMlcwwNKPllN7pCpr7bKKhF92642Mn3maU3Eqx8gukGCrcx5+FjMWT1saWl0DPQb0daJjnZGJ30qoynHlu/bx+oNPpk1kSoayjxs8biTX/PCbDkHuCuz+GGFbFsH8EO8+9gJHDhzKGN1VxPL0yz/lupWkfvA/XnASLgMsfHEhNYePpG2zU0QDJsw6vdWBBumWVq+PRaiJNLCvqpyluzaxv6aCmOncO7rQ0DTHz9ynN34vEo9RF41QVl3Byr1bKM0v4pR+w5g+eDT5gbwOTcg+66yzuO+++zL154r03gEsAp4CdHeZBw/HhDYT3iSyGwSeAD4DWJWVldpnPvMZbe7cuVmT3eQnvh/+8If85je/Qdd1YrEYPp+PQQMGZmoLAAcPHqSyspKSkpKcykRPBRXd3b9/Pw8++GBGyYeKEoyYcgo9BvQlWt/QYtKDcCO704eM4ZKxpyUye5O1U37Dd9z2KxWklE6VMJxzHgjloek6dZXVrHn/Ixa9PIddbllZlfhxLEUlWmgAQgg+/ZVrMfw+IvXxnI2cCeGQgJ0fb0ybpZ4MFRmc+bnLmPm5y2jIUW2ybVmEigpZMWc+GxevSp+YhKNPlEgmnz+DguJuNHRC2djWQto2/rwAq+cuch+sMkd31ToDRg7jxl/+F/5gALOTqoZ1JKSU+AMBDu7cw4ez33Cv9zRFOFTy1/jRjJo+gUhn2AlKiW7oNFTXsfjVd9M+kDS2dwxjZ0wjFm59ezUhiFkm/1n2LhX1NUgcKVjIH3AT1pycjOZkWhMCv9EY3KiO1PP+lpWs2bed80ZNZly/obhfP26PC2ocnjVrFt26daO6ujqdjlfgkF4J/A1YAuzAS2Dz0A5oE+GVUmrOHxkEngMuAczDhw8bl19+OQsXLsya7KqIZigU4oEHHuD6669HSumQQVf36y9y/HUzWb2Ew2HKyso46aSTOn26Phuodv/ud7/j8OHDGTWaao8mnT/D6TA4upMSQhCJRTlj2FguPPlUh2CKo6MfAb3dDTqObm9SJTuBQPfp+Px+hKYRbQizY81G1n+0jLUfLKai7KCznhuxbq+oroJ6qJpw7hmMnDaBcM5PE7szHvF45uiVaIzsnnbZ+Vz6jRuI1DVkTGbsDCipRvXhI7zx4JNqYcr1ldQhmB9i4rlnYnZSJn5rIYTAMi0q9h3I6sFbkd1+w4dw42/+m/zuRY6uuQvs67FC2hIjGGD+c68TqWvIPJvhXi8zr7sMw+cjHuv4BEZbSkJ5QZa//QEHd+1NG8EXUiKF4KyrL8II+Nzk0da3V0pJZUNtgsBKKdPLgDjavcEQOj6fQU2knudXzeNA7RHOGzUl3SaOGarv7du3L6effjpvvvlmpuR15dhQCvwFuAyv1LCHdkCrWY/S7OJcgE/gkt2KigrjsssuY9GiRVmTXbXegAEDeOKJJzj77LOxLAtN0xBCYGgamw7vZc6O1QTyggkPylQVbADWrl3LWWed1drd6nBYloWu6yxfvpx//OMfGXWqqkPtM2wQI6dOINosSqBOSiQe46zhEzh/9BS3DG3LldVUhLc9HgucyG2j64PAiWr4/H50n9s52zbVh49wYPtutq5cx+YlqynbtrOx/e6+HEvltFRIVDfKz+P8G67usKp5xwKJk5DTd9hg1sxdhNBFi/ENoWlg205hgsvO56rv3kwsEk0Y1ucapC3xFwR56Z6HOLL/UNbX/ZRPz6T30AE01HZ8Jn5bIN0IYFFpd1dn3zISD3iWxcip4/ncj28jv1sRsXA0Z+U27Qlp2wTz89ixZgPL3pybKAaTCipaOmziyYw+bXLnRHdxrst4PM7Cl+ck3qeyIrNtmwGjhjH6tMlEj7F0+bHmXqgosKHrgM68zauIxmNcPPa0hIvD8YBq8+23385bb72VzVd0HO3upcBXgH/jRXk9HCPaEubTXTeGf+LIGMyKigrj0ksvbRPZPf3003n88ccZNmwYpmmi3BwAFmz/mPe3rMRXGKKgpDvRfQcybnfx4sV861vfasNudRzUzR+Px7n11luJxWKJxL5UUB3qGVdcQKgon/rqxmQr4Spxo2ac80ZN5uzhE1yymzouGDCy00U3aXMzUus2DN3Q0Q0D3TCciLKUxCJRyveUceTAIfZu3sGO1esp37OfqkOHm2y3UbqQfT+mro9sO34lFTn1kvPoP2Koo2vN8SlxIQRmLM4ps05j7lMvO1XgdP2oaKiSxXzqps9y/g2fJR6N5qycx5EyFLDq3QUsfvXd7MiulOQV5nPmZy7E7IRIXlshNA0zGueUmacz9+lXsOImmtFYQljdmWr/T7/8U1z6rS+hGwax6CeD7AIIoWHG47x632OOxaKmIbMoiDPz2sswDKNTio/Ytk1efogNi1awc+3GDFI053yfeun5BEJ5NNTWHRPhbS9/BdV35gfyWLRzPfmBELNGTEhUumxvqNnLiy++mG9961vce++92bjOKIL7O2AOsBeP9Ho4BrSK8LpFJUwp5Q+ArwLmkSNHWkV2VTEJ0zS58sorefzxx8nPz09EPIUQWLbNmxuWsGTnBvyaTnGPYkr69EpMD7ZEdNSy5cuXE4lECAaDOTvwq1LGd911F4sXL85446uknZ4D+zLh3DOJ1IebkF2JJGaZfPrkaZw+dFxW+53nCxxFhpM1ts623V/QBIbPQNMdYqvWlVJixU3qKqupqThC5cHDlO8po2zrTg7t2kd1+RHCdfVN98W10pE40Z3WSheSE0OycW9Q10uoqJAzr7zQsQPqAhFCIQTxaJReA/vxxTu/w+w//4sj+w8dtV7/EUO56GufZ8wZkwnXhYHcvOZt26lGVb5nPy/+9d9ZV1i0bZtpF51D7yEDHEeNLnDuwGl7LBKl/4ihXH7bTbzyt4cx4419o7rLeg7sxwVfvpZJ551JLBLFzDGv5OMJ27LJ71bIO489z861GzNKGZKdGUafNqlTo7uWZfPh82+kLzQhnGh1cZ9enHLWqU6hiRy7N21pE/IF+GDragYW9+SkHv2O27ipzt/dd9/Ne++9x4YNGzL14SqBrSfwS+DLeNIGD8eArAmvqqAmpbzEtu3faZpm1dXV6VdeeWWryK4q9vD1r3898ZSnolRCCCJmjBdXf8j6/TvJ8wewLQuJoN+IwWxZvibltm3Xa3Djxo2sX7+e"
        "yZMn5yThVRYtL7/8ciI5L5MNmSK1sz5/JQXdixJRAjXdG5c2l407nSmDRmXcZ/VJns+PtGVjeVoVqfX5Ej7A0rWSi0WiVJcfoa6ymtrKaqoOHaai7CBHyg5SebCC+upq6qtqmgzoid/ThNt+EtHftkzJCSESRvzB/BAjpo5n46KV2LFYZg2obXPa5Z+ix6C+XYs0aRqxcJTRp03mtnuHsvKdD9my4mPi0ShFpcWMOWMKY06bQjA/j4aaevchKLeud3C16q77wLP/+w9qj1RlzMJXDyqFJd2ZcfXFxLtIVbVkCE0QDYc588pP03fYQBa+OIeDu/Zi2zbFvXswbsZ0xp01nfzuRUTqGhz5URe5No8VyuFg76btvPf47AQ5TPsd6ch8PnXjZx19uttvdSRs2yYYymPbynVsXrY6rV2guoanXjiToh7dm8zK5RTcdr6zaTmDi3uhH6f8DtUXFxUVcd9993HeeecllqcZE1RE9wbgfhznBs+1wUObkNWV7Sap2VLKAbZt/1PTNNHQ0CCuuuoqMX/+/FaRXdu2ufPOO7nrrrsS5EdomptBWsezK+ex+8ghQv4gtrQT0cYBI0/K2E4lhH/ttdcShDeXoCK7a9as4aabbgIyT8ur6O6QU0Yz+VNnEa6vT5Bd5bd49YSzGddvaFYEX3XOef4A/rwAobwQtmVhWxa1ldUc2X+ImsOVHDlwiMN791NRdpCaw5WEa+toqKvHaoHUKmialvA5SxBcWx7zNJyKAkjL8bG8+vu38MEzrxGPRt1StOm9LwuLu3P65Z/qsqQpUt9AXmE+sz5/OTM/dzm2ZaH7DKQtiYUjOW2vBoCUBPJCPPf/7mfbynVZR+Zt2+bMz1xEaf8+XcKZoSUI4Zy/IeNGM2zCWMK1dUhbEioqQAhBNBLtlAphnQ7hHJvXH3wi4TOc7ppQn0+/6ByGjh/TicdMAIIPX3gDaasHuZaju7Yrx5n8qbOIRVJbSHY2pJQEDIN9VYdZU7adyQNHHldpg2mazJw5k+9///v84Q9/yBT0ETiEVwfuBj6FV4HNQxuR7aOcAKRt2/dpmtYnHo9b1157rf7OO+9kRXbVlI9t29xzzz3cdtttiYisk1QF+2sqeG7lXCrqagn5AtiujsvJdjbpe9Jggvl5ROrDGWUNzz77LD/84Q/x+Xw5E+VVkd2dO3dy1VVXUVlZmXVBBV3XufiWL2D4DKx4HE3XiNsWPs3g6olnM6LXgKw6KCUbASgI5rF/6y4ObtvNno3b2L99lxvFrcIy08krNCeo4h7+BLGlnZ0VBGjCOT62bVNY0p2zrrmEmdddTvmefax4ex5A+ixlNwJ++pUXUNKvFw25GmHJAE3TsEyLhlpHHiKEQDZE3GMkcpos2ZZFfvci3n3sBRa9PCd7sislpX17c9pl5xNraNl+r6vAidRHnEIrriQoXN/gfJbj5+94QFUQXDNvERsXrcxsS+f290Wl3fnUjVd3im4XnHYH8vLYsXYD6z9a7lynKSQYQhNIy2b82afRa1C/nE+2lIAuBKv2bmXSgBFNxhJVllhBNPueWpbtOVEE98477+SNN95g7dq1mfoFHYf0noOTxPYSXpTXQxuQkfAmSRm+rGnapbZtmzfeeKPx2muvZU12bdtG13X+9a9/ceONNyaS09R0VH3UqbRVUV9L0O9vcuE7WsYYPQb0oe+wwexYu9H5XgtER0kjPv74Y959910uuuiiJiSvs6DI7p49e7j00kvZvn17VlIGpWmbcd3FnDRxLA01tRiGj5gVJ+QLcM3kWQwu6ZMV2ZVSous6u3bt4qmnnuLpZ55m7ccfY8biR/9u80gtMvF/VSzieCF5JsCWNr6An6kXzmLmdZdT2r83mqbxwTOvJc512ulEt5TnqZedR6whkrMRlmygjkvivZ7b+6I8lPO7F7H4lXd5/f4nnNmKLLW70rY574bPUFjS/ZgTfXIBahYrWX/+SYXQHH363P+8lN36bn/w6a9+juI+PZPkOx0NgRDwwTOvYlsWmq4hW4jugnOedcPg/7N33vFxXOXX/96Z2a4uWbJkuffee09xenOI00gD0iAkvBDgR02BECCEFggEQhIgjTgF0qsTJ3GKW5p7r3KTZdWtM3PfP2ZmtbKl1cqWbdme8/msZe3Ozoxm7tx77nPPc56xZ52EoRsdUGjUFFJKPKrGjtq9VNRWUppTZI0Z9mpiiy4jzewH0pNf57NgMMif//xnTjrppGQALM2Kp/PB/wEv4ZJdFweBtITX8dvds2dPWSKRuNvj8ZjXXXed+sQTT7SJ7Pr9fh599FEuvPDCJPkDkkTtzdWL2dNQQ8jrx2iGwEgp8Xi9DBg/ko1frEpbHczpHH/3u99x+umnZ3INDiucv3fNmjWcf/75rFy5MjPdrq1X7dKvJ6dedRGxcARV04jpcfKC2cwZOYPS3MJWyW5qB/LrX/+aX/3qV1RVVSXfU1SlqQSB9vfAzQSptmROlarhMycxefbpdB3Qh0QsRiIao2bPXj5/50Nr21aiu0jJ5Nmnk19c1HH1c8chpGndP82rseS1d3n6nr9alkeStHpraLRx6j1yCKNOm3b8Lfd3gNWmowlpmvhCAdYvXc7WVetbIznJSf+QqeMYd8ZMInVHR77jaHc3fLqC5QsWp43uNibXDaPboL72ZLvjt2FFKIQTUbZU7aZLbidrhQzYF66jJlJPfTRCRI8T0+NWEqrmwa95yfYHyQuEyA/moKb8nelWV50xcOrUqdx00038/ve/b21cdKK8E7CivP/FjfK6aCNai/AKIYQRi8V+6vF4Sr797W8b//jHP5S2kN1AIMDcuXM566yzmpBdaRO11bu38um2dQQ8vmbJLlizSD2hM2DCSN781zMk4nFaYr2Oj++bb77Jq6++etSivI6EQ9M0FixYwMUXX8z27dszIrsWWQN/MMBFt16PN2BVXIrpCUpyCpgzaiaFoRxMabZaYtXJIr7lllv44x//CFiWcKZNLDMpV3u44CS0maaZjHzlFBUwfMYERp82g/L+vTB0nUh9A9I0ycrLZf5TL7aq+XMihDmF+YyeNc3ybz6Go7vHEqQp8Qb9bF+7kdcf+g+rF36KabQavbEgACyyfPaNX7ba9lFITHJx+CAlKKrGlpXrkHZxIdlCf2gVU7ES/M6/+RoM3TxqOZmWs4xk/lMvWOfdgjMDNBK9iefOskq6H4WyxwcDiTUmb9q7k6JQLuv2bGdr9W5qo2HC8SgJw45U2zJEadtUaopK0Osn2xegW0EJfYvL6VlYiqbYyc8tEF+nD//pT3/Kf//7XzZv3tyatMG54N8FXsS1J3PRRrRIeO3SwUZVVdVQr9d79c9//nPzd7/7neKIztMhlew+/fTTnHnmmU3ILgBCkDAM3lnzaavaH6EoJGIxSnt3p9eIgaxe+Jml72zBr9GJ8n7ve99jxowZ+Hy+I6rldQi2qqr8/e9/5+abbyYajWZIdkGxo7vnfvNqug7sR7SunpiZoFt+Zy4aNZ0cf8iO7DZPdqUdsjVME1VVWLhoIX/84x9RVcu0PBOf5MMCIZLRaIvkNia09Rw6gOEzJzJ4ylgKSkvQEwmi9Q2ApXEUqkp9dW1Su5suEc4hV+POOpn8kiIajiFnhmMZ0pR4/D4q1m7kwe/eRUN1LQ5DySSB1CE4J335AroN7NthSyO7OAQIK8rbUFNr98fpE041j4eLvnsjecVFROqPTnTXcpTws+HTFaz8cGla311n7Os2sA99Rw8lFokeM32PlBKv6mHj3h2s3rXF1vUqKIqCR9XwNuveYNti6gl2J2JU1O5l0eZVdM4pYFhZb0Z07UPA42u2CJJzHfPz8/nVr37FxRdf3NoY7Tg2TAJmAG/iRnldtAGtPon5+fk//tvf/ub7yU9+IlVVFa0tdzsPfDAY5JlnnmmW7Jp25aGlW9ewvWYPXlVrdUB0LGnGnnlS8veW4BDOZcuW8ZOf/KS1MobtBtM0k8eurKzkK1/5Ctdd"
        "dx3RaDTjc3AG/ZmXnsf4s08mWl9P1EjQv7gbl489hRx/KBkdB2u4MB27L3sfAkvvqakqAsEXn36e7EiOpFzBSchxXCWwo97OOXTp25MZl5zLTX/6Odf99qdMvehssvLzCNfWJ03oHZcKX8DP6oWfsmfrjrTVmJxONJiTxdgzZli+u26E8IjAiay/eP+/aaiutRO0MlN8O0vXfUYNYeal59nOEy7ZPe4grT4ur7jI7sMPfDZTC8ucd8s1DJgwwpK2HCVJkpQmqkdl6Rvv2a5CaSRkdnufcM6peAP+oyIPaw/4PF4CHi8eTUO1Jx9msy8zOR55VI2gx4dX87C7bh+vrPiYhz58mc+2r7d1wAeu8jh8Yc6cOcyaNSuZ79MCUmdIN9k/XccGFxmj2QjvU089pQohTCnl4L/85S/nf/3rX5eqqirOMnhLSJZw9fl4+umnOeOMMw6M7GJllsf0BIs3r8KjaBm1WEUIYuEIA8aPpKx3dyrWb07r5ekQz3vvvZeRI0dy+eWXk0gk8Hg8GRytbXCui/OgPv/883zve99j9erVTRwqWoOiqpiGwYhTJnPGdZcRa4gQ1ROM7taPs4dMQrX35WSxC9FIbh3ohkFdLEwkESOeSKB5PBDwHLT/bVvgeOU6x9r/mL5QgK79+9BjaH/6jRlOWe9uBHOy0RMJErE44dp6hCIOGNgcp47Fr7xt/W5HFZo9B0UgDcmoWdMoKrd9d13t7mGHlBKvz0fF+k2s/3R58p5lAmeSl1dcxJzv3Zj0BXUnKscfhCKIR6MMnjyWtx59jvp91aiampRWObkLAOd+4yomnXeatUJzlCY/Vj6Bjz1bKlj2/kI7Qp0mKm1KCruUMHjqWGLhyDHbhvd3Zmh1e+tLye9oqoZH9VDVUMuzn77L2t3bOH3QWLJ8wSbPdmq0/Gc/+xnz5s1rbaxU7MOdBgwFvsCtvuYiQzRLeP/85z8Lj8cjr7/++h/8+9//9gohDNM01dbIrhACVVV5/PHHWyS7TmNfvmMje+qr8Xu8GWVuOwbfgawspl50Fv/55f1pk9egsRjFV77yFbKzszn33HMPKF98KDAMo4mt0LJly7j77rt5/PHHATIuswyNZHfgxFFc8v1voMcTRBNxpvUZzikDRlt/jx0ZB5IR3rpIA9uq91BRs5eKmkpqIg3EjAQJ00ACms9LNNckr7iQ6t17M9NSHiSklE2ylrPycijp2ZWy3j3oMbQ/Zb17kN+5E16/Dz2RQI8n7KVNpVmi6+zT6/exfc1G1n263LoOLXWI9oDjDfiZcPYpVilaV7t7ZGAXl6jctrNNmkUnSc0XDHDZj28mv3Ono1ZBy8Xhh1Myu6C0E3O+dz2P/+yPRMOR5OfSkGQX5HLuN65m1KypRI6yQ4c0JV6/lwXPvUq4pi597oA92R535klk5eceU0Vu2hsWYZY28YUvKjawo6aS2SOm0SWvU5Nka0fqN27cOC655BIeffTRdPI/gSVh8APXAd+kI1bbcdEh0VxDUQHD7/dfFo1GHwNMIYSSCdmVUvLPf/6TK664olmy68CUJv/8+DW2VO3Cq3naRMCspAeFv37rdrasWNtqxSang/J6vfz973/nyiuvBDho4rt/NBdg5cqV3H///Tz00EOEw+Em1lqZwCG7/ceP4Mrbvw2KQjwR57SBY5nQYxCGnSThdJ5xabBp3y5W79lKRd1eqhvqSeg6SkoWvB5LEGsIE20I4w0EeO/pF1nw7KuHlfB26deT3iOGkF9SRJe+Pcgr6UR2QR6+oB9DNzB0HT2eQJqmRXIt88a0+zRNk1B2Ns//+RHeefL5tANO0pz+zJOY84OvE+ng3pfHE6Rp4gsGWf7+Qh758T0ZtTNHxuAL+Pnybf+PQZNHH0XLKRdHEtI08YdCbFy2krf++Sw7N20lkB2i94hBTLrgdIq7dkkW2Tl65yjx+L3s3LCVP9/0YxKxuPV+M+3aKZeelZfDLX+9m+zCPPSEfsxGeNsbqhDEdB2/x8sFw6ce4B2faik6ZswY4vF4uv7DxIrq7sSK8lbSYhq7CxeN2J+ROgLwCdFo9EFACiFEa2TXSWS7//7705JdJ7pbUVPF9upKPBlodw/ch4nH6+eMay/lwe/eZSc+tQwnyhuPx7nqqqtYtGgRd9xxBwUFBQDJWWRzBvDOkryzD1VVm2zzzjvv8M9//pOnn36a+vp6oHG2munfZS3nGgyYMJLLf3ILOhKPKbl8/Cz6FnUBQLOPuTdax9KNq1i06gs2b91KbdU+orX1RKrrqK+qpraqmrp9NYRr64lHYyTicYxEwpokKJknDx0UhKChpo6yPt2ZeP5pxOrDmKaBkdBpqLYTVOyJgGjD8qSqadRW7eOzDKzITGmieTQmnncq5jHgfXlcwZYwdO7VHa/fTyIea1azB40etKZhklOUz6U//Cb9xgyz5SeubvdEgFAUIg0NdB/Uj2vu/j51+6rx+Lxk5eWSiMWPOtlNnqdQeO2h/xCPxjKI7pqMOnUqBWUlrpRqPxhS4tU04nqCuZ+8w8WjT6J3UVmSEzg5LkOGDOHcc89l7ty56aK8ChZP6QycDzyY8p4LFy1C7Pd/AQSAJUB/GmdSzX85hez+/Oc/50c/+lErkV1rRvfGqsW8t/5zu6Ja2wmYaZoEs7N47vf/4P1nXk5GSNMhNRGiV69e3HrrrVx22WXk5uYesG06/eDSpUt5/fXXeeaZZ1i8eHHyfVVVaU3j3BImnjuLOd+/EQkEhYfzhkwiGw8rV69i8+bNbNq8iYWfLuWTZZ+zfds2zGgCIxonkTiwaERHwOxvfZUJ559G7BATj6z7HGLxa/N54uf3pU1WcwajIVPGcdXPbyV2jBeaOBZhmiaBUJDn//QI7859ydKv09jJ7K/pHjhhFOfedDVFXUuJHqUMfBdHF9I0QQhUTUOaplWkQRFHPTJqGgbBnGyWvPYuj//8D60mygL4AgG++Ze7KCrvTCKeOOp/Q0eEIgS6aeJRVb48dhZd8oqS462Td/P+++8zY8aM1sZTAytA9yEwFYuruBFeF2mR+kQ60d2fAnfQ2KBahKNR/cY3vsGf/vSnjPxuDWnyyIevsq16d5vlDA4cb1nDMPjrt+5g+5oNGZfpTdXVdu3alQsuuICZM2cyYsQIunbt2uT8GxoaWLNmDStWrGDhwoXMmzePNWvWEI/Hk9scLNEVQuDxeZl60dmMPX0GOzZsZtembQTqdHZu3sqGTZvYt29fk2M1t49GQ/PGamgp/0nCSSo4nBCK5Zsays3mWw/+mlBuDoZ+8Mt6ln7XzyM/+hUrPliSXAJv9tj2EvpXf/UDBk4YRcT26nVxBCFlUpP77G//ztI33jtgE0VV6Dl0IOPPOYVhMyYggHgs7t6rExzSyU/oCCRRShRNJVIX5r6v/5B9O/ekleg4/dK4s07m4u/fSPh4K5bSzlCEIK4nyA/m8JWJZxDyBawPUgofTZo0iYULF6aL8jo3wwAmAotxk9dctAKR8lMCpcAKIHe/zw+AQxznzJnDf/7zn+Syf0vkxpnF7a6r5sEPXjzkE5emiTfgZ+fGrfzl5tuI1DdkrE91bLJSH6Tc3FwKCgooKSkhNzeXffv2sW3bNiorKw8gnamFGw5FIpDbqYBQTjZ7d+wmlpK4kQpHZ7y/68GRILAHA0dTfemPbmb0adOI1jccVJUhKSWa18PebTv5w/X/Rzwaa1Gl5Ryz26C+3PiH249qMY0THY59oKIobFm5lop1m6ivqkEogtxOhZT370Xnnl3RvF6iDWEgfRlSFy6ONEzDsjWc+5u/8tH/3mg1T0QIgerR+MaffkaXPj2JR4+NympHE6pQCCdiDOzcnYtHzQTshEZ7hfi3v/0t3/nOd1rzrneCcvcCt+J68rpoBY72wNG/XA/k0Up01yG706ZN45FHHkmS2UwGrh21lcSNBD7Ne0hkUSgKsXCEsj49uPTH3+RfP7kX3Y4mtrZfJxLsJIKZpklNTQ01NTVs3Ljx"
        "gO0d0umQ3PYq3FCzp4qaPVaZXydaa1WwsUmtlEfEP7i9IYRgz9bttkXZwaXQStPE4/OyfMGiVvVzDgsed9bJeP1+t2DBUYRVctVAGgY9hvSn94jBlrRE2j7Vuo4eS5CIu1EwFx0Ppmnizwqy6uNP+PiFt1onu/bnw2ZMpOuAPsdfKezDBEOaBDxeVuzcxJKtaxjTrb8lebSv3QUXXMBPfvKTZBJ4C2O6M7TMBm4H6nGT11ykgUKjzUcW8BVoIrs78AuKgq7r9OrVi8cff5xAIJCRZ6bTAnfUVh00CTrgXFSVSF09gyePZc7/3ZisJJbp7No0TXRdT0anHQLsJKelRoKd7do16Us0dbgwDQPDMJLFGQ63b+7hQNI2TTs0wqmoKrGGSGOyWkvHs63I8joVMnjSmGOqstHxCidBMR6JEa6tp6G6loaaWiL1DVamu8C9Ry46JFRVIdYQ5sX7/2UT3Vb6YLsa3JQLz2g1j8RFU0gJPtXD22s+YV+4DiXF3ahnz55MnDgRSNtXOBKGnsDMlPdcuGgWCo2R3NOBrinvHwCHmOXk5DB37ly6dOmCYRgZDV4OEapqqD3Uc24CRVUJ19YxetZ0LvvxzfgCAbtGe9vavePG4FRLOyKkUx6YyHOsw3JGk3Tp0xNpmAc1s7G8L31sWbmOinWbbX/dlrKjrfs88pQp5HTKx0joHUMH6CLprayoalLm4MoXXHRUmIaBPxTk3bkvsmPDFhRVaTFRDezorpQMnT6ebgP7uJPtNkIiURWFuliYd9d9nnzfWcmbPn16ZruxcHm7n6CL4w6pIu+LsRpPi9NUJwr68MMPM2rUqIyS1FK/qxsGDbEoiu1Z2F5wSO+Ikydzzd3fI79zJ0zDtCQC7gB72OFUWHMyrcv69KD3iEHEolEUcRD6XSwd6IoPFlmTlzRuC6Zh4PF6GXnKFKvQhHu/Xbhw0UY4OSHb1mxk/n9eTOvK0PglK89g6pfOcvMGDhKmlAQ0L19UrGfbvt3J1SFoJLytJKM7Hf4soByLv7izDhfNwiG8+cBJWI2nWQbruBHMmTOH2bNnJws3ZALHKTdmxAnHbRLUzkFNh/T2GTmEG35/OwMnjELaEVpFVdyo3yHC0Rg7EbtU2YiUEmlKDF0nuzCfC79zPd6A/6AjvKqq0lBTx/IFS5L7bw6OTrf/uOGU9u5OIhp3rchcuHBxELAm7a8++IRVErgFD2kHTsn4kadModugvm509xAghCBhGHyw0aqkqQpL8z969Gj69u2bdGVqAU7+UT5wtrPLw37SLo5JOK1oOlBAC/pdR1ejaRrf+c53WmuALUI3TeKGbkd4238ZX1FVIvVhcosKuPoX3+PCb19LbqcCa/ZtE183ApgejcRWbXK9LFJrYhqmlZRkz7qFopBdkEd5/96MOWMm1//2J3Qd2It45OAylU3DirRs+mIle7fvTJYLbg7OOYyaNS25vOjChQsXbYFpWIlqn7/9ISs+WJK01msRtrTPHwww/eJzMdyKaocEU0p8mod1u7dRUbMXoSgkjAShUIhzzz0XyFjzfwVWwM4dCFw0C8elYZr90+DA6mtJwjt69GhGjx7ddsJr02jDNDFM47AGWxVVQY8nQMCkC06n37gRvP/0Syx6dT7R+obDd+BjEElnDSGQ0rSq1kmZ9ENM3S6Ym01haQkFpcVkF+aR16mQovJS8jt3IpSXjdfnx58VxNR14uFDsOWx28ay9xYBjZXomjt3KSWdysvoM2oo8UjULVzgwoWLNkPVrOTnNx55OqPtFXs8HH/2yZT27ka4xq2qdqhQhCCsx/lk6xrKcicibCncnDlz+N3vfteaW5GCxTImAJOA93Atylw0Aw2LYoyxf2+WijpyhlNOOQWwNDWZyhmax+GdDTvL2uHaenKK8jnv5q8w9qyTWPr6eyxfsJiaPXuJR6KH9Rw6GpqQ2xQP4f2joqG8HDqVl1JQVkJRWQlF5aV06lpGTlE+vmAAXzCA5vEkqyKZpoFpmEjTJNYQtsoHH8LSnubRqKmsYuXHnwC0bAmkKGAYjDhlMll52TS4g44LFy7aCKei2vwnn2fX5m2tFjByVpJyOxUwdc7ZxCMxV0bVDjClxKtqrNm9jRmxCCGf5f40YsQIhg8fzieffJLu3jhOUypwAxbhdeHiAGhACTDE/r1ZxuDMroYNGwa0rKlsDYoQqIpySNW3MoVzjkZcRwDdB/Wja/8+DJ48hlf/8STrP1lu6XqPw2Vwh9iK/SK3qfdNUVVyi/Ip6dGVovJSSnp0oXPPbhSUFuMPBfEFA6iaiqEbGLpuSRlMg1hDmKi0QvZCkNRGCw6N6EJjlvQX8z+mbu++tJ7K0jDQfF6GTR9PIh53lxRduHDRNkhQNY36fTUsePYVKxiQydek5KTLLyC/cyc3utuO0BSV6kg9a/ZsY2R5XxK6jtfr5ZxzzuGTTz5prY93PjwL6AZswfXkdbEfNGAwjZXVDoAjZ/B4PPTp0wc4CA9NuylqiopH1Yjp+mGJ8TrJU0IReLxePD4vsUiU7es2s/rjT1j10VJ2btrWGN09Tsiu45IAlh5tf1mCUAR5xUWU9e5OSc+ulPbqTude3cjOz8UfCuALBJJFAQzdwDQMIvUN1vVJKSjikNrDRS2FsOQLy95bmDxvaRx4jxRVxTQM+owcTEmPrsSjbpTFhQsXbYNpGoSysln82nz27tjdanTX+bzbwD6MPfMkovVhl+y2I5wEolW7tjKyvG9yFfm8887jrrvuas2twUleywVOBf6R8p4LF4BFeAfS6NbQ4tMbDAbp3LnzQR7GIiM+zUPQ46c2GkZVtHZJMkoluZpHw+P3occT7N68nVUff8KKDxazZcU6jHaqjtYR4BBcKWmUJ6QQw2BOFmV9etC5ZzdKenShrHdPCruUEMgK4Q34rKhtQrcKaiR0ErEarIitSBbDOOIZxxJUr8beHbtYu+QLAMxWktVGzJxkW6FFEG5lNRcuXLQBiqoSi8ZY+vq7rbr4OJN+VdM447rL8Hg9RMMR15mhHSElaKpKRfUeaqNhcvxBTCkZMmQIw4YNa03WkNwNcCEW4T0+Ilou2g0a0M/+f1rC6/f7KSgoAGjz8rGzruBRNUI+P0aNiecQ+MkBJNfnw0joVO3czepFn7Hqo6WsW7rMqupkQ1EViyDa0c9jBqIxsuo4JaQSXEVVyCsuorx/b0p7d6NLn56U9ulOVl4OvqClgzLiOrquk4jHiUejyaitpUgQHYIsmtLE7w+ydvEXROobWizpmSx+UphP37HD7WS1o3/+Lly4OHYgpcTr87JtzUa2rFgLUmKmGReclc5pF55BvzHDCdfVu2S33SHRFJXaaJit+3YzuLQHui1rOP300zOVNQhgLFAK7MCVNbhIgQY4YdtmW5JDMAKBAF6v96AP5JQfLgjlJA/WllbYlOR67EhunOpdlaxbuowVHy5h/SfLiTaEk99pJLnmMWUMfkAEF4m0ddSKqlJYVkLXgX3oMbgfXQf0pqC0hEB2CI/Pi57Q0eMJTMOgoboWx1+yI5Hb5iAQGAmd5QssdwbbivHA7WyZw6BJo8nrVOgOPC5cuGgzpGmieb1s/GKVXVpeQcrmxwgnqti5Z1dOueoiYpGomzNwWCGThFexr/NZZ53F3XffnYmswQSKgOFYhNeVNbhIolXC6yAYDDZqOQ/iYXe+UZpTkDHZdSKaiqLi8XrQfF70WJy9FbtYs/hz1iz6jA2frWxCch2NqSnlMUNyGx0UGjW4TgRX1TSKykvpNqgP5f170bV/HzqVl+LPCqJqGol4HEM3iEdjlmF6SsLasRL5lFKi+Tzs2baDjZ+tAtLJGaz3h0wd11rn58KFCxctQgI7NmwGnIn0gds4Y53m9XDBt75KICtENBx2J9mHCRJQhcrW6t1IKZM63hEjRtC7d2/Wr1/fmqzBGThmAK8e9hN2cUxBA/Ls/6dlsX6/v11mteV5xfg0b8vZ93YkV1EVPD4r8SweibFn6w7W"
        "Ll3Gqo+WsvHzlcSjseR3rAimaFzyP+SzPMwQlsOBk6SV6qAghCC/pBM9hvWn++D+9Bjcj4KyEgKhIIqqWgQ3oRMLR6ycsmMgetsapCnx+LysWfgZsUikxQ7NkTkUdyuj++B+JKIxd+Bx4cJF5kiOOyLZj9ofNLu5I2U482tfps+ooYRrXVeGwwoJiiKojYSpiTaQF8hC13VCoRDTp09nw4YNmfKQ8Y17dOHCgkYzhSZS4TSuUCh0aEey95MfzKY4J59t+3bj1TyNfrBJdwUPHp+PeCTKzg1bWLPoc9Ys/oxNX6wmEW/U5DqRXOe7h6NyW3siGcXF8jGWEqS90pKVl0u3QX3pMbgfPYYOoKRnV4LZIVSPRiIax9B1og1hSxaiKIfdLeFIQygCPZZg+fuL0m+H1XsNnjyWUG4ODTW1x0wU24ULF0cQtkuNtSJk95uKSPafCBNfwEduUUGLu3Aqrk0491RmXHwOkfoGl+weZkgkqqLSEI+yr6GOvEBWcmQ//fTTeeihh1pLdneGxZ5YwbxqXB2vCxsa4Mlkw2AwCDRqcdsKR2agKgp9O3Vh896dyQQszevFa2tyd2+pYNXHn7Dyo6Vs+mJ1E3eFVLnCsRDJdcr0mmbTKK7m0ejUrQt9Rg2h98ghdO3fi6y83KQuWU8k7Aiu3VGL44vgpkKaEo/fy65N29i8Yg1gJbA1B9M0EYpg4KTR6ImEq6Nz4cJFI1ISz1RNQ9MsKZwQCvFYjEQsblsv6hiGldfRd/QwPvzfG5hGUxcfYRfoGXvmSZx/8zVEGhpA2vaPbrdzWCEQJIwEe8O19KQ0qeMdP348WVlZ1NfXp/+6RW67YJHeT3B1vC5saIBTb9exwWsWoVDIFvcf/NMusEhL/05d+WDjcjSfFwHs2VrBqoWfsWbRZ6z/ZFkTuYKzZG1Ks+OT3OakCnayWU5RPj2HDqT74H70HjGIovJS/KEg0jRJxONJBwUhjm+Cuz+cCc/qhZ+SiMVblTOU9upOlz49ScQTh1zowoULF8c+nFU+1aMR9PuQQEN1LTsrdrFtzXr2bKlg365KavZUEa1vINoQRtcNBKB61APrDwlrnyNPnsy537gSRVEJ5eSgJxLoCT0ZhFHsfAkX7Q2JIhSqw3VAY9JgeXk5o0aN4t1330VV1ZbKDTtV1zSgBxbhdeECsBpFVboNHILbu3dvFEUhkUjg8WQUFE5CSolpmiiKgqIolOQW0Du/lP++/Dwr3lnIyo8/SdFSWQ3cWo4yO35ikk1wRTLhrFGqUFjWmf7jhtNv3Ai6DexDVl4OHq+XeCyGntAJ11kzVcWxCTsBl+eFIohHo0l3hha3s10bBowfSSA75MoZXLg4weEQXW/Ah+bxULt3HysWLGL1ws9Y/+lyqndVHtz4YZPflR99wm+u/jZd+vWi+8C+9Bg6gM49u5KVn4s0TeIxK59CCNzJdztCYvGO6ogdybXHVk3TmDRpEu+++26mu+p7mE7RxTEKDdht/7/Z4KnTYcyfP5+lS5cyatSoJHltDVJKDMNA07RktuUXX3zB3LlzeWruXFavWpXctom7QkcnuTRah1kk14o8ax6N4h5d6T92GH3HDKe8Xy9COZYGKRGNNTop2DKFEz3hSpoSr99LxYYtbFu9AUgjZzCs1YUBE0aiJw5/aWoXLlx0XJiGicfvxeP1UrFuE4tfm88X8z+iasfuJts5fW2y8mQz+s+WNKHRhjDRhjA1e6pYsWAxAAWlxfQbM4yBE0bRfUh/sgvybJecqJVsrSiu5KEdIIDaaDj5m9PfT5s2jV/+8peZFq3qf5hOz8UxCg1Yk24DR8awcOFCZsyYwdNPP82sWbPSkt5UoqtpGtFolJdffpkHH3yQ119/PbkU4SRyHXOa3KRcQaKoKuX9ejFo0mgGjB9Jp25l+LOCmLpBIhYnXG8pRk7kKG5LcOQM65Z8gR5PpJEzCKQp6dyzK2W9e6DH4m5ExYWLExDSNBGKQig3m50bt/LeMy+z9PV3kyuETh+dLNBziOPK/snGVTt289ELb/LRC29SUFrM4MljGT5zIuX9e1ml7MNRDN2wEuTcSflBQwhBXE8kc4acazls2DDy8vKorq5O1ghIgx72z45OLVwcIWjAx9j2dy1tJKVE0zTq6ur42te+xueff05eXt4BCWyOdEFVVTRNo6amhkcffZS//vWvLFu2rPGgmoZpyxXao7zw4YTzsCXdJGyyXtK9nMGTxzBoyljK+nTHFwygxxPo8QThmvrkMteJHsVNB6EIErE4qz5KL7MSQkFi0Hf0MII5IRpq6lw5gwsXJxhMw0yWRp/32HO88+Tz1O+rAewiQ2bTPro9kJpsDCkre6akasdu3nv6Jd5/5iV6jxzCuDNPov/4EWTn5xKLxNDjCZf4HgyknbhmGsSMBH7NmxyDO3fuzKBBg/jggw9QFCWdjhegExbH0ZvbyMWJBw14H6gAyrCIb7MMTdd1VFVl69atzJs3j9mzZyejuPtLF3bu3Mnf/vY3Hn74YTZt2gSQXMI3TRNd7+Dt7wBdrtXhFZYWM2jKWAZNGEX5wD4Es7MwdN2K5NbUJzs317qmdUgp8XisYhNbVq0DaLG0p1NApN/YYdb/3QHEhYsTCqZhEswOUbFhC//9/YOs/3QFYBFd0zxyRYZSiwKlytrWLV3GuqXLKO7ehdGzpjHmtBnkd+5EPBolEXOJ78HANE2L0GqAEBi6jqZpDB06lA8++CCT65lvv/bQWIXNxQkMDagB/gd8HSu7sUW25hDWbdu2Jd8zDCMZ0a2qquL+++/nL3/5CxUVFQCoqpqM/LYwG+swaE6X6wsG6D92OMNmTKDv6GFk5ediGgbxWJxwXb1V9EFRXJLbRlh2ZD42fraSeCTaspzBntnnlxRR3r+35eQg3GvtwsUJASmREkK52Xzy5vs887u/E66pS0Z0j2Y1zSbkV1FASnZv3s4rf3+Cj55/g7FnnczoU6dS1KUzekInHoslxwsX6WElKcuUIEjjavL48eN54IEH0uX6OEw4lfC6cJEsOvEn4KtYnrwt2pM5EoS8vLykJMHr9ZJIJHjooYf45S9/mYzoqqp6TJBcUvRB0jSTutyuA/owbNo4Bk4eQ1F5KYqiEI9Gm5Jct+M6aAgBhq6zdsnn6bdTFKRh0HvkELLyc4jUu2U9Xbg4ISCtwcifFeCtR5/l5QceQ0orMayjlY2XNvlyJHD7dlXy+kP/YcEzLzPq1KmMO/tkSnt1wzQk8YitN3b7sbSwHI+aykkAhg4dan/eohzS8eINAjkp77k4waFhRXRXAj8Hfoald1HYL9KrKApSSnw+H+PGjUNRFLxeLy+++CK33XYbS5cuBQ6N6DqdhaIoh1320DSaaz04uUUFDJ0+geEzJ1Hevxe+gI94LE48HEXa3oAu2WoHSImiadRVVbPxi1X2Wy2UmrYHkl7DByLcyK4LFycMJBJ/MMCrD/6HN//1dGOScxtdfByiJFJ9c53+RiT/aeLk0JKjQ6vnbGt+nXNtqKnjvadf5qMX3mTotPFMPG8WPYb0B4SdaCdd4tsMpLQSvRvH28ZiaWVlZRQVFVFZWZkucc3EyktquZSeixMOGo0Ja3cBXYHrAIQQpqqqSmpimq7rXHvttQwYMIBNmzZxxx138MgjjwAHT3T3J7mO/OFAN/BDR6qO2FmO0rweeg8fxLCZExkwfiR5xUVJXW6Do8tVhFWO0kW7wDQlgZCPNQs/pa6qJm22rZQSj89LjyEDLHcGVwfnwsVxD2maBLKzePXBJ3jzX09b/baUmY8J++VhQNqIYLNQVMWKMu+XuJbR+e9HfBOxOEvfeI9P3nyfQZPGMPnC0+kzcihC0Ogw4RLfJlAUBU2xkpOteYnV9xcXF9O9e3cqKyvTJa456Hy4z9PFsQOH8DpT5uuBz4UQ35NSdtN1vYm84corr+S3v/0tjz32GLfeeis7d+5MPtAHQ3RVVW1Ccn0+HyNHjuTCS+awcM0XzL3/4Uys"
        "R1qFU/BC1/XkeeaXdGLo9PGMPGUKZb17oHk9B0oWXF3u4YEdaNn4+UqsCIfabGZ1ssJO/97klxS5/rsuXJwAMA2DYE427z/7Cm/88+mkXjdTstsY1LDyMDw+LwWdO1HSoyuFZSWE8nLwZ4VQ7GhxNBwhUltP9e5KKrfvpGrnHhr21TSVTQiBYtsjtmU8SpJlQVKKsXzBIpYvWMTgKWOZdtHZ9B4xCNOUxCIRBHb54hMZtn5XVRQ8SqMbjxAC3U5c69mzJ0uWLMlkby7hdZGEo+F1iK2iKMqfy8rKnrjlllueTCQSp6xevdro2rWrOnr0aEaOHMkNN9zAQw89ZH1Z05KENROkFlswDCMpWxg2bBgXXXQR55xzDsOHDwfgoUceZi4PZzKDS3ssh0ybpklOTg6nzJpFwYjelA3tS25BPoadTBCPxhCKWwziSEBRBPFojA2frQQaZQsHwO73ewzpjy8UoKHara7mwsXxDNMwCWRlserjpTx/3yNJD+5MxhgnOOJ4xHcf0p+h08YnS7l7vF5Uj2ZX8pTJUU8IS9pm6DqmYRCpD7N78zZ2btrK5mVrWP/ZCmp278VMdWew/eMzjjjLxkizRcgly99fxIoFixk6bTxTLzqLnsMGWEUsIrET3tVBSolP9aIqapOkIueaDBo0KNNdlR6G03NxjEJL+b8E5E9+8hPvHXfcUXXrrbfeB5wKiFdffZV7772Xt956K5k0IKXMWGfrkE/DMJLktby8nAsuuIALLriAGTNmNJFNAAR8/oP6g5qL5vbp04cvX/Flrr7iKnZ4oryz7jNIGHY015YsqCdu53IkYRWb8FC5tYJdm7el3dYZIHoM7o+pGyf0AODCxfEOR760b+du5t7zAIau21KG1jW7qS4vw2dMZPKFZ9BtUF88Xi96wvJHj8diyGg0haQ26kKd5GUhBL5ggF7DB9F39DAmnX8a9ftq2bJiLasXfsaqj5ZQtXNPkoCn+v9mCuc8nXP+fP5HLHt/ISNOmszMS8+nrF8P4uEoekI/YVcZJZJQkgMcmEffp08f65PWr3tZe5+bi2MX2v5v3H777Yk77rhDXH/99W898MAD6//5z3/2vvrqq03sJDZVVTOOuKbKFhyf3mnTpnH11VdzzjnnkJeXl9zWMIxkh6OqKn37WWWwHR1USw1bpHRUqdFcv9/P9OnTueaaazjnnHMIBoPsDtfw+jvPoWDJFdxo7pGHNE08Pi9bVq63o+pKsxFeR8IdyA5R1rcHejzh+u+6cHGcQ1EVnvvjQ1TvqrQ8djNwY7AKQZh07tmNs264nIETRiFNk1g0RiIaT0ZLUyt2pYM0DGJhHSmxCbCfwVPGMHjKWOr3XcTapV/w2bwPWb3wExKxePK8D4n4GiZL33iPlR8uZcrs05l0wenkFOUTrY8gpXlC6XsFVt+f6w9ZbzTjG9WjR49Md+dEeDt2hSsXRwQHEF4hhLztttu0O+64I3zVVVc99qMf/einQgi5f4Q2HVKJrq7r5OTkcPnll3PllVcyYcKE5HbOvlRVRbWXqh3SOnTIUEaOHMknn3zSLMl2jmEYRpOs3d69ezNnzhzmzJnDiBEjku8n9AQvffYBEtA0tcUiBy4OL5zlw03LVid/b+5OCKEgpUmXvr3Iys+1ynW6hNeFi+MSpmESzMniw+dfZ8WCxW0gu9aEedSp0zjvm1cTys0m2hBOfnZQK3cOObZ/lYYkUm/t0xvwM+qUKQyfMYkd6zex5PV3WfLafBpq6oBDJ76R+gbe+NczfDLvA6Zfcg5jZk1H9fiIhcPW33OC9IFSSoqycq3/c6Ckobi4GI/HQyKRaCkg5nwlj8ZqaykhfRcnIlp6ejRAVxTlBtM0/4JVkCIj8WQqOe3UqRPXXHMN1157bZMlCEdj1dLD63x/0aJFzJkzh61bt6aN8nbp0oVTTz2VCy64gFNOOYVgMJg8lm4YeDSNL7Zv4OnP5hPQvC7ZPcoQQvDHG37Azo0t31dnwJt52fmc840rXf2uCxfHKaSUqJpGQ3UNf7zhh9RWVdtRvvT9tEN2T7nyQk7/2iXEY3GMeOLw9hMSTGkiEHj8XjSPh73bd7H4tXf4+IU3qamsAhorwLXZachZrbRJcO8RgzjjusvpNXQA0XAEQzeOe5mDEIK4nuDysafSp1MXTClRHK98e8V3165djBo1ioqKipaKFklA+P3+vVOmTBn15ptvbsGttnbCI+2TY5pmMba2t9Ud2dpZwzDIycnhRz/6EZ9++im/+tWv6NOnT1Jq4ERmWyK7pmkmI75Dhw7lmmuuwefzHXCsUaNG8bvf/Y63336bZcuW8fDDD3PuuecSDAaTUV8hBJqqkjB0Pti4DBXR3k5nLtoAKSWqR6Nqxy6qduxOu63TgXXp1xNTN0+YyIYLFycapCnx+r28/cTz1O7dh5KBM49ik93TvnIxZ153OdGGCKZuHP5Jse22IBTLaixcW09OYR6nf/USbv7r3Zz+1UvILsyzotN2vkubICXSHrsURWH9pyt44P/dwUt/eww9oRPICjbxjj/eIBAYpkmWL5CM8DbX9efn51NUVNTyfqwBw8zJySns2rVrT4CLLrrIHUROcLT0NDpP00asKHCLDUUIgaZpSUL7pS99iSVLlvDzn/+csrKyJPl0CHE66HaSQiQS4e6772bgwIHceeedJBKJ5APuRAS3bt1KRUUFY8eOJS8vj0QigWEYFqlS1aRvoxCCFTs3U1GzF6/maVK5xcWRhTStpJSK9Vss/W5LA5u98KR5PJT17o6RcPW7Llwcj5BS4gv42LZ6IwtfeqvR/SANnIje5AvPYNY1cwjX2laSR7iPEEKgqFaCdENNHcHcLGZdM4eb/3I3ky88A83rSU7c26rBdVZChaKQiMV569/Pcv/NP2XVx58SzAmhKG0vwHFMQIBhGuQGQuQGsuy3Gu+rM2Z4vV7y8/OT7+0Pe1yRdXV1jBs37gyAuXPnHv7zd9Gh0dJTaGLRjleAHVhyhoQz63RemqYl3Rr69OnDU089xdy5c+nTp88B5LM1OEltixcvZsKECfzwhz9k69at1smkPNiOr+GePXu45557GDBgAO+99x4ej+eApARFCExpsnjzaitqcLBXyUW7QNjZCNvXbLB+b6FdOBXVOnUrI7sgz0poTFP4wy0K4sLFsQlHzrDg2ZdJOIVl0hBeYZPdPqOGcM7Xr7T0uhbbPXInvf852cTX0A0aamrJzs9l9re+xtfvu5PhMydavrJ2MaU2E98Ufe/ODVv4x/d/wQv3P4qU4Av4MQ/CsrMjQwCmlHTJ69SirMWRPBYWFqbflxAiEomgKMqZUkqftWvpDhYnMNJFeAVQCVwLRAGPlFI6kVzTNNF1ndzcXG699VY++ugjLrroouRn6WQL+8MwDFRVZd68ecyaNYvPP/8cTdNa/b6maWzbto2zzjqLhQsXNvHsdaIEm6p2sb1mjxXdPU6XgY4VCCHQ4wm2rbYIbzrnDYDOPbsSyApZ9zRNU0iYh7cMtQsXLtofUkq8fh8V6zfz6bwPWo3uOmQ4lJfDl75znUWOzI4jd7KIr5WsHa6rp7xfL7582//j2nt+TL+xw5vIFdpKfJ1oL8A7T/yXv9/6MyrWbyaYk52sHHo8QAKKUOhVWJr8vSV07py+poRiRdrkvn37BgBDnN23z5m6OBaR7uY7VmQvAZM9Hs9/e/ToQXFxsVFWVsb48eO5++67WbJkCffccw+FhYUYhpGRdCEVDtldtGgR559/Pvv27WtSgS0dnKordXV1fPvb305KIqSUSX70xfb1GKbhxgCPNqREqArR+jCV23Yk32tpW4DibmWoHi1txMcwTWb0HUHI6z8+l/hcuDhOIU2Jx+NhyWvvJgv/pE2yEBZJnnX1HIq7lxOPxjqktaSzEhqPRImFI/QbM4yv/PIHXPOL79NrxCBrldI0k2XrM4W0ia2iKmxatoa/fut2PvjfawSyQk0S3Y5VCKyKrbn+IOX5xdZ7zUxm"
        "nPfKyjKy2DV37NjhAU5vvzN1caziAFuy/WBiTZSWGoZxwcKFC+/xeDy3mqapFxQUJL/rEF21jQkDzux8z549XHrppdTV1bXJ59c5thCCJUuWsGnTpqScQlVV6mMRNlTuwKO60d2jDSnB4/GwY/3mpIVPS3fEIa4l3csxWignrAiFcCLKzL4jmdZnODtrq6gK1xFQXcs5Fy46OqSUaB6N6j1VfPLWe9Z7ZsvPraPb7TG4P+PPPolwXX2Hd21xIrLRcAQhBIMnj6Hf2OEsX7CY9556kc0r1gAkgzSZjlGmYeXEROvDPPObv7Fr4zbOuv5yhKaRiMc75CQgEwgBCdOgR1FnQl5/0pGhJZSWZlZEbdmyZQBnSCl/gevScEKjNcILYF544YUqQKdOnX4EjAOmmaZpSClVx3XhYOBofH/84x+zfv36ZKnig9mPI7GAlIy7vTuojtQT8LhWZEcb1gDnoXLrTvREosWCEw5UVaW4e7l1T/fr9FQhaEhEGdm1LzP6jkAi6VdczvKKjZimbFKZSUDS6seFCxcdA9KUeLP8fDpvATV7qpIlhFvc3iY/J10xG9XjIRFLHDPVMR0CGmkIoyiCESdNYuDEUSx7byHvPfUi2+ychrYQ36SUQwjef+Zldm3aypzv3UhuSRGxhnCHnww0BylBFQqDS3tav5NWyUaXLl3s7zV/vezAibJy5Up27NgxrLS0tIcQYqOUUhFCuMT3BERGU8G5c+caTz31lBRCxCORyJdN09yqKIqqqqpxsLNJJwq7ePFiHnrooSb623TYn7g4EorevXvTvXt3u/Sxtc26Pdtcp+mOAns5cteW7davLRBQ5/3sogKy8nMwDbNJp6cIhYgep3dRGWcPmQTCWgrrVVhGbigbze8lmB0imJ1FICuELxhEtV1EjvUlPxcujhcIRWAaBp/P/9j6PQ21cYhg75GD6T92OLFw5Jj0orXGSkGkvgGkZMxp07nh97dz0fduoKRH16QWV8kw/8WRRiiqytolX/DAt+9k5/rNBLJDx1wymxCChKFTkpNPzwIrcqu0MkZ07tw5rT+/xQUUUVFRYdTU1GQDTtWrY2Om5KLdkXGvIYQwpZRqMBjcqijKbKAay73hkJ6s3/3ud+i6nrbh2sdPNnTH+cGJLJumybe//W0CgQCGaaAIhWgizrbqSlRFcRlvB4AQAiOhU7mtwnqjpXtt3+P8kiI8Pl8yuxmsDjBuJCgM5jJ7+DQ8igrSmjzlBEIMKCpnxcdLefc/L/H6I0/x/jMvs2bJZzRU1xDMCuELBpKrAS5cuDg6kFLi8XrYvbWCjZ+vSL6XbnuAieedhurRjnl5mhMkCtfWI1SFCeecyjf+dCfnfONKcoryMW2Ho0wT20xbUli5fSd//+5dbPx8NcHsrIwq1XUUCMCQBqO69kPLUJZWVFREYWFhWumD8/5HH30EcFr7nbGLYxGZSBqSEEI4MobFUsoLgeeAHNpQiQ0ai0ts3ryZ559/3srOTUNC9ifD+0eCv//97/PVr37V9vu1TmN3fTU1kXo0RXO9dzsAnHtctWMPQIv3xCk1nFdciC8QINLQkKzKp5smfs3Dl0ZOJ9sfxLQjHKqq8swzz3D77bc7eq0myOtUSK+Rgxl16lT6jhqKoqnEGsIdQupgJVgKN+bg4oSBNCWaz8uGz1YQbYi0VCkLICl1KO7WhX5jhhGPRFuM/B1rUFQFpCRcW4/m8TDz0vMYPnMS7z/9Mh/+73VikaidyNd61TnH676uqpp//N8v+PJP/x8DJowkcixonREkDIOirDyGlvUCWo7uQiOJLSoqoqysjMrKytYCZuL111/n6quvnrB48eIgEJFSCiGESwxOMLR5XcgmvZoQYh5wHrAXi+xmLL51Guarr75KfX19csmqheMhpSQYDPLEE0/w1ltvcckllzB+/Hguu+wyXnvtNX75y1+aQghTUZQkb9hRs5e4obv1CjoCpDVwRcNhavfuS77X4sZYEV7VoyYjwY627fzhUynNtWb12BrwP/7xj3zpS19i2bJlVr351JcQVO/Zy9LX3+XB797FP77/C9Ys+gxfKIjq0Y5KFERKmTyuqmkgrESUdBpGFy6OF1hyBpO1i79ofVu7Rx82cyLB3CyMZjT9xzoUVUGakoaaekK5OZxz01V8408/Y8jUcUhTOkvzre7HsS6L1of510/vZe3iz+1Ib8eWNwgBumkwqccQ/Bnk2whhuTn4fL6kjrelwIXNK5T3339f1tbW9h49enQfm+gee5oYF4eMg7rpQgjdjvS+A8wElmNFiw3aICB44403DigWsd9xEELg8Xh49tlnueSSSzjppJN44okn+Oijj3jsscfkrFmzdPvvUEgh3RU1lSi4xSY6AiQSVdWordxnRVbTbWvfsJzCApsUWm0gpic4bdBY+hV3xbRlCYqi8P777/Otb30rWbJa2lrd5Mte7nKixGsWf87fv3sXT9/zV8LVtZbe7QhKHJxEzWBOFlJKGmpqrQldbhbegC95zi5cHI9w2n9dVTWblq1OvtcSTNNEURUGTx6DHk+02b/2mIHALl6hE66po6RnV6684ztc8sObyO1UkOzvWluRcuzO4tEo/779t2xavhq/XY64I0Kx+/au+cWM6NoHiWxTBH/YsGFpr4lz3bZu3WouW7ZMA8a2w2m7OEZx0L1HirzhCyzS+yRWpFeQhvg6HV4kEmHVqlVpNZXOUtf3v/99TjvttGT5YMMwTMPSNQhAM01zNVaBjG124zf31tcc9eVqFxasakoqtZX7iEdjyfda2BiAUF6OlZChCCLxGFN6D2Vc94FWuWgadXB33nlnsk21lPTotLFUXdzHL77Fn77xY1YsWEQwO8syhT/MRNPSLnqJ1Dfw0l8f5a/fuoP7v/lT/nLL7Tx9zwNsWbEOfyiIoqquztjF8QlplRffsWELdXv3JZNZm4PzrJb3701J93KL8B7nfbpTvCIRjROPxRh7xkxuuv8uRp4yJaUPa430Wv1cuLaef/3kXiq37cQb8HXIPsWUEk1RmTVgzEHl20ycOLFVZwsncPbMM88AzDiU83VxbOOQpssppHePEOJS4AZgJ02Jb5OnzGmYe/bsYdu2bU3ea3JitmtDaWkpt9xyC4ZhSFVVDfulqKqqmqa5G7hdUZTxwGOAD6AuGqY+HnUT1joKJCiqSkNtXavLc05byCrIBQmRRJzh5X04pf9o67u2xEUIwbp165g/fz5woK67xf2bJtjnsG9XJQ//8Ne89MCjqF4vqqYeNieHRt/RvTz4vV8w77Hn2LZ6PVU7drN9zQY+/N/r/OVbt/Hs7x4kHoni9fs7bFTGhYuDhZQgVIWKtRvtvqBlfanDbfuMHEIg69hzHjgUCMUiaeHaOrLzc7nsxzdz4XeutVeBWpc4SNNEqAo1lVU8+Yv7SETjVpJ3B1o9UoQgEo8zufcQuhWUtOq7mwpnu1GjRpGTk5O24p5NiMULL7zAnj17hrtlhk9cHPL6kE16he1t9wAwGrgPqMcivgoW8TWwGpkEqK2tpaampkWxudN4Bw8eTFFRkakoilCs3lEFtgA/UxRllBDiDiFEDVBmghegLhoWkUQMRShuwlqHgAQhCNfUWr+20qkJRSE7L5dIPEqPws6cM3RSE+mL016WLl1K3DZab2t01rSX/iQw79HnePzO3xOPxPD4D18kRFFUnv/TI1Ss24SqaUmNsbCt9YyEzgf/fY2/f+dnVG7bYS9FnjiDvIvjH0IRGLrOluVW0YV0BMyZ8PUaMcjy7j7Oo7vNQVFV9IROLBxl8uwzuO63P6VTeakt9UifjCYNSw6yefla/vvHhyzXmw5CeIUQRBJxBnbuxtTew9tEdqFx9bdLly6MHj06+V5zsGUNYu3atcydO7c70MXW8Z54DeoER7sIooQQ0rEtE0JUCCFuxtLK/A7YhkVSVUARVqs2wuGwYX+32X06D+aWLVuk/b2oaZrvAF8DRgohfiqE2C6l9NgzNRPTVADC"
        "8SjRRPy4yeY95mGXvQzX1me0uaqqoCkUhnL50sgZeNWmThupqwTW7g/uPkvTSnxTVIXP53/E3793F9W7KvEFA+1KNKVp4vX72bpqHSs+WIJQLK2eo9eVppkk2aqqsn3dJh783l1UrN2IP9Rx9XcuXLQVQgji0Tg7N24F0sgZHD/uwnxKe3VHTySOu2S1TGFNigUN1bV0H9iXG35/G71HDLLsyFrxIzZt0rv41XdY8vp8Ah0giU0g0A2Dztn5nD98irUSexD31tHnnnPOOa0f04I5d+7cHKAHIObOnXtiNqgTGO2aAZAS7VWFEKuEEN8GRgEXY2l8V5vWyK5mZ2ennZ7aBEBu2bKFp59++pfAGFVVZwoh/iGEqJJSqjbRNezZWkIidbAIb0eZybqwptFSSqINEfud9B68qArZgSy+NHw6uYGQrds9sG8KBAKt+jdnAmtQUNm2aj0Pfu8X7Nq4FX87LqFKrISU7Ws3WkuNabY17EFs3849PPzDX7Fn6w68QX+H1N+5cNEWONUW9+3cTV1VddptHcJb1rs7obxsDN04ISO8qVBUlWhDmKz8XK7+xfcZMH6kVZinVXmDFT19+YFH2bt9Jx6f9+iOj7YrQ1leEUGnhPBB7MaJ6J533nlkZWWllTUYlrex/OCDD/jqV786CZBz5sw5+L/BxTGJdk95taO9hpRSSdH3PmVrfEdpmjYWOG/NmjX3Aq0N5GY0GhUXXXTRKiHE8uuuu87jEF0hhEN0nSc3LhAJgEgifnBPkIvDBCvCG4tEW9nKgs/v48LR0+kUysGQ5gGReqejGzhwYDJhobmOTiiWO0Mm0QMnWrJnawUPfvcutq5c1+4ViyJ1DRkN2k5Upnr3Xh6943dEbJ9OdxLn4liGlBLVo7G3YhfRcCT9ZNV+TEp7d8Pn97tVEm0oqko8FkfzeLjyzm/Td/Qw25khfV6EEILavdW88c+nO4S0QRGCuljEKh98kBMZR9bQq1cvTjnllIzs2+LxOI899thFWCvOrqzhBMNh83gRQpipEV+b/IaFEEuFEM+ff/75vwRsUWd6oa2maVc+9dRT6t/+9jcjhejujwgKMcDy33XbcYeDkUhY/2mlr832B+leUGJZ1DRzH52BcvTo0QwePNiKHGlaUuerqqptUWY7gGSoDzMNa4msprKKR370a7au2mAny7TPYJtTlJ/xQOOQ3op1m3j6Nw+gah3bPN6Fi1YhLZJSvasSID1Jsz2pS3t1txJST/DobioURUFPJFBUjctvu4Xyfj2TdmQtwYl+Ln3jPTZ+sRJfwH/Ufb8NwzjkUdrpT7/+9a83+b0FqIAZi8WGYbk6mbh+vCcUDvvNdiK+Dvl9++23NSml2qdPnziww96spVaqAFLX9fFz5szpRTMNVAghbWlDvTRlLYBuZFwDw8WRgG09ZOiZ3RfDMEjoLU9anKptXq+XX/3qV1Yd9kQiGe21revoOaQ/J18x23JfyLBUp6MLq927j0d+9Cu2r92IPxQ4JNIr7LLK3Yf0t8obk36wT56LLbVY9t5C3nnyeYt8u5EuF8cohLCer6qK3fY76csJCyEo7t4Fwy4976IRiqKQiMcJZmdxyY9uJisvx84NTlOhzE4YfG/uS3b/c5RXjNrhlqqq1bfPnDmTCRMmJKu4tnJUE/g50Bsrmd4lvScIjuiNFkLImTNn6ralVC2w1v6opSfPaZwh4PKU95rZtZACscv+zV377XAQCJFZc8vEYky1vWrPOussnn76aUaMGEFOTg5FnYqYPnUav/rDvXzt1z/i7Buv4Jq7/4/cToWWr28GZTYd0luzp4pHfnwPldsPzcdSCEE8FqO4axkzLj4nWRQjUwIuFMHrDz3Fxs9XWoTZJb0ujkUIgWkY7N1hddMtS/mtLj4rP5ecwnw3abMFKIpCNByhtHd3zrrhCiu6ma4Ig2GCgJUfLmHnhi1HXdqgKVq77Mc0TTRN43vf+14mmwusllcI/IXGolXujOoEwNGa2TiN66M2bPsVIBeLAO/fOJ3ft0H6OtwujgLsfli0klHsIBKJkHDkD2ngaLhmz57NkiVLWLp0KUsWL+Wdd+dz8VevIoFJ/b4aBo4fyY2/v53ug/pZWt1MI7128tijt/+OSH0DHu/B62gVRSEWiTLzyxdwylVfShLX1kzkLdsmgZ5I8N8/PkwiGrOIsjulc3EMwjRNanbvBdI0Ybv/zisuxBvwt5roeSJDVVXCtfWMPWMGw2dOsov1tNy/KUIhHo3z+Tsf4vF7j9rkWUpJwOuz/n+InZkT/Dj33HOZMmUKhmG0FuVVsSK7pwL/h1Wh1Y3yngA4WjfZecpepXUdjWJv0xWL9LZYB9uQcjmAR9VwGUFHgsV4vT6v9Wsro1c0GmXvXntQbIVgOgVKFEWhd+/edOtmlR5ev3UTuqGjaR7CdfUUlBbz1V//kGEzJiYjuK3BkRRsX7uRx3/2R0zb8P1QoiJGQue0a+bw/x78NQMnjk5WRUoHZxDbtno97zz5vOXP60Z5XRxjEIqCHk8QrrPtCVu0JLN+5hQV4AvaBVjcIEaLEMJaFTv1qovwh4JpPW2dK75u6TKiDZGMVrzaG9ayrSTbF2h6UocAp9rmz372s4z6dhp5xZ1Y1dcMLCLs4jjG0SK8TnbkMuBz+73WRnAJfBsoSvl+EwgplwH4Na9wE9o7DpxMXF8wkH47+6ZJKZMeu5mQS0fHZZomhmm5OtTHo7YGWKKoKrFoFI/Xw+U/vYUpF56ZTOJoTRvouDesWfQZz/3u73h83kPuoCMNYYq6dOai791AYVmJFcFq5TyckqLz//MC21dvsCouuY3cxTECJ4M+XFOfLC/eMqxnIacwH4/Xi5Tu5C4dhBDEIzHKendj1KlT05YfdiK629ZsoGb3XlSPdhT6EQFSUhjKtc6pHfbolJafMWMGF198cSZRXucCqcC/gG64et7jHkeT8CpADHiCRl1NS3BmY+XAjzkwKiwBwoaxEajPCWUp0mUDHQd2dnYwJ8t+o2Vy58zOt2/f3qZDCGFZkKn29+tiYSsam7JfwzBIxBNccMtXOO2rlyQ7+tZJr4miqSx65R3e+vczBHMOzbxdVVUidQ3kFORxwbe/hqKq1jmkOQ1pexHHI1Fe+fsT9jm7TdzFsQNFUYg2NJCIxdNvaD+X2fm57qQuQwgh0PUE4848CY/Pa/nzpunXErE4ldt3HJVyw6Y08WleikI5QPuJZx33nl/84hcUFBRkYlPmVIHtCjwKeJxdtdMpuehgOJqzGSdK+yhQZZ9LuidPxSK6NwFTSFmCsKu8iRyfbwOwoSA7F6/mkabbWXYQSBCQlZ/X+HsLcDqo5cuXA636NDcLwzQJJ2IHaLmFsCILkfoGTrtmDud98+rk2WRCeoWi8NpDT/Hp2wsOmfQqmkqkvoFBE0Zz0mXnWzKLVpL6HCnGyo+W8sV7H+MPta9PsAsXhw3ScgmIx+IYifRuLY6mM6sgz7LOcuUMrUIoVgW70t7d6T6on/1mCy43dvS3auceFFU9onxXIDBMk2x/kJLsfOu9drq/Tk5Hjx49uO2229IWokiBo+edilUZ1sSVNhy3OJqE14nSVgAP0ujI0BpU4O80JrA5f4MihDCAD3O8Qfwei/C6XWUHgBCYhkl2fi5CkJH/45o1awBaW5ZqAmevumkQjscsScN+vbkVSRU01NYx/ZJzmf3tryXHhbSdo5TJ1zO/+Rvb126yShAfgpZWUS3Se/KVs+k+uJ/tyNCKntf++eYjTxOtb0DRjnyExoWLtsPSlSaiMXTbnrDFZCX77WBOyHUkaSOEopBdmGf9v+WtAIhHokd8LiEE6IZBl7xO+DzeQ05Y2x+OtOGmm25i+vTpmUgbwOIQOvAN4Eb7/y7pPQ5xtPUqTpT398AuWie9zhLEAOCvNEojUh/bdzQEIa9NRtzowFGHsO2Isovy0bzetEkVDoH8/PPPicViB5UkZkqThJ6wlrhaOB9FUWiormXyBWcw+/9d"
        "m/phi/uVUoKi0FBTx5O/+BOxcARV0w6JcErTRNU0Lrjlq3gDfmtf6c7BdnbYsWELC19+G38wiLuS0XEhEK5rDCS9p+OxWGMhmJb4rt2e/aHQQZedPXEhibdS0dJBquTrSEECqiIYVNLd9k1v/2M4/ftf/vIXQnYbakXaIGhcQf4TcDoWz2gf3zQXHQZHm/A6EdodwO20LmsAq2HqwCXAT2mcjTnfWwDs65xXqBqm4TKBDgBFUQhHIgzo24+S4pK02zqD3dq1a1m1alWT9zKFYZoYsnUrI0VVaaipY/IFp3PeLV9JS8ST52fLCirWbeKF+/+N139oyWNCUYiFI3Qb1JdZV19kdc6tESRLg8G7T71AzZ4qNO1oJJ64aA0CgW7qRBJxTLttKXaipBDixKoGaRdFSOp3M5gEOI4DrqSyLRBojhtOS3A00gV5VgT9CF5ewzQJev10yS86bMVEnHyNgQMH8stf/jJTaYOzgQI8BgzDjfQedzjahBcaSe/fgPk0amrSQbO3uQP4GqALIRQppSKE2Aos6FvaTRpSmm5XeXShCEE0EackK48rJ5/BmFGjrPdbmHE79jKmafLKK68AbdHxWh25bhpkOk4qqhWxnXbRWZx53WWt+lg656OoCgtfeotFL88jkJN9aHpeVSVSV8+UC8+g57CBrUobHGJevXsvH/7vNdur1CW8HQlCCGJGgn7F3ZjUawheVSOWiBNJxInpCRKGjm6eWPproQh0m/BmQna8fq9dTOFwn9lxAilRFEG2kyvRwnWT0nKu6dyzK4ZuHNEqdqqiENXjzF36DpUNNSh2otnhkDbous5NN93Eeeed1xZpgwkUAP8BinGdG44rdIQb6bR0E4u8VtPY8NLB2eZvwJcB/eabb/bYZYafLc8vFtpRWLJx0QhFKEQScbrlF3PpmFMIaj7GjB0LZDbgPf3008lSkW2JYLY1cuaQ3pOvuJDpF5+TkU+vNC3S+cL9/6Zi3cakQf7BwhmEzr7xCjSPx5Y2pP0CCMEH/3udvRU78fgOviiGi/aHYZqEvH5OHzSO0waO5bop53DusMmMLO9Lj4ISOucUkB/Man1HxxlatySzIBQFzXPoFoAnEqS0kmGLu5W1uI1QFIQQdOnbg6LyUvR44ojOJ6SUqEJha/Vu/vnRq6zYuSm52tHe/Zcjh/vzn/9Mly5dkn7trX2NRtnkfwCf/b477ToO0BEILzRmRq4DrqdRy5vuCRA0NsJ/Atf++c9/jtlE6tUg2p7O+UVKQk/IE2rpsIPAIrsxehZ25tIxp5DrDyGRTJkyJbnk1BLpdT5bsmQJ8+bNszTAbSCTSYevNvSfihBE6xs468YrGHny5FZJrxNlDdfW8b8/PmxHog7+cXKkDT2H9mf6xWfb0ob0x1eEoKG6lg//94YlrXCjvB0CihDE9QSTew0hNxDCME3yAlmM6tqP84dP4arxp3PNhDOYPWK6FeE62id8hCClJN6aJZkNVVNRVKXdI3/HMyzJSIJuA/uieTzNOlw41l1jz5iJPxg4KkU9JODXvEQSMZ75ZD4vfPEBDbFo8txS77ml85WY9s+2kGLHtaFLly489NBDqLb9YxucG2YAD9DIT1wicYyjoxBeaBSJP4VV/cSRLaSDSPn5N9M0f6AoihRC7PBpnucGlPcUcUM33ZyRIwtFKIQTUfoVd+WS0acQ9PowpYlAMHbsWLp27dqqXtYhm7/61a8O6vht7pqEQErQY3EuvPV6egzp36q0wCHF65Yu4/1nXiaYHbIGkIOE5VMaYfql51HcrUur2jNpR3kXvvQWe7btcKO8HQACQVzX6ZpfzNjuA6yIlh1pMu2XEAKPqtEpKxef5j2xErNaa5+OY4pixzPc5pwxhCJIxGJ06deTfuOGW21Ps0meoqBoGqZh0Gv4IMacPoNIQxhFPTotz5QSVVHRFJVFW1bx0Ecvs2rXlgOivQ5BTdW+g+XwYcrWp0OOtGHWrFn85Cc/yTTKCxbBTQBXAT/C1fMeF+hIhBcaG9VtwENYpDe9aWMjtTGBXyiK8s9OnTplAX/sEsjDo3qEW4TiyEERCg3xKENKezJn1Az8Hk8yS9Y0TYLBIGeddRaQXtbgdExvvvkmTz31VLLjSg9rf5qioClqm8dKoQgMw8Dj9XDpj75JXnFhq1XQHOL+1r+eZfvajYeWxCas44eys5h1zRwQ6a9RMspbU8eH/339qEV5T7D0q9Yh4KR+I/GoWrINOoN2akKipqjkB7MwTgQ3GfvPa/3ZEPa/4ri/JIcLpmFy1nWXkdupACOhW5FR08TUdcr79+KSH96EojkysaN3kaVNWIMeP/vC9Ty19G1eWPYB9bGI7bAj2ReuY+u+3Wzcu4NNe3eyq3YfUT2edD8RtN6mHKuyn/70p5x99tkYhoGmZWTA4ATdfg5cisVFXOeGYxgdsUtxpAomVhW2S8isoUn7O6qiKB9le7Ivq45W/+Lfn867ZO22jUbA61dd3nt4IYQgEo8xplt/zh46KZmQ4JA2J3Hg3XffZcaMGUD6zsrRYJWWlrJw4cKkDqu15IO4nuCBBS9QHa63iW8bbc0Mg0B2FqsXfspDP/gl0jDTSiocMj9gwki+cvf/EY/EWiztmQmkNPEG/Dz8f79i5UdLEYrSoj7YsV4LZoe4+a93k1dchJ5IHLFEFCEECUNHIA7qWh9PEHaC5vAuvZk9YlraVQzns5eXf8xHG5cR9PqPa3s50zQJZmcx79HneOmBR1FUtflET7s4jC/g59sP/4bcosIj2p6PB0jT6j92bdzGW48+Q8U6yzO8//iRTL3wDPyhEIlY/JD6qPZGsihQIk5pTiE9CjuzrWYPNZEGYom47boj8GoeAh4vxdn5DCjpRt/iLoS8Vsn61uwuhRDs2bOHKVOmsHbt2mS/3QqchzIKnAJ8QGaJ9S46IDpahBcaG5jASkZ7jMwjvSqWY8OEmljNh9dd89WGfp26RH2BgGIahmvQf5jgdDLRRJzpfYdz7rDJB5BdIJl8NmXKFMaPH598ryU4nVRFRQWXXHIJDQ0Nydl6OqiKil+zTc0Pok9X7NK/AyeOYtbVc1rV8zqfr/roEz5/50MCWcFDKkiBtF6zvjIHj8+bNoHNifKGa+v58H+v4/F5j5hZv2JPcLrkFXHO0EltF04fZzBMk2xfgJP6j2p1W+cqdc4pQBEnSHKtoFWS5Xyqejz2M3dCXJl2hVAU4pEoxd27cPlPv8WNf7yT6397G2dceykev7/DkV1IifZ6fVQ21PDRphXsqN5LNBFPSoA0RUU3dWoiDazatYXnPnuXBxe8xDtrPyWSiCU1wM3BIbfFxcU89thjBAKBjGwosZqkBAJYXKQc17nhmEVHvWlmys8rsIpMOMsLrfWAmmEYphCi5O+PPPTV/3fpV5XqzTtEIDuEUJVD0li6OBCKEBimgWEanDl4PCf3H53svJrrTBxyePPNN2e09O+4NLz//vvMnj2bqqqqpLyhJVKpKgpZvgCmNEFKTNNEmmbyeNJ+L+3fpSqEa+uZeel59B83onWrMPvn6w89RUNNHaqmHvRYbSWwRek2sC9jTp/RqnG6o+Vd/Oo7VG7bgcd3aN7AmcBx4OjdqQuXjD6ZEeV96JSVRyJNMuLxDCdRbVqf4eQFslodTJ1PuuUX4/cc/vt11CEtmYLm9WS0uaqp1srGcX5ZDheEopCIxYmFI/iCAYQiCNfUYxpGhyO7qTClRFNUAh4vHk1DdTS7djKbs5Lk0zwEPD7qohHmrVnKQx++zJrdW5NSiObgBEvGjh3LX/7yl6QFZgb9lePc0AN4Eov8QsdcIXeRBh2V8ELTSO+NWNmSmS4lKFJKKYSQH763wPuHr/+Qlx54nEQsTjDHsgI6pAicC8AiPTFdx6t6mDNqJuN7DEom5ezfE0gpk5FZ0zS59NJLOemkkzKSKDjbvP7668ycOZOFCxeiaVpS8mAYBrquo+s6iUQCXdcJaB4MmywHskL4s0JoXg+KquL1"
        "+whkWRV40o6o0qradsEtXyWUl9NYIaq5TW0iv3trBR8+/4aVAX0IbcxKQIkz7aKzCWZnJW3Qmj22fV4NNXV8/MKbln/pYdTyOkmJQ0p7cvmYUwh5/Zako6QbumGccKOAEIKYnqBnUSmju/XPKHLkfF6YlUNJTr4lCznGJgptriAnsFYsMoD1bB7ESbkArIimoioIRbGkI9KaxB8LbSyZkNZCUpqk0blBVRSCHj9VDXU8uXge76z5NJlR0Ny3nWDJVVddxQ9/+EN0XW9LEpsOTAb+SKNzg4tjCB2Z8EJjl6cAN2BpejORNwAIKaVQ7JnuvEef5U9f/zEf/u91kBDICgGWwN/tWNsORQgiiRidsnL58rhT6V/SrdkqYYZhJG3GVFVFVVUURWHHjh1MnDgx4+M5pPfzzz9n2rRp/N///R8bNmxI7lfTNDRNw+PxoGkaZcWleLxewnX1LF+wiBf+/E/+cvNt/P7a7/Ovn97Lp/MWoPm8VtS2BdIrFEEiEqO4exdO/8rFyUhqS7AkFIIFz73Cvl2VaN6DLztsWQzFKOlRzoRzTmmdRDmODS/Po3L7LjTv4XFsUIRCOB5lRJc+zB4xDYG1lK8oCmN7DaIoNx9dmkizbRZCxzKkHZU6dcAY1MwGT8CKZgkE/YrLkxrFYwHOMx5JxC3v10zt+CSWx7TzSxoYuo40DTdx7SAghCCWiEcM0zCSzgbH6XW0yLGJpqp4VJV5a5bywhcfWKt7LcCJ9N51113MmTOnrUlsOla9gBtwnRuOORwrj4Ej5lKAf9PGjEnnoXcibuX9ezP94nMYPHk0vmCQWCSCkbCWeo6FGfDRhJVcAJFEjIGdu3Pu0EmEbPmAM/CZtnwgNXIbi8X48MMP+fDDD3nttddYtmwZe/fubfPxUxMNcnNzGTJkCDNmzKBfv374/Vaksaa6hgWfLOK9jz+gatsOavdWN7uvMafPYPb/+xqQvqa7lBKPz8uD372LtUu+SJtE5pzfrGvmcPpXLqahtg6l9Qo/LR5X0zTq9lXzh+t/QH11bdqsZOe8Tr5iNmdefznhmnoUtf3mtFZkN8bI8j6cO2RSk/u7b98+9lVV8e7GL9gYrSIvL494LIYe1zv0EuqhQhGCcCLG5F5DOG3gOMxMSkPbSFbMi9Tzt/df6PBRXkUITCmJ6Qk8qkrvTuVM6jGY9zd8ztrd2/F5Wp5kOUlrn739Af/66b1pnyGwJA23/vN35HfuZBVH6MDXpaPBJrwNqqJ6NVX1nCgTTwEIodAQjzCu+0DOHjoxWdJ6fzjXpKGhgZNOOolFixZllB9CMsOCOJZP78e4SWzHDI4Viw2n/LCBlcgWA66mUTyetjd0DKutma5g2+r1PHbn7ynv34sJ55zC0GkTyC6wB+hY3GLWbYjUnChQhCBhGkgpmdF3BDP7j0oSMEU0FpNwrp2Ukvfee48XX3yR//73v6xdu/aQz8FJZFNVlZqaGhYsWMCCBQvSfsfyobR8dklpB4tffYfs/FzOvPHLROvDLd9z+zunf/USNn6xCiPR8gKDxNr2o+dfZ9xZJxHKzcbUjYOynRJCkIgnKCgtYewZM3j78f/ZusYWBjA7yvvR828y7syTyO1UgJ5oHxKVjOx27ct5QyejKArhcJh///vf/Oc//2HVqlWEG8KgCLSQn0ETRzHt4nPIKcwnEe14STLtASEEccOgOCufaX2G2/e+bd+XSPICWQzs3J3FW1YR8Pg6nFuDQ3QjiRg+zcPQsl6M6z6AbgUlAKzdk8+a3duS2T0twTRNfAE/QKuJlYZukIjGXKJ7EJBS4vf6Qm0t1HCsw+rerQqHCzevIjcQsp7LZlbHnABYVlYWTzzxBFOmTGHnzp2ZODc4zdwP/AuYQObVYV0cZRxLrM6ksbFdA/yGxuonGTU0x4/QieRuW72Bp3/zN+678Ye8/vBT7N2+C18wQCArlHwgjlTGe0eGwK5ElogT8vq5eNRMTrLJrukkhdkRXUVR2LRpE/feey+jRo1i+vTp3HPPPUmy62hvD2Ugk1Ki63qSXDtShlRpg6qqyTKaUkpMozFxzbqvEqEIPn7xLap3VjaW823u71cUYpEoPYYOYOwZ6ZPIHK1t7d5qvpj/Md7AodlNOWby48462dLytmJ1Zfny1rLg2Vfx+A6t3LGDpGa3S88k2V36yVKmTJnCDTfcwNtvv82OHTuoqa2hprqavdt38t7TL/P37/ycfbv2HDZ5RceA5LRBYwl4fLaZRhvbtX1ZJvYaTMDjw+gg18nxOZVSEo7HEAhGlPflqvGn86WR0+lWUIIhreepLK9TBvuzng1fMIjqseMsrfQB0XAkuaLkom0wU5J0TzSYUhLweHl7zSes27OtRfcGp+Jn7969eeqppwgErFy0NiSx9QP+gNVC3ZnZMYBjifBCU03vd4GbsKqhOA0ws53Y+kKhWBVoKrfv5NV/PMl9N/6Ax+78Pcve/Rhd1wlmZ+ELWg+BadperCdYH+JEd8KJOP2Ku3L1hDPoX9IN0zQtwb8dbRVCMH/+fK688kpGjhzJrbfeyqeffgo0JbmOu0J7dMYOeXWS1VKT1wzDaOLMcMB3sUhvNBxhb8UuNE1LK2sQCBLxOFO+dBa+YMAisS1Zhdk/F7/6dvrIcQZwyoUWd+vCyFOnJtttS3AI8cKX32LH+s2HVgiDxhLRA0q6ccGwqaiKwtp16zjzjDP55JNPkhMN5/46L82jsXvLdt781zMWwTnOBl9FKJbndPcB9O1U3iYpQyqcwbgolMvEnkOIJeLJzPSjAef+6aZBOB7Dq3kY12MgV088gwuGT6VLXqfkipmCtW1ZbiHZ/oBVQKPlHSPtCK/HayWutfhX2n9/Q02t7YxyfLUdF4cfTjt+dcXCpGVZc3CS2KZOncr999/fanXL1K9iySqvAC7H4h+unreD41gjvNCooVGBPwNnAZtp1NFk3DtKUyYraSmqVdb103kf8MiP7+GP1/+A/933MOuWLkNKSTA7i0AohFAF0rQjhsfZIJ6KZFRXT6AqKmcMGsdlY0+hIJhNws5s1TSNaDTKk08+yUknncSMGTP497//TXV19WEjue0NIcDr99pJDi2fn1AEiWiczj26Mvq0aWDLOJqD06a2r93Ehs9W4A0cWqRVCNATOmNOn47m8VgTrzS+vEIIog0RFr/2DprPd9DHdhITexR05sLh0/CoVmTu9ttuY9euXXg8nuREw7m/zsvQLXnLhk9XEAtH0lq6HWuwpAwJSnMLmdl3pH3ND2mHSCmZ3GsIvYpKCSfimSeCtQOcaC5ATE8Q0+PkBbM4uf8orp10NmcPmUhpTkETaZjzkkBeIIuy3CISZnr5jGma+LNDrTo1OHuordyHogiX7rpoM6SUeFUPu+uq+XDj8uR7zUHTNHRd5+qrr+YHP/hBg22FmX5AsODMxu4FutAovXTRQXGs3hyJRW414A1gCvAijRKHNgnInSVvZ4lcCEHlth28+9SLPPDtO/nT13/EC3/+Jys+WEwsHMUX8hPKzU523NI07SXz46NrVoTAkCaRRIzeRWVcNf40JvYcnBT0ezSNyspK/vCHPzB69GguvfRS3n77bYBktLcjk1wLjYN25fZdeP0+QKSNRAoBRiLBuDNPwuP1ppUqOAP/svcWWvZph3KmikIiGqVL3570HTPMshjKwBN4+fuLaKiuQdXaHmFVbKut0pxCLho1A5/Hi8RKTnv55ZeT9zgdLAmRTGvndqzB0ayrisLZQyYSsK/LoTgsON/UVJULhk+jMJRDVD/8pFdJjeYmYkgp6VVYynnDpnDt5LOZ3ncEecGsA4huKpzne0hZr3T1UZISsWB2CK+t420JzjFqK/chVPW4Wx1wcWQgpcSneViyZQ3Vkfq0Hr2applSSn7xi1/s69mz539N01SEEJkQXhMoAX6NK23o8DhWktZagmMLsg04B/gm8DMgF4v0CtpA6lNF/k6ik2mY7Ny4lZ0bt/LOk8+TV1xIr+GD6Dls"
        "IF0H9Ka4axm+UBBFU9FjCfREIqkRBY4p54ekfCEeIz+UzZReQxlV3jeZja+qKuvXr+fhhx/m4YcfpqKiwvpeSpJaBlmuHQO2x6Oe0HnyF39i387dnHT5BSRisRbHV6EoxKNxuvTpSd/RQ1nx4ZIWkxycjnXN4s+prazCFwoemp7W9tEcd8YMVn60JO1EQpomCMGerRVs+mIVgyaNIdIQznjJ3Ypg6uQFspgzegZZvkDSFq6iooLa2trWCyvYUcvczoV4fD4SicRxMRIIO+o9a+BYuuYXH7SUobn9SinJDYS4dMwpPLX0bXbVVRH0+OxknEMnfQKBI4k1TZOInkARgvxgNn07dWFoWS/K84uT25t25Dr9fbZ+9isupzgrj6pwXYslpqVp4vF6CeVms3f7zmQZ4ZZ2Wlu1zyW7Lg4aEivXoi4a5tNta5nRd6ST19scFCmlKYQo37Bhw01CiLCmaZfput6aVMFZWb4M+CfwOq5rQ4fFsU54odGpQQL3AfOwZltn7vd5m0YlKSXSsEIWAkvra5oG1bv3svSN91j6xnt4fF4Ky0roOWwgXfr2pLR3dzp1LcMXtHRqUkr0RAIjoTcmwAl7+dB+6hqLNFj5eM0NbsLa8LBFSxuzsOP4NS8TegxiYs9B5IdyktssXbqUv/3tbzz++OPU1dUBjaWC26uIR6oONBWpS+Xtfg2E5cX8yt+foH5fLWff+GWbnLWQGIZEqApDZ0xgxYdLWozcOslr+3buYduaDQycMIpoQ/igl/atxLkYfcYMpVN5GXu2VljOEy2sKiiKwDQkn73zEYMmj838OEKgGwYBj4+LR59EQTCnSZJeKBRCVdVW77lzVqeccyaKpkIsBse4rMHR7Q4p68XknkPbjew6cCJQnbJyuWLcLF5e/hErdm7Co6hoqpaMLmf6BKQWgDGlRDcNEoaOqiiEfAEGdO7OgJJu9C4qw+9plBk4xWMy+dsEVt/h17yM6dafl1Z8hEdV00waBdkFefZ3W1gztr9cW7mPeCR2XMlhXBx5aKrKyp1bmNF3ZHKsc35C43OiKIrTHK9RhXK+oevdhRCTpZStkV7nQfk18A5WXlFrpiUujgKOB8ILjS4NKrAcS9d7GfATYID92UERX6Rd2cWOXKYSskQsnoz+gmWqnl2YR1nv7pT27k5xty4Ud+9CYVkJvmAAzevB1A0SsbiVYGXoVkUZSJIKRSj2UiPJU5VSYpgGXs3T5EE9FNgkWxrSkOF4Ap+mySGdeygTew0WTpRHAq+99hp/e+ABXnrpJeLxuPV3alqyoMShQFGUZIQ0mRSYARyT8FTd6EHDnvELReW9p1+ia/9ejD59BuG6+mZlA0II9HiCXsMGEsrLoaG6tsVIlRO127R8DYMmjTm09S4Bpm4Qyslm6NRxzHv8v2mX6BwivG7pMmr37iMQCiZt41o+hLXsrCkqXxo5nc45BcnBwbnGpaWl9OzZk7Vr17aY/WyfLhIo8+TQo6iUT6tXEPIHO7DEJT0siUeckpx8zh4yASHkYRnOBNY1zfEHuWT0SXy2fT0fbFjG7rp9tsG+hioUa0K2Xz0Baf/jVKLSTcOqfCcEHlWlIJhN98ISehSW0r2gM9m+QPK7VhEMMia6Tc5ZWH3UyK79+Gz7OnbU7sOraQfeaykRikJep4K0+3PadPWuSmLhCB6fNxkscOGiLZBS4lFU9tbX8s7aT5jWe3iyX09t5xJJNBFXI/GYiSLO21Zf9bfpV8z52ppnX39JCNFLSplOn+skzQ8HrgL+jhvl7ZA4Xgivg9Ro7+PAC1hODjcDnVO2aZPUIRX7yx7AilogQU8k2LdzD/t27mH5gsWAVUozuyCP8n69Ke/Xk9K+PSjr3Z2cgnyKs7IJeXz4hUZI8xP0eMn2Bwn6/HhVD6rtuRrTE6yvrGDJltVE9USS+ErZfPnE/c5XCiGkfc7SZiKKIU1hgsgLZYkJXXoxrtdgcoQV5dlTWclzzz7LI488wocffpjclxPZa0272RJSPXqdRCeH5CqKQvfu3RkwYAA9e/akpKSEmpoa9uzZw44dO9i8eTNbt24lGo0ecPz9iXNbIVMG+7VLlzFq1rS0WkQ9niC/cyfKendn7ZIvWp2EbPpiFYauH3olLSHQdYNBk8cw/6kXMdLcB6eN1uzZS8XajfQfNwIjrLdIGqxmYWJKyewRU+hZWNqkmIij2fX5fFx00UXcdddd6XXEdpTw13fdzZ/79qJ7365s21GB33dorhFHA5bO1cTv8XLBiKkEvf5WJR2HejxncjS8S28GlXZnza6trNm9ja37dlMXi6CbOsZ+GnkhBKqioAoFTdXoFMiiU3YeJdn5dM0vpjg7H5/mSW7vRIsPhuQ2OV8EJhKvpnHKgDE8uugNmpsNSFuWk1dc5Jxw8zu0H+F9uyuJhsP4gn500zwuJDEujjwsX33BO2s/ZUPlDmYNHEvnnAI2Ve1kR81edtZWsbe+hnAihmGaipTSzM7Ovnb2tRcP/7PY93Lds4uuEohsKaS0R9B0h/oGlrQhjhvl7XA43ggvNI321gF3YxlEfwP4KlCcsp3j9nBQcAYbadht2l4acQobSGmSiMWp2rGbqh27+Xy+RR6z8nIYOGwIJ0+fydRJk5k8YSK5eXktHAQQ0Le4nIGdu/PhhmWsq6wgmoijqSqqUKRAgIKJ2WjbZg/GQtNUoaiqUFUVFIuYmbpBridA3+IuDWN7D9qSpfo3Aw1vv/32mBdeeKH7E088IXfu3GkrKUSSTB5sRNeRKqRGhYUQDBkyhPHjxzN+/HimTJlCeXk5WVlZze6joaGB2tpaVqxYwfvvv8/atWtZsGAB27dvJ2Hrph04muM2Jc3Zg380HLFdENJbf2kejfJ+vVi75IuWt7NvR9WO3dRV1RDMCSXdCw4GiiKIR6N06deL8n692LxijSVdaFHWYN23NYu/YNDkMUli3xLihsG5QycxsHP3JmQ3dX8AX/va1/jjH/9IfX19i1HepFtEPMaPv/Vdfvuvv5GXn0ddTS2eNBW5OhqSMgIpOX/YVEpzCttdytDSccGKvHoUjcGlPRlc2pOYHqc60sDehhrqYxFiegLTtJLovJqHkNdHjj9EXiCLgMeHtl+Vv8Zl3KaSh0OFswrQq6iM6X2G8+bqJYS8gQNKvAoEOUV2hLdFy0Dr/XgkSu2eKvKLO4FMuBFeF4cEn+Zly77dPLrwdUJeP9XRBuJ6AlUo9uqqcIISSmX1XqMokD3uG9fekP9QQry++/mFpwGhNLt3Vo8LgCAW4XXRwXA8El4HqZHc7cAPgT9hFa24Ess02oFub3doYjF5oAY3GQW2SbBpGNRX17Lo3Q9Y9O4H/BIoLCxk2PDhTJkyhbFjxjBu3DhKSkrsHTTuvntBCd3yOrFp7y5W7NzEpn07qY2GhRSgaJrq9Xosv9hEAsM0SCQS4qKxXQAA1xdJREFU1NTXR6KJWFVcNyqFKlZm+0NfTB814dMzyod+BsTOPu207q/Mm3eKqetnYj2sEhCp+tyDIbpONTTHrQHA6/UyefJkZs2axZlnnkn//v3x+Xz7XcLGxDdH5qGqKqFQiFAoRGlpKSeffDIA9fX1VFZW8tFHH/Hee++xYMECvvjiiybnmzH5FQJMk4LOnVA1NX0Ez9b9durexTrPlvZpf9BQXUv9vmqy83Mx9ENb5ZJS4vX7GDRpFJtXrLFJQHryuPHzlYRrG1BacGsQwioqMmvgWEZ17WcTugMfBceovUePHlx33XXce++9SUufls5VURR27N7FD667mZ//44+YOdk01Nbh0Tytrk50FMT0BOcMnUT/kq5HhOymwjmWIznwaV5Ksr2UZOdn9P1kFBca8wcOE5zJz/S+I9gXrmPJ1rWEfP7GyaiwVnayC/KSrg2t7Wv31gp6jRgEMXFYo+oujn84rg2mlFRHGtAUBY830NgPpejjVVQ1GomaRb6svt/4f7f4H9z7641bF3w21F4tba4ROo9ZBEvD66ID4ngm"
        "vNBoX+YQ3wrgLuB3wIVYEd/pNF4HSaOXXrv0rMko8H4k2Imc6rrO3r17eXvePN6eNw+AvLw8hg0bxqRJkxgyZAjDhg2jb9+++P1+hKLQs1MpPTuVAiSqYvWxDXsq9qzbvX3Tturd23fW7K3cE66prWyo3bUrXFMR8asVW8YWbqgfc0elc/xfwQisEs1nAqOALCd65wxE7RHNdYjQuHHjmD17NrNnz6Zv375NtjcMI0mMnMHM0ejufw2hkbgKIcjKyiIrK4sePXpwySWXEI/HWbNmDa+99hrz5s1j/vz5NDQ0NF53RbE6tf1In2JLRxRVYdj0Ca2SUodi5hYVOifY7HbOecajMer21dJFbYVIZwhTN+g5bCCKpmKmuU8OodixYQu1e6soKC1GjyeaHF8RgnAixrTew5nca0irhM4hIt/97nd59NFH2b17d9pynKZpWtX3tmzm9hu+zbfvvRMKs2moqTvgPndERPUEswaOZUy3/s1GvY8UnHtiCZNSpwqpba8x/TWZbNuOUdyMYJ/n2UMnoZsmn21fR9Drtz8SmLpOTlEBvmDASuJsYYVAKArSMKjcuhOEkqiPhutyQ9kF7ZUg6+LEhNPWHCcRKdNOupRwJCJzcnO6FgVzyrda74lWVqd2AtH2PGcX7YeOP+K0D1KJrwqEgX/br0nAHOA8oAeNEgeTRvLbrqOcs0SaWtUllShWV1fz7rvv8u677wLg8/koLi5mwIAB9OvXjx49esRDodBCI2G8HatrWNAvq3TlJadcuVOcLPZfRskCugPTFJQhKIxFiLFIWeKchw3DNM2D1jWnyh6cAamkpIQLL7yQyy67jPHjxzchN7pduMKJAmeyfwep2+/v3uD1ehkyZAhDhgzhO9/5Dlu2bOGVV17h1Vdf5fU33yBc30h+HYJvysZzPv3aS+k2uB+xhgyqo0mJx+dJvw2NBDEeDrfLFEpRFOKxGGV9e1JU1pndW7anTR4D0ONxdqzfTHHXMvRYPElKFKEQjkcZ32MgpwwYbZH+Vsi4E+UtKSnh7rvv5itf+Uqrrg22kTvr1q/nZzfcyjfv+hE5A7pQX1PbrCvH0YbASgaM6QlOHTgmZSLQ+uOx/wQ3E+lG6jVo7Vo4kVrR9J0OA+dsNEVl9oipBL0+Pt60Eo+qoSkKhm6Q26mAYE4W0YZwmj1Z123nxi36Z1vXzC8P5I+0254UHa3BuDjmkOnqksfrEXt37pbrFn8hIO3z7HCFVVhcw4Mb6e1wOFEIrwOJJV9wyJ0BfGC/fgycDswGTgE60ZQApn6v3Tpc5wFKjaimRoBN0yQWi7F161a2bt3KG2+8AdZ9GwkMxfIe1uGqmFBETAjFQEpNQgBkNpKQxCJ2NOUkjozDmQS0Gc45pmpzJ0+ezBVXXMGFF15IUVFR48FSSG57Rfb2J0upEwlVVenWrRvXX389119/PfM/W8g9D97PkhfnsXPT1iYErai8lJmXnc/YM2YQD0cztEGyktcyRSzSfpN+0zQJZIXoM2qwRXjtaFhzUFQF0zDZumodo5zSxNhkNxFldLd+nDV4ou1PmVmzdu75Nddcw/PPP89///tfVFVNuypgGIYlb9i5g7u+8V2+9M2vMvycmcTicfRYzJpgdAAeI4RVdMU0TE4fNJ6JPQcdEPVuzibPSZzMlLimw/6rHsciv7OS4RTOHDyB0pxC3lq9hLpYBK/qwR/wk1dcRNWO3S06nJimKQVCrFzy2Z7uu2Y0DB3eqzAaiUhFSVNX24WLdoRpmviDAVYsXyLq9lW3FlhQAHzDumWV3Da7dMuFv9/Bbbcp3HGHuyTRgXCiEV4HqRFfh8TWA0/br0LgJOA0LPLbnabXKjX66+yj/U4uhbhBI7FLeSlSytD+g640JbJ5JxTnfFOjuId87x29rd/v59xzz+W6665j5syZTZwYnHM+EsvXqROF1PNTVJVpw8ex4ZpqRp86lYpVG9i5aRsg6dS1jO5D+pGVl5tcYm0N1mAO1Xv22gem9Vzc9mwh0kpg6zaoHx/89/WMzPl3rN9CLBxNJmeE41GGl/fhnKGTaFaRlgbOdZZS8te//pVPPvmEzZs3t0p6HXlDQzTCP+/5E+M/+ZxTvnoxBV07E2uIYOgGiiKOGvFVhEJMT+BVVc4fMY0hZT0xTMNyYLGlKE4lwZbaSSKRIBKJUFVVRVVVFeFwmHA4TDweR0qJqqoEAoHktl6vl5ycHIqLi+nUqRM+n++AVQ/nmqYS6sMJZ6XGuceZrsSkwjlLy66sL90KSpi/9lOWV2wkZugUdilhw2cr0njx2rtIGPk9AgUTRAY815FvtIdtowsXYLXNdUuXAZYbUzJB/UAoCGH6BnUNhz/b+rX8753z7L477ljObSjcgUt6OwhOVMLrQNKo23XIoAT2AnPtVxYwHkvrOwWYAARoGv1tjlC2axS4hZnl/m8eKOiz0K6yDGcg7NevHxdccAFXX301AwYMSH6u6zqqqrZ5kGxvOETblBIhYVB+OZt2VjBwwkgGTR5jJdHYvsiR+obWZQwOJCiqyrZV663j0LIfroNAVroE37ZBKIJEXKdzz254fF4SjkyhObcE28Fh7/adRBvC+IMB6qMRRnXtw7lDpyAc27E2noOz+lBSUsLjjz/OKaecQjQabVVekSrj+fjNd1n1yRfMuOx8xpw+nWBuNrFwBCNhT5SOcDCvPhqmUyiXc4dNpmenMgBU5cA2HI1G2bFjB2vWrGHDhg1s376drVu3sn37drZs2UJlZSWxWIx4PJ6RhZ+mafh8PvLz8+nXrx/Dhg1j6NChTJw4kb59+zaZLDpk1CHe7QknCu+8UuFMVtoKpz0UhnKYPWIao7r2ZXnVNj4q75L8vCVIy+vYn9hX77dWJmzrmSYexNazZ0qJblhWbanWay5cHCwURSEWibDx81UALRb5wY6BKCHfHk9p3i4ZTfhVKS/P+/5Z/6m+46XPXNLbcXCiE95UOFFfaEpc64G37BdY0d6pwFhgHJbZ9P4EGCzJABymKHDKeab7/bDAGcQuvvhi7rzzTgDi8TiqqqIoSodLRnJ404iufVm8dTWVdTVodqLa/lHhjPanqtRX17JuqWVJlo7gOZ8FQqGMIrGZwPICjtOpvJT8kk6WrKEF0u0cf9/uSiJ1DeDTGNO1H+cOn2x/5+AbjSNtmDRpEg8//DCXXHJJk+h6S0jKAFSFmr37+N99D7Pw5XlMvuB0Bk8dS25RAXo8QSIWxzTsVQ5FJInOQSHVszo5gbTuuxQSKQRTB43ktD6jk1+JxWLs2LGDlStXsmrVKtasWcPq1atZt24dlZWVRCKRjA+/v/QmFbquo+s6DQ0NbNu2jXl28qrP56N3796cfPLJzJgxgxkzZlBQUHDgCsYhSh+cCbUzQV22bBmvv/46W7Zswev1cvbZZzNt2rTktm09TrJAioQehaX0KCyl5qSzePFvj6b9niIUTGlSsWELA6eMxTAN+z17BUxavtGaouHVNDrlFFAYymHN7m1JVwsXLg4GUko8Xg97tu5g3+7K9BsLIZFSiMKsrWpeMG5UN/ikR5GaqV5c8J1zE1V3PL/ClTd0DHQsZtJx0Bz5FVgkdrP9etR+rxtW4tsILF3tCCz97/7Xdn8ZxeEiwYcdTrTnZz/7GRUVFfzmN78hz/YRdqK7HUt3KJKWNFN7D+eZT+dbiVYHQUANwyArN5uPX3yL3VsqLP1sCwlbzsQglJdDTqcC9ITebtdFmia+YICyvj3sxLX0fNpI6Ozcuo0vT5jCyb1GwCGSXQeO9dzFF1/Mvn37uPHGG5MErLWMetMwk3Z9O9Zv5unfPMC8x//LsOkTGDx5LKW9uxHMsXyZ9bhVnVCaEnmAxVzqX9H0fSdSLBSBoqgoqoKiqmh2JbB4NIZHCgbnl5O9K8rf3/57kuCuXLmSXbt2pSW2ziSvOV1vKpqzKmwOzrWTUhKLxVixYgUrVqzgvvvuo1OnTpx88snMmjWLU089lfLy8gOSQVMnb5m0"
        "NcMwks/rO++8wz333MObb76ZrKoIcM8993DLLbfw61//Oumh3GbSa09WdENHVVT69eqTtLRrcVXAPsberTvwKhq65sGjeQh6/QS9PnICIQqDORSEcijJzic/mI2qKDz04cts3bcbr3rsWN+56GCQEtXjoWrXHmINYbt/baktWaKwrJ6dd4qcQExW1QeEKXUpUIViXpJ/89kP7rvjji0u6T36cAlv69if/EJjklcqAX7Cfi8PK+o7GhiMRYAHYJlRHzck2Fma/sc//sHHH3/MnXfeyfnnn58cgJ3oU1sH4MMFZ1AdUtaDT7atZePeHfhtT8ZMYRomvoCfqh17ePUf/2lRRrD/MfNLOpFXXIieSLQf4bVlFcXdy52DtbitIz/oJnI4pffIpK65ve6GU2r6hhtuQErJ17/+9SbHTf93SHAIlBBUVezinSf+x/z/vEBpr270HDaQbgP7UNq7O7mdCvF4PXiDATSPB4RV3KWR00jLwcB+GYaBkdAxEjqJeNyyh6uqpmZ3FVU7d7O3YifVO/YQ3VtL9e5K9lbva/YcUzWsqdp64KBKbKeLfreUvGoYBnv27OHJJ5/kySefJDs7mylTpnD66adz2mmn0a9fvwNWVhw5Rar+f/9zUFWV3bt3c9ttt/HXv/41+bmzL+c6/uEPf2DXrl08+uijSYJ/MG1ZVSxy3bV7N/Ly8qisbDl6Juwbq9bFuXLEyaiKgqaoeFQNtZkVGeeyDurcg017dyK09PIaFy5agpRW/xWprQdAKGqLicGO3jyxbOs01efRlYm9P5UxXSWaMKVHeBWvuKTwu+f+Ze8dd9QhkwtrLo4CXMLbNjgN1ZEr7E9UdaAamG+/wCLHZVjEdxiWs8JIoJzmSfBh1QO3J5xl0GXLljF79mwmTZrE5ZdfzoUXXkhJSUmzfrrNFYDYf+Bs6+9tgSIUTh0wmkc+ejW57Nla7yOlxDRM/KEAsXCEx++6j5o9e1vVqzroM2owXr+fcF19O5J+iRBQVFaSPMeW4BBPLapbgnVbA9qecCK9N954Izk5OXzta18jGo2mLUyRilTiKxSBaZhUrNtExbpNLAA0j0Z2YT4FnTuRW1xIVm4OgewsvH4fqseT1PuaCZ1YNEY8GiVS30Cktp76fbXU7NlL3b4aErFYWp9lJ9q5f8T2YMtpHwqaS151Jo91dXW88sorvPLKK3i9XkaNGsVJJ53EtGnTGDt2LAUFBWmlRU47fOqpp/jBD37Ahg0bmjjDpP69Qgg8Hg9PPvkk559/PhdffHEyMtxW2JF/WVJcbJaVlamVlZUtPkdOBcH1q9eixA2ycpvq4K2iGtY6RapN29Cynny4cTkN8QiqUFx24aLtsCO6Fes2Z/yVcGVNTuS1T873rd7WN3j2iFeUgtx66iIID8UmzEbyL26/TcAdbpM8SuiQROoYR6oEIjU6nAov0BkYgkV+HSlEF8DfzPZHQg9sHSBlefZgv5Ofn8/UqVOZPn06o0aNYuDAgeTn5+P1etvlHB3bpv0jV5mQScdiav7aT3lr9VKCXv8B5U8dSNMyJlc1DX9WkIp1m5j767+wefnatFIG62RIGv9f//vb6DVsILFwtN0SsayKa342LV/NA9+6PVmyuLn75pDO666/ngf++ld0XT9sOmtn3/Pnz+eKK65g69atrbo3tAShCIRQmpEwHDqSNmhSJos5WD+OjbEolfzuf20LCgoYMWIEw4cPp3///vTq1Yvy8nLy8vLw+/1IKVmzZg2/+MUveOGFFwBavUdO5cUrrriCRx555GAJr9MfagCzZ8/Wn3vuOS2T9rFk6VJGjhyJbCV5zukX5q/7jLdWLSHo9bmuDS7aBCeQE6lv4A/X/yDz4Ial5QUQasi/N3TasOe9o3ttkJV1fjQlYEqe3febFxa4SWxHD26Et/2xP8l1GI5DVk2sOttb7NfL9uderIS40ViR4GFYJLiUlqUQqfs/JEmEQxxTl2szjWA633H0h/v27eP555/n+eefByA7O5tu3brRtWtXysrKKCwsJD8/n9zcXEKhEMFgEL/fj8/na/IKBoMEAgFCoVDyPY/H0+JA61SIS0eEhb3dpF5D2Lx3F+v2biegea1B0SHt9hK25vfi8XqoraxiwXOv8c4T/6Whpq51soudcGOa9B4xmO6D+hGLRNs/q17XyS3Kxx8K0VBT2+J2zj2s3LMH4LC6ZzjyhunTp/POO+9w3XXX8dZbbzVZms8UTWz2hKMDbczVtzaS+6l2m/wD2GOQQ2jt5KljvWKXIxmCA6UPVVVVzJs3L5n8BtZ98Xg8SbLoVCBMlX9kcszs7Ow2nab907nYKqDpul6nadqdixYtKhNC/D/TNA1a8AJ3Is5ffP45o0aObJ28Cmu9eEKPgSyr2Mjehho8qnbMTGRcHH1IKfEG/Cx7f5FFdjPo7+0vOsYhptEQLax/fskVgYrq10JnjfjIqIkKBTmr9LazV++448VKV9pwdOAS3sMPp1GnI8EGFglea7+etD8PYkWBx2CR34lAL5qXQhw0nAiLlJKJEyfygx/8gKVLl3L77bdnvCQNHLD0KoRA13Xq6upYvnw5y5cvz/icHGskVVXxeDzk5ORQWFhI586d6dq1Kz169EjaNxUXFxMKhZq1U2oOQggUFC4YOY2HP36F+lgEr8eLsB0mDEMnXNfAjg2bWfbeQj6f/zF7t++0vptJ55fCa6dedDaaRyMRj7erzZYzOQlmZ+Hxe6Gm9e9UVlYmI2DtUeK4JTjtqVevXrz22mvccccd3HXXXcmoYHOyllbhuCzI9GlIJ+II0pL0wbnPTpnv/Z/j1Oc+Hex2IqWUzJ49O+2p2D+dB8TpoxwyWwE8EQ6H/56bm7sauHi/7x2AJOH9onVHFLCX1aTEp3k5dcAYnlj8ZtrtXbjYH1avKNm5cYv1u2hjvyJREEhTN7SGj9acZeytL8mePe4VqeGL1cbOBP7V7iftIiO4hPfooCUSnKrbNbBKIC+0X2ANHN2wor/9sCLCZViFMnz2Z6X2/ltlM6nRnW7dunH77bdz9dVXI4TgrLPO4rPPPuO5557D4/GQSGReVSw1+pR6nNTXAUUz9pNROH6juq4Ti8Wor6+noqIiOfA58Hg8FBcX07NnT3r27EmPHj0oKyujtLSUkpISSktLycrKwuPxoGlaMjtfNwy8isKMrkN4atFb1NfUsW/Hbvbu2MWuTVvZsnIde7ZUJM9JKIpFtjKY6SuKVd1s+EmTGDRpFNFw5KA8TFuFKfH6fQSzQ1Tvqmwxic75G+rr69F1HY/n8PuUOsRWURTuvPNOTj31VL773e/y8ccfA42R4OMl8tZkFcEmiPav0gEHjptiv/+L/d4/qBnJ/s/f/ufnINNouxDCNAxDmTNnDjNnzmxiYbb/pvbP1Ma+B3gbeAV4XghRZZ+j6vF4tum67kgcmu2znPaxapXlhepIK9JN1pI+4cXlTO09jLfXfkrI5z/oqL5s8tzbrh9HMAE3U4cPF+0HKSXxlDLsbd9BUtIoo2srxpgPvV0cPGfUs57uhX0Kb5o1YK94fZUrbTjycAlvx4EzIKY+AKkk2MQiwRvt1/7IBz5O2VfaJ9XpOE3T5PLLL+eee+6htLQUaLQW++c//0ldXR1vvvnmIRGUtmqCW+rU95cqGIZBIpFg+/btbN++nffff/+A73g8HrxeL6FQiKysLILBYDLiJaUk3NBAZVUViUSiWTKrqErSCiujc7fJbl5JEWffeAXGYUh2cipKGdIkbugEcjJbZo7FYhiGcUQILzRquw3DYOrUqcyfP5/77ruPe++9l507rYj5sUZ8U2UyKZM3aZNHCZj236Jy6Hp7w97nIWv3D/b6KopimKapDhkyhD//+c+6aZoqIOzr4CTYxoEYUAdsAtYDK4ClwKcOybXPQ507dy5CCKcvqwEKaKHPckjqypUrCYfDBIPBzP4WITClZEbfEVSF6/h8+zpCXj9GG66Dcxyvz4fms+Qghm4Qj0bRE7pV"
        "bvowElBpmiAEmseTXNEx7H7rsEygXQCNDg35xUV2AOGg77H1zArM+J6abvrjC67JnjX81cCkfuOBddyOwR3tdtouMoBLeDs29ifB+7tCOE+jDtwC9MUaJNOKNJ1lQlVV+f3vf89NN90ENHpyapqGaZpkZ2fzzDPPcPXVV/Pcc88lJQMHk4DUFrQ0oDX3vkNAUsuuprpBJBIJEokEDQ0N7N69O/2BhUgOYk6ilFP4IBM419UfDHL5T79FbqdCYmmiu4oQGNJEYB/XvqXNhwHtilKmiW4Y6KaBV9UY0LkHfcq7s37pMmt/ac7PIbxHEo6ll2EY+Hw+br31VubMmcOvf/1rHnzwQWKxGECyzbV1cnS4kEpsnftnrzrYOW4y9bnUhBDk5eWJgoICpaCggEAgQCAQqAf21NXVrduwYcPqnTt37qKxJLnEWpXJsl+dgGKsxNUC+739n2NHu3/Yk1dpJO8q8Onll19+S1FRUUMkEvEHAgFhn1sUSGAR3b1AnRDigKUgKaVT6dGwia6DHcAurL+3+ZOw28KmTZuoqKigT58+GclxHBm3EArnDZtMwtBZuXMzQa/Pdndo5Y+XEkVV8Hi9bFm5jvVLl1G3r4ai8lKGTB1HXnEhsYYIpmlYqz/tDNO0LBClaVK9ey/xaJRAVhbZhXmomka0IWz9nW7Et91hVbJM0HfMUHxBy5mnNQvKtLAkDqYZjefWvbz0S/reurySp77z7i5x70YuQmVu2m7bRTvCfVqObTgDZydgOZa0AdLcV6eDDAaDPPbYY5x33nktFotILSf6wx/+kLvvvts6aEqlp45ATtKhmaXmJkhaGx3in6GoKqZh4A8FuewntzBo0mgidQ0o6oGDoUN047pO0OvDME0iibjFYJzI4X7nZ0qJR1Hxe3wUZeXSq6iMvkVd6JLfiWu/9jUe/Mc/Wsy0d5Z4u3TpwooVK8jJyTmsGt6W4ExEnOXwzz77jPvvv5/HH3+c+vr65HaO7MSJ7h2uNrZ/20hN3ExZ/papL4/HoxUVFdG3b1+GDx9Oz549CYVCsqSkpKq4uHh1t27dlhYWFq7y+/3rgJVARSgU0tsw2fAAJUAPLP/uwVhVHQdjeXynQqedy4bbSJ00/xe4Fmil3FQjUgiuBEwhRHM30Jmw/w84lzQTdWci+eyzz3LBBRe0ySFC2tlBumHw0vKPWLJlNT7Ng6ooLSbAWWRXxdR1XvzLv1n48tuYKfcuv3MxUy48g/Fnn4wv4CfaEGn2OT9YSNPEFwqyefka3vzn02z8fCWJeAJ/KEiPof0Zd+ZJDJo8OukxfTgI94kOKz8ixNO/+RsfPv96UqZ2SLALDgohhKdz/n8Su/ZdYpcrVmnezclFO8MlvMc2NKxB71vA72gluusM6Jqm8fTTT3POOee0alGV1LAKwbvvvssPf/hDFixY0HgCHSwyd6QhUiLC+Z07cekPb6L3yMGEa+tRUgZlh8yaUhLTE/g0DyPL+zKqWz9iepzdddVUh+upi0WI6XEM03Kb8CgqIV8gWVWqc04B+cGmEoZrr72WBx98sEXC6xCG8vJyli9fftQIr4P9ie+6det46KGHeO6551i7du0Bf8P+3rjOPtqC5lw7mpFSJImtoiiysLBQ7devnxg2bBjDhg2jX79+FBUV0b1793120tVCYBnW8v1yv99f7USs9z88jVHZAy7Hfq+W0BkYD0wBTgcG0visO9Hmlo6RKZwRXQEagNuAewEuuugi9amnnpJz584VF110Ueq5N/l/C+S2OTiD/K+A72H1Y812RE7i7J133slPfvITEolEm2Q5qYvSH21czjtrPyWSiOP3eC3XltQ2YP9X9Wg8/rM/8Pn8j5qu/KToebsO6MO5N11Nr+EDidTV2w4ih/ZMOStEqxd9yqO3/y4Zyd0fI0+ZwrnfuIpgThbxWNyVOLQzTMMkmJvFopff5slf/CkZ0GgOQrGy2trQJxmAqijK03l5eV+tqqqqxSW9RwQu4T124dw7FXgfGIc1YLVIeB1CdN9993HTTTe1yY/VifaapsncuXO57777+OCDD5o85M6+9s/CPx6JsFUgQUl2goMmjeH8W64hv6QT0YZwYyUzO1qrmyYJQ8eneehbXM7kXkMoyy066OOb9sCrqmrGhLd379589tlnhEKho0p4HTjtxCG+hmGwePFi5s+fz4cffsiiRYvYuXPn4ZJhSEBqmiazs7NleXm5GDZsmDp8+HCGDRvGoEGDKC4uxufz1QDbgE+wNKmfA18IIZrVxzz11FPqRRdd5EQvzf0S2DLB/g4u+9scOp+NBc4GvoQVCXag05T4tnaTHbKcmjD7AvAj4AsaI7Xt/RA7k/WvAP8gA8J7xRVX8K9//eugPIBtHQpCCHbU7uWdNZ+yZvdWJBKv6kGx7cxMwySQHeLjF97kqV//pVmik7oS4PH5OOuGy5ny/9k77zgp6vOPv78zs3v9Do7eBWwooFJUQEVsGMWagD0x9ho1msQSBYxiVExiFBGNif7URMXeOyCISlcElN452vW2uzPz/f0x891bjrvdPe4O7o7v+/UalpudvrM7n3nm8zzP+b8gVFFZr++VlBLLsigpKOSf199Dyc4CTL9qjB8ZjIor13XpenAvLp/wRzJb52CHwjrS25BIMCyT4h35/PO6uygrKtn9XPBvhKLVUQyBH7FNBgcwhRALpJQXACvRorfR0YK3+aIS2Q4DFuFdLGr9PJUYOu+883jzzTdrtTHEo/qF5quvvuKtt97io48+YtWqVTWWL1MezngRuuYijquXWwNo07kDJ196PoNOHw4SIqEwhmkikTiug+04SCAnNZ2DO3RnQLeDokJ3l/0mxndY7WOM9RyKmPfVTcjll1/OCy+8kFDw9unThwULFkSbD+xrwauoLnwVpaWlLF26lGXLlvHTTz+xcuVKVq9eTV5eHhUVFVRWVhKJRGo871Td2WAwSEpKiszJyZEdOnSQXbp0kb169RK9e/c2VWWPnj17qtkqgU14wnYBXvT2eyHEhpq2uwZx25gnb/UKLooAcArwWzwBnBbznhLLNYlfJWJjReZXwCPAB/7fjXkBVss+CfiCKkG920mpzt8hQ4Ywc+bMpCo11EbsfCu2bWTe+p9Ym7+VikgIEy9HwbJMnrn9Adb9+HO0619NxJYoPGHMKEZdfxl2OBJtu15XVJnBt/7+L2a9+VHcqKL6rnc79ECueuRuAqkpuH4Nck3DIF2X1IwM5nz4Ja89/JQ3Ung11qX/PkD3PgdSWlhE/pbtGGadrA/qiexG4GJgJlr0Nir629F8URGSG4BJxLEzqB/BzMxMFixYQO/evfc407f642jwqjosWrSIuXPnsnDhQhYtWsS6desoLi6msrKyTstXAlkR6+Pc22K4ekJcrLBKy83hmDNO4tizT6VVh7ZUlpZ5rVAFuNIlaFpkBNPo3Koth3boTq+2nclOTff2xfcMN8TjT8MwGD16NK+//npCwdu/f3/mzp1LMBhsaMErY16Vdt+j0lrKPxvvZiwUClFQUEBBQQFlZWVUVFSosnlSSkkgEJBpaWkyLS1NtmrVSrRp08ZMS0urvphKvISpH4H5wNJwOPzjaaed9vOMGTN2U9BJ+lL3JrE1vBV9gTHACLwb4VqTwWKoxBOcTwPv++PU59eYJZPUDXtvvJuLbGqp1KBultu1a8fy5ctp1apVvSOp6skLQF7xTlbv2MLqHZspCpezZdNmHrviDirLK6pi7LUQG+096pTjGX3HNWAYuLZTt7rbUmJYJmWFJTx+7Z2UFBRFawrXhvq+Dxw5nAvvupFQRQVC6ChvQyKlJCU9lR+mf8sX//cGW1avr6rekZpC3+MGc84tV1BWWMJ/H3icjT+vTr5RhYe6bpcBVwKv+n+7NPxTlf0eLXibL0rwvghcSpxHguqH8YYbbmDSpEl72hZ0N5S4qmlZxcXFrFu3js2bN7Nlyxby8/PZtm0b27dvZ8eOHeTn57Nz506Ki4sJhUKE"
        "QiEqKiriNrmIbSyRTLQ4Wap7O2PrE8cuLzU1laFDh3LOL8/jgGP6Q2YKhcVFRMIRDMMgJRAgOzWD3PRs2mXm0CmnDZkpVUIr1g/dEKiL/mmnncZnn31Wq+BV44877ji++uqrujae2M2nWe1VZezXNm9sR8A6CeHYhgr+ZyJjbtKq+17jbUcFsA1Y6rruwvLy8h/Lysp+nDx58s/jx48P17BetaymIm7joSK/sVUjwEt8OxLP7tAPr2FNG7yOjqV4UaW5eCL3h5j59naEKSCEWCGl7EGVD7lGhBD88MMP9O3bd5eE2j3F9b8DsSdiRDos/PEHjuk/oE7LUtHYw48bzCX33uKVRauD6HVdl7TMdH6cNZfn734k+h2Nh8Avg+i6XPnI3fQ55kgqyhqp5vd+jEoirCwrZ/2yFRTkbce0LDr37kHnAw8gEgpjBixCZRW8/tgUfpj+bVXd9uSuR7Hn/R14vvnGshLt1+iyZM0TVYrMxCtFpsbViP+lC1122WWulDKNWiIpdaV6JFZdhAzDIDs7m379+tGvX7+4y4hEIhQXF1NUVMTOnTvZsWMH27dvZ/v27axatYrFixezevVq8vPzqaysjFs8PtY+keiCod6Pzcivafp27doxcOBATj31VE499dRd9yfJo+hdWBu+hJBa3na/ZXCiH9e2bduqaFT1BKPqr4rqFQBq2wEXryzVGrxIxYF4daEzqf03RtVwrXXZNYhckziC2bbtMFCAJ+aWSCmXGYax2nGcpStWrFjet2/f2sRtbNtv6R+Xhi+g3DjE3lTERn23Ap/4QyLUMXDYu2JXABEp5U94TXRqPYHV93XVqlX07du3QZ72GDFlDJXHN2CY9OrYlaysLEpKSpISngCu42CYJktmzeW/f3mci++9BcMy6yR6hTBYtWCJ///E6439+Zn99sccMvgIHcFqBIRhECorxzRNDh7YHzNgefa1cJhQeSVCQKQyhJUa5NKxt/Fx11f58qU3vXmTO3/U0w4BTMRrHnVHtfc0DYAWvM2bVnh1O6GWyIhhGNJ1XdG+ffuio48+OiKE6NIQ0d0a1rNLZEFF52IjsbGln5RYCwQCtGnThjZt2tCrV68al11eXh71b27YsIHVq1ezevVq1qxZQ35+PmVlZZSXlxMOh/cowUkIQUZGBjk5OXTu3JnDDjuMo446Kvratm1VcplqpGCYBkYNjw939dt60SOjgYWuWo8QglAoRH5+ftxp1bHOzc11fL+s6Z8DyW5YCC9KWo4nKDcD64HVwDo8ofuzEGKnv21BvLJaPf3hYDwR3BuvhF4OXqRxT0JRlXgRygJ/vav8Yb1lWauBlUKIoppmjCNuW4pnrvoNxC77yq6CUkWx1Y3HXj8GY8eOFQ8++KAcMGBA6pw5cxBCiNrEgar/vWbNGqBhvf7qe6oOUKvWrenbty/ffPNNneqOu/7vwo+z5vK/B5/g4vtuQZje4+1EN7uGYRAqr2DV957gTVwp2F+n9D7yNd8vY+fmrbTu2A47HNFe3gZG+E10QuUV/rnnd9wzqp4KurZD2HYYdd0ltO/Widcfewbbf/qXRJc/FdF1gNvxRO+VeL932tfbQGjB2zxRv8+ZVNXejUu7du1+Mgyjd6NuVQyxojYe1W0JsQJZeWfT09NRpaGqU1FREbVJKItEWVkZJSUlFBUVRbuqRSK+7SAlhdTUVNLS0sjJySE3N5dWrVrRvn17unXrRmpqao3b6DhOVNTHq2xR/RFpY6EE75YtWyguLo6Oi0fPnj1NwzAIh8PSNM0SPOFYDOTh1VndGTPkA4X+UOC/bhdC1FwnqWq7hBAiDCz3h+rvt8Frgd0VT/x2ANriieBsqn6Twv62Ffnbl4cXtdwGbBVCbE2wHbHR6ZYobhORaF/3aRRbSmkKIRwp5cVPPfXUCXPmzHFN0zRqszSp35K1a9c22jYpn75lWQwcOJBvvvmmzsLRdVwM02TxV9/x2l+f4sK7bsK2I0i3dguRdF2CaalsWr6GbWs3+eOSFPR+LkBlWQUFW3fQtktHbFmPlriauAjDqPX3XX2+ZUWlDD7zZLLb5vK/CU9QsrMw2WQ2dRNq4yWxdcTz5O9Ei94GQQve5k0KVVnZcX/hWrVq9TNepK1JUd0/WxPVo8Wx1RLS0tLo3r073bt3b7BtUjaH2PUkW75tb6HE7YYNGyguLo57/BzHcQGxcuXKV4C3y8rKtgSDwU3ANiFEaa0z1r5u5cVVgjLqdfWCdNH31RD7vhLUC+u63iS2IypshRDVLROaJoJ/M+Lm5eV1AB7t3r27n59Vu8hT761btw6g0Xyq6ns0bNgwnnzyyWQic7uh7A0LP59FIBhk9B+uJRwK1+qbl3j2sLU//oQdSToiuDtSn+5NAcM0KC8q5uDBR3DNxHt5+f5/kLdmQ9yqG9VQ+Tkn4VmSfon3JE2N1+wh2t3evFHV1+OFAyRASkrKVprpl0UJT9X2ODaDP9aHq6K5anAcp8ahpmliawerKG5dy7btTdS2rl27NlrVoDbBoIThCy+88IQQ4rXc3NyZQojVSuxKKYWU0vAHU0pp+YPpD+o9r3KaLyiFELY/OLGJXTHvO9Xfj1mXWds6EmxPvO1o6glmGg8hhJCtW7ceB3TOzc11LMsynDhltdS5nZeXF80VaIyqLWr9gwYNijbV2aMSY47XcnjOh1/y+t+eJZCaUqufU+AlyC77tu73gEr4p2Vm0K5rZ5yIraO7TQDDNKkoKaVDz25cPfHPHDign3cjlPyNmhK3A4HP8Tot1pqYrkkOLXibN0nf0kcikTDNVPDGQ4nhWEGsBtM0axxqmkZFcpsLfgTT+eGHHxJNqnJbIjk5OXlSSmvatGlWLcIxVqQqoerEvFdvhVGLGN5lHQm2R4vaZoyU0vCtDEcFg8ErAbdr165GVlZWwnnBS9CsqKhotO1TvwGdO3fmwAMP3GVcXZGuizAMvnvvc9587BmCqSmeUI+J3krXJZASYOvajaz98Wegypdb6zYahtfF0S+HJqXk5Mt+SauO7YhEtH+3qWCYJqHyCjJaZfPbCX/kqFOPj96sJfkZKdF7IJ7oHYIWvfVCC97mTYiqwvI1igAVUdi5c2cQL/GI2qbVNAskYJumaQDm3LlzAeI9AlWf9YaioqJtQgh7xIgRWjhq9jVj8Z5QyU6dOons7GygdnGpfse2b99e59redUH4IjI9PZ3DDz8cqJ99Qoneb9/7nFf/OgmEwAoEdqkMY6WksPDzmYQrKr11JfhWStf1Ho1LSfvunfnVHddy/Ogzq+bXNBkMw/C74AkuuvtmTrzw7KrObMmLXgfPz/uhaZqnoUXvHqMPWvOmAi+pJ2GB+a1bt3bAKxmlab6oIuUWsHHdunVT58yZc7kQorVUqcO7oy6fS/E+f13mRrNP8L27Ukp5DF5XOBcwAoEAnTp1ivpz41FaWkp+fj5t2rRptG6BKgrXp08ftd31Wp50XQzTYN7HMygtLOHCu28kPTuLipIyUtJS2LFxC3M++DLpUor9TxzCIUcfQU67NnTvcyDp2VlUlpXryG4TxWtB7xJxQ4y64ddkt83lvUkvRM/fJM4vVU2lleu67wSDwUvD4fAbaE9vndG3g80T9Q0pwstcjx23G0II8vPze+JlvcedVtMkUU0FTKDEtu0HgKMOOOCA/1ZUVORIKV1lT4jDAv9Vf+c1+wz/qcLN+I09XNcVAD169FDv1zifEgWu61JQUNDY2whA3759d1l3fVDVG376dgHP3v4AW1atI7NVNoGUFD5+7n+UFhbHFT9CCCSSlLRUfnH1RZww+kwOGXwEViBARWmZFrtNHCEESEllaRknXng2l467jWBqiid6k4vKG4ArpUx1HOeVjIyMuM2mNDWjL37NE4n32ZXj1SJV43afUErhJzT1WL9+fRaAushomjyqdJSqRPA/YFAgELhXCLEjGAyeQeKIrfqOfxOzTI1mr+J7d12/o9rZ+L9hSuB16eKVE09GuJWXl6tl"
        "Nsq2qm047LDDgLh2oTqhEtk2rVjD07eOY+6HXzL9lXdZ+NkshCESNtVBQvfDDiK3Y3sKtu6I1oRtDBuDlBLp+sNebuneYhECYRiUFRVz5MnH8dsJfyKjVbb3BKAOotdxHDMUCr2YmZl5DVUNqPQ1PQm04G2+qM9uWYLpVDH3Xh9//HErvEeK+nNv+ihvtoVnRbhGCHGxEGL5lClTAkIIwuHwUH/a2n7slM2hBFjkj9N2Bs2+QJ2jFwFZ+Oe3ElMdO3ZMvABfiCrB21io9XTo0IHWrVvvMq6+eJ5eQUVJGa88NIn3Jr3gj0/UVc17/6hTjscKWF492AYWulJKXMdLgrMCFsHUFIKpKZh+tQotfBsGwzQpKyrmoEH9ufqRe2jTuUNdWmUbALZtuxUVFVNycnJuFUI4VDWZ0cRBC5/mi/r1mee/JjrZzZdeeqkDXkmgxtsqTX2JtS9U4EV1jxVCPCulNF977TXz2muvjUgp2+GVrIHav8dK3M7Da9igP3jNXse327hSygBwiT96l3O2ffv2SS+vsQWvIjs7m27dugF1ErwJVaFqQpHsMr0uX9CuW2f6HX80oYrKGrs81jo/Vd0tDX/YZd2+0DUti/TsTKxAgKIdBeStWc+WNespKyomLSMdKxDYpcKEZs8x/bJlXQ/pxdUT/0znAw/wRK+Z1OcqAOE4jlNWVvb3Ll26/NkwDAfviYn+jY+D9n80X9QP6zd4HbMy2bW9ehRVS/K7777LnjNnDkcffTSO49AYLYY19UIlpQngDeB+IcQPUFXOCe87K4AT8TqUqXniMQ1P/OokB82+QJUiOwY4jJjfKSW8VPvueG18lcc1Eok06saq9aSmptK5c2f5ww8/JBMkCOO1y1YVc+LOUJdoqd+Vg6HnjiS9VRblRaVRYSTUv2LX6fE3QkqJ7Tq4UiKpat5jGIKgGfAT6kzSM9PYuXkrP0z/hmXfLGDb+k2UFXopHznt29D7yMM56dLzaNu5A+HKcLSlrmbPMUyTitJycju158qH7+LFcX9j7eKfk21QIQDDtm0nLy/vL/369Wu9bNmy24UQYuzYscb48eP1nUkN6Ahv88XFO+k34CUkqcjgbiifVzgc5v7779+Lm6ipA8qLtQ34tRDiV0KIH1SzBb/uLvjdxIDR/t/xrpym//4HMfNqNPuK0/GuOcquE6VVq1ZRoZlIXO6NG3UlvDt27LiLMK8B9f0rwSu1Fq42vl54dXsl3Q89kGNHnUKorALTF7tSShzpYrsOYTtCyI4QioSpiIQpD4cIRcIYQpCdlk6n7FwO6dCNob0O5+z+wxhz1AiChomZEkS6ks9feJ0nb7iH9ye/yKpFSyjJL4w29CnI2868j6fz7B0PsGPTVgIpAW1vaCAM0yBUUUlmq2yumPAnDh58hN+pL+lIr2HbtvPTTz/9/uSTT35eSmmMHz/eHTt2rNZ2NaAjvM0b1Xf7Y+CEeBOqiO4HH3zAa6+9xpgxY3SUt+mgIrffAL8RQqxQPms/qqsw8C6kPYFf+P+v7QN0/ekXAT8Qp1azRtPIqButk/3X6MVYCcns7GxSUlLi1thVIisYDDbKRsauyjRNAZSUlJSsBo6IU/YPvO9VG+BNYCXwMlUBiD0XHqqbJJKTr74A25CEK0JYlollmAQsi4BhETAtgpZFeiCV7NQ0slIzaJWWSXZqOhkpqWSmpJOZkrbLosORMOnZWSxfvJS3//Ec65Ys91ZpGNGIsjreQghMy6IgbztfTX2fX/mtkrU1rmEwDINIKEwwNZXf/OUOXnnoSRbP+C7ZFtNCCGGGQiF7xowZv7nssssypJSXCCHCOtK7O1rwNm/UyfwKcDdxbA1ANHpy++23M2zYMLp06VIXs7ymcVClZd4FLhFClEopLSFETdYDgfeZX4X3WccrS6PE7dsx02k7g2avElOdoRtwiD96t9+nnJwcUlNTkxK8mZmZjbGpu6zK9zD8+NFHH70PHOE4Tm2/q4IqW1Ev4L9AKjCFqqYBexJVkMIr22Y8NPERxvz2MiKVIQKBAIYwCFgWQdMiaAZIsQKkBhLfBEjppb65rkswEOSnbxbwr7GPUlHsWSS8qgzubnfFUkpU2+etazcSLtcNLhoaYRjYERvTMrnk3lt5PW0K8z6ejmF6NXzj4V/XrfLycvutt9761U033ZQppfyVEKKs2tPB/R4teJs3KoKwBvgMOI+qhKfdJ3ZdTNNk48aNXHHFFXz00UcAST1G1DQKSoi+BFwhhIj4P1A1CVNVfqwTcA3xo7vqvTDezRBoO4Nm36B+WA4FWlMt6ql+d1JSUrCs5C5Hubm5u8zbCKhtfLu8vHyDaZpxvcVU3Vym+vP9G8+a9C+gA1UWjmRVomsYhuG6rvj973/v3Hn7H5LKwI+1Gaj/Cf8flbjmOg6WafLBRx/y7N0PUel3Z0skqvwysmS2zsYKBghXhvQ1o4ERhvBuLKRkzJ+uJ5iawuy3P0kq0qtEb1lZmf3888+fHggE3pNSniOEKNGitwp9m9b8Ub86T5FEsoSyMXz66afccMMN0S+T9mTtdZTYfRn4DWAn+GFSloS78JLVlIe7JpTP91NgObq7mmbf08t/3eU8VL89q1atoqCgoFYRpcYHg0E6dOiwy7hGwHQcxwXeosqTmwyZePuXArwPHO2/mlR5lx2qvp+xSP892xe77nXXXbfsscceMwFhO47jSokalOVARguWEa3EUL0ag/B/JtRv/7fffssFo8dUid2kKi94/uq+xx2NaZn6etFICCGQrkskFOa8W6/ihNFnVj2FTXC6x4reZ599dsRdd931rpQyy3/CorUeWvC2BFQNvi/8Qf2w1j6D42BZFlOmTOGOO+7Ab0zRYAXWNQmpLnYlQByxa+J9pscC15PYG6giQs/6f+tQjGZf08d/3e1cFELw+eefRwVZTWJKiduOHTuSk5PTmNvp4tUHXmAYxgrTNEWC6G4sylegElDXA2cBlwI/+uOU+FVWCJuqQIUJWFLKJaZpnj158uSBjuP8Aci3TNM0hHANIaQRI2yr5GyCnfKf7m3atIkLL7yQsrKypMWuaXlVAw45+kiOGDGEirJybWloRFTyZqiigrNv/i0nXXqeX2nJiPq6ayNW9D755JMnPvjgg+9IKbO16PXY7w9AC0FF/x4kiSgvgG3bmKbJY489xg033AB45vk6/Lhr9ozqYteFaMvVmlDR2QzgGapsSLV9xuoDXAR8RJXvV6PZF6jz+uDd3vCtVGVlZfznP/8Bau9qpgRvz549ycjIaJwt9TcLwHGcl6WUIhAI7GlWb2wzgJeBY4AL8aLGa6ny9qoyg9uB6cDVUspjHcf5QAhRYVnWRGAQMJVdhXLddsoPaFx99dWsW7cOy28mEQ8lqh3bofNBBzDmT9f7C6vr2jV1RZ3vlWXlnHntpZz229Fe05Ik6jcr0VtaWmpPnDhxxKRJk97RkV4P7eFtGagfz+nAi8CvSSJZQt31T548mU2bNvGvf/2Ldu3aYds2hmHou/iGRT2yVJ7dy0ksdpXvzwaeBPqR+HMV/vA3IEJVdFij2Reoc7uD/xq9Wtu2TSAQYMKECaxdu5Z4Xll1kT/88MMRQmDbdtKe3zpuqwkU79y5801AhsPhujwdqf49VorSxGsD/6o/tMKzeOT675XglZfcEDOvIYRwXdc1hRBrgDFSyt8CfwXas6ugjov6nX/88cf56KOPsCwL246fvyoMI9pk4ogRQxl1w2Vkts4hUhlq8A5vmppRkd6KkjJGXnEhQhh88u9XEYYX149nK/FLkVqFhYX2uHHjTszOzn5HSnnW/p7IpgVvy0FFdu8CTsO7wMSN9qrsW8uyePfddxkyZAhPPPEEv/jFLwDP+mAYhk5OqD+qRJGFZzO4jiobQzyxq8rO3YsnkONVZYAqq8MPeBdWHd3V7FOEEKqVeVbs+EgkQiAQ4OOPP+aRRx5J+HhdvTd48ODG3FyV8Pthly5d1vvbVBd1V+a/Vv/BrJ60"
        "VohXO706SsC6eJ3p8Bt2qBKF/5FSfoYnei/B7y1Rw/qqdsiPCq5atYpx48YlZWNQlQHadOrAObdcQZ8hA7DDES129wFCCCSSipJSTvvtaAJWgPefeSka6Y0nen3vr7Vjxw77zjvvHGFZ1lQ/kS0ipRRxrj0tFn32thyU2NkM3E7Vo6+EJ7WyN6xatYpRo0Zx8803k5eXh2ma3iMtx9H+3j1HXexM4K9CCFVhIZHYFXgC93bgfpIrb6SWdz9eso2BfgCp2UfEtDnNBtLAu4l2HIdAIMC0adO48MILsW17l7qv1RFCeKW0gkFOOMErN95IT5/U9r4Q8+i4LobhsjjvqSc8sb8HsYOyLu32my2EcP3H0aYQYqMQ4lLgaqoEb63fcWUbmTRpEoWFhQkFr/ArNhw4oC/XPT6Ow4cNpLKsHMe2tdjdR6hzsaKkjBG/Po/zb7jcK5khEiduKtG7efNm++677/7Fe++995If4RX7YxtifQa3LJQo+i8wiao6kIln9KO5ruvy5JNPMmDAAB566CG2bduGaZpexx8/Iuw4js7STYykKnmlHK/s2F0x0ZraDqDqjuYC44GJVN3MxPuBUp/9F3gF8HVlBk1TIcV13YBKSjNNkxdeeIFRo0ZRVFSUMFKlnjIdc8wxHHDAAY1VRjH26cjnUkrTt1e08d+P94OnrqPFSUyr3neqDcl0GHCklIYvfP8F/IE4ScpSSkzTpKSkhNdffz1641Dr8n0bw8GDj+C3E/5IdpvWlBeX6qd8TQFf9JYXl3LcxWdz6a3XId26id61a9fad95555ivv/76mdTUVFcIYexvolcL3paH+uG+FfiEOjQcUD+GpmmyZcsW7r77bgYOHMidd97J999/73Xc8S9Y6sdTCWAdAd4FFcWx8JLHTvIfR5p+tKamC6JBld+2FfB/wH1UlR+L98OkHmuGgTti/tZ3JZp9jVi7dq0wDEOq35Urr7ySyy+/nPLy8uiNdCKklIwePTqaWNsIAkxtxPNCCHvs2LEB/++21d6vaT4Dzy9fkGDaeuN7L10pZUAI8TeqEll3E73qN3nevHls3LjR27Daoui+2O12aG8uHXsbpmkRrgxh6E6cTQchMIRBaVExx1xwBjf88TZf9CZOZPN93NbSpUvt22+//crvv//+b5ZlOSeeeKK5P4le7eFteSixYwNjgA+BYST2f0ZRFxTDMNi4cSMPP/wwEydO5Nhjj+WMM85gxIgRHHTQQbRt23aX+WoqbZZsVmnsa31Q272PUPU1TaASeBx4IInuabFRmhHAE8DhJN+lSXkPH8ET2DpRTbOvESeeeKIJ2AcccEBhKBRyn332WR588EHy8vKivwnJVApwXZfc3FwuvPBCgMZoh66+s/nA/wCmT5+uvqvdk1xGMV6lhUbH90Xb/tOim4CD8H43dvmNV7+n8+bNQ0pZa7Ka8LtKZORkM/pP15OakUaovBLD1PGwJocAyzTZvmMHR190BjmpGTx0/wNepzwn/vXTf8Jifffdd/YNN9xwW15eXkHbtm3/Mm7cuP2mC6cWvC0TFeUtBs4G3gBOpA6Zvcq+oB5nOY7D119/zddffw1At27d6N69OwcddBAnnngixxxzDIceeugeXYwaOloTK7z30uO4WKELXte7e4QQc/3tqal7WmyJIQfoDPwRuJG6tSRVYnc+Xlk6bWXQ7EuiN3AzZsywgbRu3bpdmZKS0nbVqlUAIonOZVFM08S2bW6//XbatWsXrdXbwKjv0CtCiDz/SYwNBKgSvLWpPxVgyAe2xIxrVHzRK/wEpIuBmcCBxPxuqBv/FStWeBuVwCN97Fmn0O2QAynJL8Bs+AoYmgZCAimBICs3rGPIxWcx3rAYO25c3ConCv/7Y37xxRfONddcc7+UcqsQ4plp06ZZI0aMaPGiV5/VLRclevOBM4C/A9f67yXd310Jx9jIqeu6bNiwgQ0bNvD111/z/PPPk56ezmGHHcbhhx/O4MGDOfDAA2nXrh25ubmkpaWRnp5OIBCI+u8cxyEcDhMKhaioqKC8vJzy8nIqKiooLS2lqKiIoqIiiouLKS4uprS0lFAoFI1QBAIBUlNTadWqFW3atKFr16506dKFTp060aVLl10uirE/Ag0sgKsL3UV4iWmvAkgpLcCJKQGzSxa2P64z8FvgZqpKN9XaHroa6gpWDlyJF1U20YJXs/eJfVLh4CWqXQbcuHHjxj6AaiqRdCMHJXaPO+447rjjjqqOUw2LsiSEgKeklGLcuHHqB6I1VR3iavvRUN/BtXhRsr12wxmTyJYnpbwQryxlhr9+Q/3Obdq0Ke5yVCWHw4YOJFIZwjC0jaGpI5GkB1P47ufFnHvtJbi2w/gH/pJUyTnHcYRpmsabb77pXn755ZOllDuEEG/651KLfjKoBW/LRoneCrxSWIvwkqAyqEqoSkr9qYivorpwLC8vZ968ecybN48XXngB8ERy69atSU1NJS0tbZe6ma7rEolECIVCVFZWUlFRQWVlZf32FujQoQMHHHAAgwYN4uSTT2bYsGG0b99+l2mqt1KObcmZJNXLDC3Esy+8KoSo9C+apu/VNWLmib0QHoUnCC4COsZMY5C8t17V9b0Z+B5tZdDsG2IrDLTCqwN+M17EEcAxDMOsS1MbdeHu2bMnL7/8MsFgMCrMGhh1c/mBEGKJlNIYP368+nHoiyd6k2nmo8qM7VU/pJ/IZgoh5kspLwNe99+KbnNFRUVSy5Ix/2qaPhJIDwZ5f/5MrrzzVsrLy3n0b4/VRfTKF154wWjduvVLUsozhBDTW7ro1YK35ROb9PQ08B1etHe4/37S0d5dFlqDV7d69MVxHPLz8+u03PpEYB3HYevWrWzdupXvvvuOSZMm0aVLF4YMGcKIESPo168fBx98MB06dKhx/mTaK/v7aQJEIpGFtm3/Y/LkyVNvv/32CsCYMmVKwH8cWtMvzpHAycB5eG2C1XFXQrcun4Py6z0O/Js6JCdqNA1EbFQ3gFcr+o/ECF1/GjPZpFaVGGvbNr179+bdd9+le/fujR3dlXjfI9g1QfQ4/zWZJy4LY+bfq/ii1xJCvC2l/B1ehR7bFy8iUW6EqpKRt2Y9PfsdgqwMJdmwWLOvEYaBKQ1e/eZT7p/4V0rLypg85emkRK/ruoZhGO4///nPtC5durwmpTxZCLG4JYteLXj3D1StRhPvh/lk4Pd4TSpa+9PskfCNrqBaBBiqvLnJJq1B4iSWeNQUqd20aROvv/46r7/uBT46derEgQceSP/+/enfvz+HHHIIXbt2pV27dmRnZyfjD6wEPgf+np6e/qX6UTFNE9d13WuvvVbtQCfgAKAPcDxeRPdwdv3O7YnQhSqx+zbe51hraSKNppFQVUAcvKTYh/1XqHq0X6fzWlVsUDaGF198kQMOOKCxfLtQJWQ/FkJ8Va0DlaBqf2pD/aaW4T09U8vc6wghbF/0PiWl7ArcZRhGBAi0atUq/ryGgXQcNq9YixCGV+NV0yyQUhIwLUoqy/nfd5/x1NOTKSop5r///W9C0etfdw0ppXPfffe169Sp0xtSypOEEBtbajc2LXj3L5TAcoBH8ZLZ7sKLzFjVpqn3LX5DVl9Idn3V1xUbMZZSsmXLFrZs2cLMmTOj0+Tk5JCbm0vr1q3p2rUr3bt3p23btuTm5tKmTRtatWpFIBCQkUikYNGiRVPuueeej/BKgJ2VmpraKhwOZziOkwt0AXoAXfH8uB3ZHSVM90ToxrYn/gK4GF2CTLP3UeebgVc67x6qEi1VOb6kiU2MNQyD3//+9zzwwAOkpqY2ptiN/d5M8Mep3zwX73usBG+ihLWVwIqYefcVyt5wt5SyNZ6NzW7Xrl3cz0P9Zq5ftoLyklKvFFkyJg5Nk8CVkvSUVFZt28QXyxfw8ssvU1JSwnvvvZeU6DUMw6yoqHD+8Ic/HOSL3lOEECUtsRubPqX3T6qXwjoWr6PXOXiPJol5r0VlMFS3TCRjY4ghekz8iG4i"
        "Ma8Eqqg27AmxiW+v492klKGrMmj2LkokZuDVij6fqkYpdfqtUDYo9WRo0KBBTJgwgVNPPRWgsWwMCvVE6wMhxKiYiJayBt0G/I34T77Uk5ZH8awc+9xD79dUFYZhuKFQ6MVAIHDpww8/bN95551WIvFjWCa3TPkrHQ/oRiQc1s0mmhlCCCrDIS495jR6ZLXjlNNOZebMmUlVb/Cnsfv162e9+uqrH/bp0+dc3yojW5Lo1YX29k+UEFNRxm+B0cDRwLN4BdRVy0uoyrxu9id+bLMM1TBDXXhVUw01WJaFZVmx40zTNE0hhHQcx5FSKq+uze5dk5QItahqHbqnVxD1iNjAq7V7AVrsavY+6kY5iHfTdT5ewwWog9hVPl1lg+rVqxdPPvkkM2fO5NRTT412cmxEsavilw7wgNqsmHHKj6zG14aqiPJ+o2zlHuCLE+m6rnj77bcvB14/9thjLcBJ2FLYdli94EeslIDX0EDTvJAQsAK88/0swobkzTff4Igjj0jqKYmq0bt48WL7+uuvPyM/P/9py7LcqVOntqhubFrw7t+ozGolphYB1+D5TW/GE8JQJX5VSS1HCCFbSgRARXljhbDjONi2jW3bu433fwBMPDGrBjNmUMezvgdI3WRYwEbgV8CfqLpga7Gr2Zuop0J/B07HE7sBkj/PZazQ7dSpExMmTGDevHnceOONpKamqo5QjR1dVNVrXhJCfBuTpKMS2M4B+sdMV9syBPATXiKwGrfPURG5MWPGuMBF6enp/23Xrp3puq5b23GNljBb8BNB18RF6sS1ZoZEYhkGpaEK3l74FW3btuPtt96md+/eUbtQPJTonTFjhn3rrbdeEYlEHhgzZky9cnuaGtrDq4FdH5cLYB3wpD8MAkYBp+BFgAMQ9X25gBRCCMP/NikfbUP7dmMT0lRWcWOtqwkQzW7HuwD/C7gXyKMqqtTidlrTpFGP688HbsB76hCIO8euOIDpOA7t2rXjuuuu48Ybb4xWTFEX5L3QJVHdLBYCY/2b11g/bxDvu6amrQ0lhl/Bq+G7z+0MscQ0prCllJeVl5cPAg6WUtYo4qUf/V264Hs6VJpUpqdTWVGBaZhI/VPTbHClJDUQZPmOjXy1fBEnHHwkb771JieffDI7tu/AMIy4Fj7HcbAsy3zxxRft3r173yOl3CaE+GecTqHNCh3h1cQSG/FVd3XzgHF4JXr6BgKB3+bk5PyvR48e29u1a2cEg0FTSmn40U/XdV1HSukArhBCqotYdbtAvEHNUz3hTEVhVdRVeWhjl78P2wrXF3XsoSpKPA3vRuNqqsRui7CWaJoV6mlCa+Af/rhkv2jqptgMBAKRW2+91V64cCH3338/HTp0wLZtpJR7I6obuz0GMFEIsQ6vQYPyH7vArVRFd2uLbKnqDCXAyzHjmhRCCDl8+HBLCOGWlZW97Y+uUe2oz6A8VMnCL75m1BFDqQiHdJZPM8SVklQrwMw1i1m3M4/+/frz1ltvkZGREW38FA/btoVpmua4cePcl1566XEp5UWqCshe2oVGQ5/OmngoX5tBTI1XwzD48ccfO02fPn2w4zjnrV+/fvCOHTsOz8/PZ9u2bWzevJnNmzcTiUSgqiSaYo8uDNnZ2bRp04a2bdvStm1bKioq5OrVq1m/fn31czg2v1hlktcnWawxUZHa2CYWEvgU+CfwoT9OR3U1+xJ1ozUW7+Y32cec0emEEFNzcnIeKCgo+A9wVCQSkaZpGnv5BlVtzw/AELwSg1IIYeL9vh0FzAJSif+boZbzPF6XxKbspTellK4QYgReOUWoZb/Uk7N27duxevlKFu5cxydLviMjJa0lPkVr0QghCNs2nXJyuWzQqaSlpPLGG29wwQUXALs3X6ppfsBNS0sTn376qT1s2LBzhBAfNfdIb1MUAZqmSfVKA9GTXkqZjldsfuDOnTuP3rFjR++ysrLuxcXFHYuKinIKCwspLy+ntLSUsrIybNtGCEEoFMIwDILBICkpKQSDQbKyssjKyiI7O5tWrVrRoUMHsrOzycjIICsra7eN2rZtG0uXLuXbb7/lhx9+YNmyZaxYsYLKysrqX+pYwdggZdfqSOz6YwWuYiXwHvAiuxax115dzb5EPepvCyzGK7cX20GwNpQoXAf8wTTNqY7jUFlZeVtKSkqi6geNgaokIYEThRBf+5UZVKJaW+ArvLrZ8by76gbeBgYCS2jadbDV55eGJ/QPJM7+qYz+iRMncvvtt/PZ0rnMWvMjKVYgKog1zQNDGJRHKjm+dz9OPWQQCMGkSZO46aabsCwrmhxa6/ye/cHt2bOnMWvWrILOnTuPFELMbc6NKbTg1ewRY8eONZYuXSq2bdsmZsyYsdsdn5QyC6/NaA7QDu+CklltUEkiBpCClwRjE/+8TA2Hw6mGYaRalpWGVx4p1X8vtHnz5rIZM2bsnDJlSmTWrFkdHMfpg9fwofrFVVYbYiM6df1eyBpeY5dX28VzMfA18BZe4ktRzPqb8kVUs/+goruXAC8RXwwqoiW/8JJgNw8fPtyaMWOGK6Vsh3fet/Wn3VvXoGgJMSHEH/2LttrWTLxKC8NJvH9q3/6FZzVqytFdhfoMHwTupupY7IZq/tG1a1e+//57WrduzezVP/LFz17n5IBp4cqmvrsahQBs1+WSwafQq00nEIJ77rmHCRMm1KVcmXv88ccbX3311QbgNCHET81V9GrBq6k3qvYjIKZPny5GjBgRW5ZrX5OKV0R+IDAU6Ascihep2ptsAFYBS4E5wFxgNd5jVYXFriXNNJp9jRJL/8Mrh5eo3q662ZuEV+lFVRlRrW4dKeX9eIlhtQqvBkaJ1IXAcVOnTg09/PDDxvz58yNAe+A1PLGbKOqsbo4LgAF43+nm8ARGfYZ9gQVUHfMar/9KCD06cSJ33H47ACu2b+T9H7+hsLyU1EAQ2HsNhTR7jhCCiGPTJiObK4ecQdAMYBgGl112GS+99FJSLYj9aZzrr7/efOqpp1ZUVFScmp6evq45il4teDWNQowIht3Ps4Y879SjSgEwffp0Y/r06UyfPp0ZM2ZAjPUihg54XdH6AYfgtQDujBeRzvKHFKou1rEXQVVrV9XeDQPl/lCCl/1dAGwC1gNr8YTtDmBnDdtisevjVo2mqWEC84EjiC8KVXT0STyxa8SMV78JAG3whFdXkrNH1Af121AaDoeHpqSkLI0ppn8s8G88G0MyFgs1zS14HvsmVZkhASoS/S5wFklEebt06cLixYvJys7GNAyKKsv4/Kf5/Lh5NUIIAqY3uxa+TRtDCMojIY7p0Ycz+w7BdV3C4TCnn346M2bMSCrSGwgEiEQizpQpU8xrrrlm8apVq0458MADtzU30asFr6alU917HK/KQQpeRDgVr+SSuhjHXpDdaoODV5Yo7L8m2hZVaix20GiaIsr/mYsnUHtQ+yN/NX4pnoUIvO+QTcw5rjqaSSmVRaIxo7zRDnAVFRUXpaenv+Jf3DPxOqndg/edr4vYVZVT1PKby/dXifORwMcksG4oEXTvvfdy//33Y9s2luV9TD9tXc/MlT+wqXB7VPgaQiAluoRZE8VLYotwyeBTObBdFySwNS+PESNG8NNPPyUsVxbTFdGePXu2NWTIkDlr164d2bNnz8KYLoVNHi14Nfsb1SPP6v8NVe4rViDHenqr+3w1mqaOEryt8SK8PUmc0BXG88PeiZeICbt2bCTG2vASnje4QUWv39hCAliWJYCbhRBPDh48uM3cuXN/BdyE93ifBPtDzDQCyMeLCq+keXh3q6N+72bhVamoVeir0lUZGRksXLiQ3r1740oXwzB9X6jDsrx1LNiwgg0F2wjbEQKmiWm0mB4FLQplbWiXmcOVQ87EFF4pzyVLlnD88cdTUFCQMClRRf47dOhgL1261GrduvVXO3bsGNWuXbuS5iJ6teDVaDxELf9PRPVfCC1oNS2F2KTLecCRJCcQAYqBp/C8vBtjxpsDBw40Ro0aJceN"
        "G5cOfAEMcl3XNgyjzqJXNZ6JbRGuSirhCbpHhBBvBgKBSyORyGg8GwVU1RtP9F1XUWIDOBfPEtCcrAyxqO0+B3ibJKO8o0eP5rXXXou2qK1ey3VT4Q5+2rqOn7duoKC8ZG/VU9bUEWVtOKH3EZxy6EAidoSAFeDDDz/k7LPPBhKXK1PnxNFHH21/9913luM40wsLC89p27ZtcXMQvfrM1Gg0Gk1tKJH0LHAVdXv8D553/VV/mIV/Q6geoZaXl3cPBAKfWJZ1aCQSkYZh1HpNUhfi6l0Xq1NQUCDnzZsn3nnnnS2TJk1aYxjGsVJKw59fdTFMRrRLf3oL+AMw0f9/s61DStW+zySJKK8QAtd1+fjjjxk5cmRU9AK7Cd9FG1fy9g+zSLEC2tfbhJHAb44ZSddW7QhHIgQDAf7xj39w2223JeXnVYlu119/vf3UU09Ztm3PLC0tPbt169ZN3t6gBa9Go9FoakMJ3pPworHJ1s+NFYuKpcBnePVuf7zmmmvWPPPMMxEpZSvg/4DT/emTvi5VVlayevVqfvrpJ5YtW8bMmTP54Ycf2LJlC0DsY1qb3f34yW7/RDzB29zFLlR9nqfgNbiJmzSobkwOOugg5s+fT0ZGRnS8wnFdhIAfN6/hre9nEtSCt8liCEGlHaF76/b8+piRGMLA9doJc8MNNzB58uSkKjcoYfz888/bv/nNb6xwOPx1MBgcJYRo0qJXC16NRqPR1EbsNeIz4GTq1jQimjhWbXwpkGdZ1krbtn/Ozs7efv/991/ZpUuXHtnZ2cI0TaEsCqFQiPLycsrKytixYwd5eXls2rSJlStXsnHjRvLz8ykvL991o715pe/lrWunRVUxxQT+DvyeltXtUPmP3wHOJsHnqcTNrbfeyt///vddorzgtbI1hGDx5tW8uegrLXibOIYQVIRDnH7YMQzpdTiO64JvCzrzzDP57LPPEkZ61Q1PdnY2X331ld2vXz/Ltu3ZlmWNEkIUNFXRqwWvRqPRaOKhBNJheA1SMqh7OTGVuKl8o9F5G6KDl2ma0eXEDntArPi7H6+dsmqQ01JUnPo8D8fzZgeJc1OgbjwApk2bxvHHH1+jtWH5tg28Ov9LLMNsMQeqJSIQuNIlaFlcNXQUrdOzop9nXl4eQ4cOZc2aNQkrNyhRfOSRR/L111/b6enpluM4s0zTPFMIUexV/xNN6lTYq43MNRqNRtPsUBHapcCV/jhV4i9ZlHfUokpAuoAjpbSFEI5lWVHhGusNVYLLsqzoYJomhmFEk9Qcx8G2bRzHSZh4Ewcldsv8/RzLrmUEWwrq81yCV084bsUJdfPgOA7XXXcdpaWlUW9vLKmBFAxhtKgD1RKRSEzDoDRUyVcrvgdQJcfo2LEj//vf/0hPTweIm4CoRPKiRYu49dZbLcA2TfM44DUpZSogYmpvNwm04NVoNBpNIpQYfA34DV4bcJM997QqAWwClpTSVII1OkGM8HVdF9u2o4MStvUQt7E47CoCT8NrSKH8ri1Rw6lI+wS8DpBxRa/rupimydKlS7n11ltrjP6lBz3Bq2n6uFKSagX5YfNq1u7Mq6qzazscc8wxPPHEE7iuG1v1pEYc3//77LPP8txzzymP+0hgim9pMJqS6NVnp0aj0WiSQYneF4Ez8boIqnbYDVamq7otoRH9oKp5jElVh7ihwGyab+mxZFHe5iI8j7KquVwrSvQ+99xzvPrqq1XJTapmbzBN+3ebEwJcXL5cvgDHdUCAYRnYts0VV1zBrbfeuptfuyYcx8EwDH7/+9+zePFiJXp/LaX8s9+FrcnozCazIRqNRqNp8ijR+zkwDE/8qkitEr5NXfEooau8xN8Cp+K1Qy6m5Ytdhfos3wVeJkHEXt18CCG46aabWLNmDZZlRSO9QdMiOzUdR7q6Fm8zQEpJihlgXf5WFm1cicDrlqe8uQ8//DAnnngitm3HFb3qnCguLuaKK66gvLzcdF3XAf4ipTzPbzLTJDqSaMGr0Wg0mrqghNJm4NfAGXilxpTwVf7eppSl7VIl5pTQXQxcDhyHJ+BVI4r9QewqVKT3NmADVTcuNaIafOzYsYNLL72UUCjkZfhLzxfaKi3Ty/rXNAskEDBNZq76gbJQBUZMfetgMMj//d//0bFjx2gUtzaUtWHevHncc889wjAM4VdIeUZKeZgveve53tznG6DRaDSaZkdsA4ePgBOBs/BKXYWosgmoafd25FfV0Y0Vuaom8Ew8oX408AJVAr6llB2rCyrSvR24niprQ63HQVkbZs+eze233+5HBL3D3DYzB1e6CF0AqlkgpcQyTPLLS5i9Zkl0nEpi69atG88//3yNyaTVUfaHf/zjH7z77ruGaZou0BZ4SUqZThNIYtOCV6PRaDR7QmyNXQm8j9d+ty9wH17JK/W+ivxClRBVUWBJ7SKrNuEVO4+yUqhl4q/LpErkrsZrc3w8MBzPilHJrhHp/RUl+D/Aawed0NKhRO+kSZN4/vnnCfje3faZrQkYFnK/u29ovrgSUqwgCzYsZ0dpUbS8n2ma2LbNyJEjuf/++xNGeWMtL9dddx0bN25UFpmjgAlNwc+rb8M0Go1GU19UtFcJUPDEZh/gF3htbAcDXfbS9pQCC/AS0D7Dqx9cFvP+/hrRrQ1VhzcDmAMcSlX0t+YZ/Ihfeno6X3zxBUcffTTbSwp47puPcKV6AKBpDhhCUBEJc1TXgzj3iOOiwlX6DSlM0+Tss8/mvffeS9iUQr1/xhln8N5770nANQzDBM4WQrwnpTR98bvX0WekRqPRaBoS5ZGtngCVjSekBgAHAQcD3YFOQA4QIPlrkrJJlAJb8PzEq/xhIfAjsLXaPLE1dbXQ3R0V2T0WmIb3eShfc42o8mQ9evTgm2++oWOnjjw78z3ySgt0pLcZIqXkt0N+QeectlHRq3zbW7duZfDgwWzcuDFhUwpVweOxxx7j97//vfTtDnnAILzvq9gXndi04NVoNBpNY6Cihga1e3iDQBqe4M31/5/uT1uJV+836A+G/3cYT+gW+q8VeL7hmrCoijpr9ZUYVVbqd8DjJNFGWkX0hh03jOlfTOPLNd/z1c+LyExNw9UlypoNKsrbr3MvfnXU8KjghSp/7qeffsoZZ5wBELcGtor+p6WlMXPmTI466ih1Hr0uhBjtV21w93YnNi14NRqNRrM3iBXA0PCJbMqvG2ur0IqrbsTeoPwPuBBPAFvxZlIRvWuvupr7Hn+EZ6e/TcCMO4umCSIAR0p+c8xIurVuv4votW0by7K47777+Mtf/pLQ2qCiwEcffTQzZswgGAw6vrXhIiHEK/vC2qAFr0aj0Wj2FaLaa+z/ZQ3jYsdXF7Na3DYMyoudA3xDEn5eANMycWyHP955J4eNOZk1mzYQDAQaf2s1DYaK8h7WqQcXDDhpF8Gr/LxSSkaOHMmXX36ZtJ/37rvv5sEHH3QdxxG+teEIYAd72dqgBa9Go9FoNJpYlJ93EDAdz2qiIvQ1IoRAGALXcbnsTzdxxKgTqSguxUjQqUvTtFBR3suPPZ2urdrtInpVu+E1a9YwaNAgCgsL43ZDVC2LhRB8/vnnDB8+XFkb/k8I8Zu9HeXVZck0Go1Go9HEooTJPOA64vuwAb8slSsRhuDliU+xePo3ZORkx40AapoeQggijs23a5bu9p6qz9uzZ0/+8Y9/RAVwbfV5lRi2bZtrr72WoqIidR79Wkp5+t7uwqYFr0aj0Wg0muoo0fuSEOKveD7euI+fpZQgwXUlrzz0FD/PXURGVhauFr3NBldKUqwAP29dz5aindHyZAolei+77DIuueSShPV5VVmzn3/+mbvvvlvgNaAAmOg3pJB7qyGFFrwajUaj0WhqwgXM++67757MzMyPATOR59J7BA6h8gpeGvcP1i1bQVpmhha9zQhDCMJOhHnrf97tPWVTcF2Xxx9/nJ49eyYteidPnsynn35q+DaGw4Fb/fNpr2hRLXg1Go1Go9HUhATk+PHj3fHj"
        "x1/VtWvXAimlMAwjvuh1JcIwKCsq5v/unci29ZtJzUjHdfZ66VXNHuBKSdAMsCxvHfllxbtFedXfbdq04emnn47aGhJZG6SU3HTTTRQXFxt4N1N/lFL29lYpG12PasGr0Wg0Go2mNtyxY8dat99++6Zx48bd0rNnT+G6rowX0QOQvr+zcNsOnr/nEQrythNMS9Git5lgGgaloQoWbVxZ8/t+6+HTTjuNm266Keko74oVK7jnnnsEVZVAHvLr8Ta6rUFXadBoNBqNRlMrUkoxZswY46OPPnL+/e9/v3/jjTeeuX37dkcIYdaWoa9Qj7879e7OVQ/fTWZuK8KVobjiSLPvEYDtuuSkZXD1sLNICwR3qdgAVZHbsrIyjjnmGJYtWxa3C5uyQ0gp+fTTT+XJJ58spZSGEOIUIcQXjV21QQtejUaj0Wg0cfGFiZRS9vzvf/879/LLL2/lui6u6xoJRa9p4Dou3Q89kN/+9U7SszIJh7Tobeqourzn9j+Oo7odhCslRjXbgurCNm3aNE499VQgfhc2JYgPOeQQ5syZ42ZlZRmu635nmubxgNOYdXn12abRaDQajSYuQgjXF72rL7744tueeOIJw3EcNxnR6jouhmmw/qeVvHDvRCrKyggEg7VGAjVNA4knehdtWoUr3d3ELlQ1lxgxYgS/+93v6lS14f777zeEEI5pmsc4jnORf441WpkyHeHVaDQajUaTEL98lOHXT33pL3/5yyX33XefbVmWZdt2wvkN08R1HHr2O5TfPPAHUjPTiYTCOtLbxHGly2+O+QXdWrfbzdYAu1obBg8ezM8//5yUtcEwDGbMmOEOGTJE2La91rKsI4FSf5oGvxvSZ5lGo9FoNJqE+MlFri98r7/33nuX3HDDDZZt245lWQnndx0HwzRZs/gnnv/zo1SWlhNMSdGR3iaMIQQR2+aHzatqnUYJ4KysLB5//PFoxYZEVRsikQi33XabEQqFXMuyejqOc5MvdBslGKsFr0aj0Wg0mqRQGfVCiBLg0kmTJhX/8pe/NGzblnURvWsX/8Tzf36EsuISgqm6ekNTRUpJwLRYsW0jZeHK3UqUKVRDipEjR3L55ZcnbW347rvv+Oc//2l4q5K3SSk70EhlyrSlQaPRaDQaTZ1QGfVSyvOBN0aMGOFMnz7dsCxL1MXe0PWQXlz+4B/Jzm1NqKICw9xrnWY1SaKS184/8gSO6NK7RlsDeCJWCMHOnTs54ogj2LJlC0KIuNYGIQSZmZnMnTvXOfjgg03Xdf9qmuZdjVGxQUd4NRqNRqPR1Alf7JpCiDeBP37yySfmwIEDXdu2qUukd+PPq3nujw9RsHW735xCd2RravghfZZuWeuNqMWqoHy7bdu25dFHH61VGEeX679fXFzMrbfeqqK815eXl3enEaK8WvBqNBqNRqPZE1wppSWEeDQYDD7y1VdfmX379rVt28ZMIlLriV6DLavX8a8/TmDr2o2kZWZq0dvEkFISMEw2FGzzOq/542pCVW246KKLGDlyZLRsWW2o9z/66CPx4osvuqZp5pimeUdjNKPQlgaNRqPRaDR7RLXKDc+UlJRcPXDgQHvFihWWZVkkY28QhoF0XbLbtubX42+nZ/8+lBeXaHtDE8IQgvJIiDMPH8IxB/SJG711/S57ixcvZsiQIVRUVEQT1Wpctt+Mol27dnLRokW0b9++PD8//6h27dqtxPOLN4jBW0d4NRqNRqPR7BExlRsMIcS1WVlZz3733XdWr169nGQjvdJ1EYZB8Y4Cnrvrryz7ZgEZOdk60tuE8GryGizftgEgrlVBJbD169ePW265JSqAa0O9v23bNnHHHXc4pmlmtG7d+vcNHeXVEV6NRqPRaDT1wo/04ndjm7xt27brhgwZ4qxevdowTVM4SYhXYQikKwmmpjDmj9cz4LQTKCsqQRi1l7jS7D2klFimyVVDz6RNRk7cKK+K5paWljJgwABWrVoVN4ENonYI+emnn3LqqaeWrV+//oju3buvoYGivDrCq9FoNBqNpl740TjVgvj69u3b/2X27NnmQQcdhOM4bnKRXk9AhStD/PfBfzLrjQ/JyMnCNE1dtqwJYBoGZeEQq3fmJZxWidusrCz++te/1mpnqGnWW265xQUyu3fvfktDRnm14NVoNBqNRlNvfHEifdF7X4cOHX735Zdf0qdPH8NxHCcp0etHDV3H5c2//4upjzxNZXkFaVleBYc6CCdNAyPxROOq7ZuSmt40TVzX5fzzz+eUU06J1t6tDVW7d9myZcbEiRMlcPmqVasarGKDfkag0Wg0Go2mwfDtDaYQwpZS/nLDhg3/Pvfcc7MXLFhgm6ZpOo6TWHsIEHhNDtp27cS5v7uCPkMGEK6oxI7YGKaO1+1tBOBISUYwlauHjSIzJS1h6THlz503bx7Dhg3DcRxc1631xkUtKzMz01m/fr3ZqlWrR4UQf2yIurz6jNFoNBqNRtNgCCGkL3YtIcQb3bp1O+Gdd95ZcOKJJ1p+FC+xP0F60V7DMNixcQvP/WkCbz/+nB/tzUDGEU2axkEClmFQVFHKpsIdSc2jEtgGDRrEb37zm4Qd2JSALikpMe68804JXLFq1aoG6b6mI7wajUaj0WgahZiObNmrVq2acPnll984a9asaOOKZJYhDIH0zBK07dqJUdddSt8TjsEOh4mEwrp82V5ECEFlJMzQnocz8rCjE0Z4oaoD2+bNmznyyCPJz8+PW6YMPKFsGIazbNky88ADD7xLCPHX+kZ5dYRXo9FoNBpNoxDTka34wAMPvOmhhx46o1evXquFECaQlHiRroSYaO/zf36U1x6ZTEl+Eek5WZ54ipP9r2lYDCHYWLgdV7pJVc9QHdi6dOnCbbfdlrBMGXjC2rZtceeddwJcs3379iy8KO8eB2q14NVoNBqNRtNo+KJXjB492jz++OM/GjFixFDTND8CTMDGe1qeENd1oyXK5rz/BZNu/DOz3/oEKxgkmJrq1e3VNodGxStNZrGzrJiC8tLouESo5hI33ngj3bt3Tyh6feuD8cYbb7hz587t2bZt2zF+UuQe61ZtadBoNBqNRrNXGD16tDl16lQHCAL/Ai4DXDw9krQmUd3ZAA45+kjOuOZiuh3am4qyclwncQRRs+coW8MFA0ZwWKcDkrI1QFUb4SlTpnDddddF2xDXhv++c/bZZxvvvPPO98AxQMTfhjrf2WjBq9FoNBqNZm9i4IlcgHHAWP//Dl7UNymEENF6r6kZ6Zx44VkcP3oUwbRUKkvLdcOKRiLq4+3Vl5F9BicteJVvNxwOM3jwYH788ceo3aE2/MiwO3v2bOPYY489Swjx/p56efUtkEaj0Wg0mr1JbER3HDAaKKCOFgcppW9zMKgsK+fj515l8u/GsmL+D6RnZ2Calm5P3EgIIcgr3okkfpvh6vNIKUlNTeXee++tyzzyb3/7mwSuAxg3btwe+Vb0rY9Go9FoNJp9gaBK5B4CPA2ciCd46+bXFALDj/YapsHQ807npIvPI6ddLhUlpSB0tLehUPV4M1NSuWrIKDJTE9fjVagor+M4DB06lHnz5iW0NgghMAyDadOmRY4//viBQojFfnOTOmUq6givRqPRaDSafYHEE7sm8DNwOvCQ/55BHaK9xER7Xcdl1usf8uSN97Dgs5mkpKcRCAR0tLeBkHhthosryymsKKnTvCrKGwgE8CswJMSv5es888wzAeAqtag6rXhPZtBoNBqNRqNpYGJ9vSOBfwCHsifRXtjFG3rkScMYeeUFdOjRhfKSMpBebV/NnqN8vOf2P46juh2UdIQXdo3yHnfcccyZMyeZKK8bCASM6dOnbxsyZMjhQogdUkpRl+Q1HeHVaDQajUazr1G+Xgv4BBgGPOm/V7doL1XNDoQQLPrya566+V6+fvNjgikpBFKCuI6u29sQbCnJB+rwwbBrlPdPf/pTUvMYhmGEw2HnjTfeaA9cADB9+vQ6dRzRtzgajUaj0WiaErFNKU4F"
        "HgGO9P+uUyUH2DXae/iwQZx53aV0OKBbnby9SqRpPIQQhCJhDmzfhUsHn1anCC9URXld12XIkCHMmzcvbsUG//g7nTp1MmbMmDHvoIMOGgK4hmHIZD8XHeHVaDQajUbTlHCoSmj7DDgBeBAo88e5VNkfEhKN9hoGS76ex6Sb7+Obdz4lJS0NK2DVGu0VfiIcQNiO1GuHWiKGYVAaqiDi2J4grcO8qpycZVncdttt0XG1Ib1Oe+aWLVt44403BgHHCiGk67pJ3/xowavRaDQajaapIamK5pYAfwYGA2/jaZc62RxU+2HDMCgrLOb1iVN4+S//oGRnEWlZGVHRGytyI45NeTiEIQx65HZo4N1r3kgpMYVBaWUlpaEKNbJOyzBNEykl559/Pn369Em25bAzdepUUVBQ8BuAqVOnJr0+LXg1Go1Go9E0VVS01wKWAecBvwbW+OMEexjtXfTlbCbdfC9Lvp5LenYmEcemIhyiIhLClS7tMltxXO/+XH7s6Vx+7C8Y3ONQKiKeANZ4Ed7ycCVl4Yo9mj/aNCQ1lRtvvDGhLcK3O5gLFizgzTffPE9K2W7MmDGOlDIpL4X28Go0Go1Go2kOKKXpArnAH4FbgRSqhHHSatQwvRJmQghOuvQ8Lr7xKlqnZdEmLYte7TrTMSsXy/SemEs8z+k7P3zNgg3LyQim4u7nnl5VqeHCgSfRp2OPOvt4gagvuri4mL59+7Jx48a4Xl7Vbnj06NHma6+9dp0QYsq0adOsESNG2InWpW9TNBqNRqPRNAeUd9cE8oE7geOBL/xxBlXJbokX5rjR8mRfvPgmb/5lEid0PIQTDjqCrjntsPxH7p4oExjC4Ox+wxjc41DKwyGvVppuZkFRZRlQt0oNChXlzcnJ4aqrroqOqw1Vuuzzzz+Xs2fPvlBKKU488cSkorxa8Go0Go1Go2lOxNoc5uJVcrgGyMMTvpIkbQ7S9QStaZp88MEHDD72GGbNmgUCbF9cCSEQeFFe0/BE72mHDQYpCdkRjP28i1txRVm95lfH7vLLLyc7OztqO6kNwzDMgoIC8dFHHw0D+vi1eLXg1Wg0Go1G0+JQXdqUjnkWGABM8f+uU1Kb4ziYpsmaNWsYOXIk//nPf7BME9d1o4/dha+ppJQc16sflx59Gt1at6ciEiZi29GEN7GfuEXVXpaFK+u1HGVh6NGjB+eee66qyJBoNuezzz4LlJSUjK62ObWvp15bqdFoNBqNRrPvcPG73QJbgOvwWhQvxE9qS7Ybl+M4XiJWeTlXXHEF48ePx1Qe3hi/rhACV0p65HbgN8eczjn9htEhO5eQHaEiEsaWTrTpRXSIo8cEJJymKVMRCQH1SwpT1pFrrrkmKoBri/L674n58+fz3nvv/VJKGRBCJLQ1aMGr0Wg0Go2muePgaRoT+BQ4DhgHVEgphWEYbjK2AyW0DMNg3LhxXHPNNTiOE/WaKgy/EYVpGAzofjBXDj2DiwaeRP8uvclKSSdsR6iMhKmMhAnbEWzp7CqaVbQYcKRLxLGJuAnzrpoUEr8BhV+juD62DnVjMWTIEIYOHZowymsYhmHbtvz888/7AccCTJ06Na6mtfZ46zQajUaj0WiaDkqRmkA5MD4rK+vTzMzM/27ZsuUAQBqGIWqrAKBQwtQ0TZ599ll27tzJyy+/TGpq6i61YpXAk1JiGSaHdOjOIR26UxqqYGtJARsLtrG1pICiyjIqwiFCdoSIY+NKF9evY2saJimBFNKDKbhSUlBe0igHprEwEEQcG9t1sIw6NcDbDcdxsCyLq6++2vNRx8H/DJ1Zs2ZZq1ev/iUwM9Hym2f8XKPRaDQajaZ2xGGHHRZYunRpuLCwcOBf//rX2Y899pgViUSEaZpCZfsnwrIsbNvmzDPP5PXXX99N9MYS9fpWi3S60q0SvK7j+YKRGMLAMkws0yQjmMqC9Sv4cOm3pAWCzaLkmRCe2G2bmcOVQ84gxQqSVPZYLaiyZvn5+Rx++OHk5eUlajfsAsbkyZN/vvbaa48UQlRKKWu1sGhLg0aj0Wg0mpaGXLp0afiaa64JtGrVav5DDz30yFdffWV069bNVZHEZLBtG8uy+OCDD/jlL39JZWXlbvYGhfLrqpq9ru9LNYRBRkoauRnZdMhqTaecNnTOaUvH7FzaZubQKi2TgGmxaucmhNiz8l77BOlZMxzXxY4ejz3feiEEjuOQm5vLL3/5y+i42jAMw5BSMnv27EPwuvBBHL2tBa9Go9FoNJoWyZQpU+zXXnvNfOaZZx469thjl8yfP98888wzXduvqpCM71SJ3g8//JCLLrooKnZlLVFYgaiq2KBsD74Irj64UiKRFFeUsalwBwHDqnW5TQ+JISBi2zgJbCJ1WqqUXHrppUklrwHO7NmzWbVq1dkA48aNq1XXasGr0Wg0Go2mRSKEkKNHj+baa68tLy0tva1du3bu+++/L++55x6pRGcSJbCiovftt9/m+uuvj4qxZMWpEsHVB/XemvwtlIUrktqWpoIEDMOkJFTOpsLtVSPrgUpeO/rooxk8eHDc7m3+e2L16tV8+umnv5BSBsePH2/XVq2h+RxZjUaj0Wg0mjril6wys7KyPrNt+7+A+cADD7gvvvgiaWlptXpyq6NE77PPPst9992n2tzWb9v817U7t3oCrl5L2ze4UvL9ppUNtjxVHu6iiy4CEtsaXNfl66+/Pgzo64/Wglej0Wg0Gs1+iZRSCsuy/uy6br7jOOLSSy9133//fdq0aYPrutHoYjxUg4q//OUvvPLKK1iWVS/RK4TAcR02FW7HMiyajZvBR0pJ0LJYl7+N/PJiz8Ncz51QNx+jRo0iIyMjWhauJpStYeHChWLDhg2nAEyfPr1GbasFr0aj0Wg0mgZFSimklIaU0qxlMKSUe02DqIx+IcQ6wzAeNE3TCIfD8qSTTuLTTz+lc+fOUTEbDyll1Fd67bXX8sMPP2D6HdnqihKG20oLKaooxTAMZPNJWYtiCoOycAXLt24E6p90ZxgGUkp69+7N8ccfHx1XE+oYrl69mmnTpp2cmprKiBEjarwD0YJXo9FoNBrNHuOL21ghK4QQUgjhCiGcWgbXF6FUE8aNqUtcf/lPAouCwaAZiUTcAQMG8PHHH9O9e/fo4/QE+4sQguLiYi699FJKSkqi4+uCmnpj4Q4q7QhmPRo37EskYAiDlds3IWkYW4bjeI06Ro8enXBa0zSNyspK5s+fP7SioqIjfjS/+nRa8Go0Go1Go6kTMSLV8MVtrJCVUso2UspDpJSDpZQnSylPk1KeKaU8VUp5vJSyv5Syqz9/rDB2Y5fdkNvs12cVQogwcAvgBAIBGYlEZL9+/Xj//fdp165dUp5eZYFYvHgxd911V9x6sbVuj//qJXw1v8iuQkpJwDTZXLyDwvLSBrE1mKaJEIJf/OIXtGrVKq6twRe3ctGiRZl4HfagBn2rBa9Go9FoNJqExERyRYxIdaWUXaSUZ0gp75RSviGlXAR8D8wDvgU+Bz4B3sdr+/sVMAdYACyRUn4spbxfSnm2lLJrtWU3qPVBJbAJIb4C/g2YgUDAtW2bfv368dZbb5GZmZlU9QZlgXjqqaf45JNP6pzEJoTAlS55xTuxDLMZS14vwlsWqmTtzrwGWZ4SzZ06deLEE0/01hHf1uAsW7aMjz/+eCjAuHHjdlPHurWwRqPRaDSaWvEFpxBCOIDjjxsAnAyMAvoA7eIsIrYrgfCHFH+edsChwEh/mu1SyoXAm8CHQogNMdugvLgNsEvSAO4DzgQ6WZbl2rZtDBs2jBdffJFf/vKXUS9pvGilev+mm25i7ty5ZGdnxy2lFZ0PiUBQXFlOcWU5hjCaUf3d2lm5fRNHdTuoQZalbijOPPNM3n777VqnU+XJduzYweeff360/9TArt51TUd4NRqNRqPR7IaKrqqIq29BuElKORsvcvsIcAKeaHUA2391/UH6g+EPJlW6Q73nVpu3HXAa8DSwQEr5nJRysPL8"
        "qghzffbLF81CCJEH/AFPgEvLsohEIpx77rnce++92LadtLVh5cqVTJgwIXlrgy/DtpcUErIjSTXAaOpYhkleST4VkZDfca5+GIaBEIKTTz45YbUG1XVt1apVRwIdapymntuj0Wg0Go2mBaGsCzEis7+U8gk8C8ITwBAgQJVIlXhi1qJK1BpURXOrI2IGo9q8kioB3Ba4AvhaSvmKlPIo3+ogpZSJa4jFIcba8F/gNX/dTiAQwHEcxo0bx5lnnplU5Qbl+X3iiSdYunRpUqJXVWPYVlpA2I5giOYtx6SUWIZJQXkJ20sL1ch6LVNF2Hv27MngwV7n4DjlyQQgf/rpp4z169cPApg6deouB7V5H2GNRqPRaDQNhi8CpS8I+0spn8WL5t6EF321qYreKpEqVLmumoZEtoBqCKoEsBK/AeAC4Fsp5RNSyrZKsNYz2quy+W8B1vnrdZV/9JlnnqFLly4Jk9iUhaGyspKxY8d63twEglf49wE7S4sxhKA5J61FEeC4LuvytwINs0fKEz1q1CggsY9369atfPzxx0cCLFmyZJdzQwtejUaj0Wj2c2LKiTl+EtrjeEL3KiANcFzXlVJKy3Ecw7ZtEdtaVwiBYRg1DqqNrhLFjuNg23a09FQclPhVwjeIJ7xnSyl/4XuK2VPRG1ObN8/fTxtwDcOQruvSuXNnHnvssaQ8uaqc2RtvvMFLL72EZVnYth1v3QAUVpR6/t092YEmhgrZb8jfFv273sv0j9Pw4cOjxzTeZ1FaWspnn312cEpKCuPHj9/lrkMLXo1Go9Fo9mNi7AtSSnkV8B3wO3yh6ziOdBzHFB6YpollWVExK6Vkx44drF69mh9//JFFixbx/fff88MPP7B8+XK2bNlCKBSKimI1vyo9JaXEcZx4AjhW+NrAQcCHUsr7wSs3tqeVHGKsDZ8D9+BFlh3DMHAchwsuuICzzz476aYUAFdddRVffvllwi5sjutSVFnm2RlagOKV0vPxbi8tpCxc6ft4G6brWv/+/TnggAOA2m0NhmEYkUiEzZs3D6ysrEzBexIRnVhXadBoNBqNZj/Ej4wavug7GPg7cAaA67q267qmT3SeHTt28P3337NkyRJ+/PFHfvrpJzZv3kxpaSmVlZWEw+Ho43whBIFAgNTUVNLT02nTpg09e/bk4IMPpk+fPhx00EEcfvjhZGVl7SImlUhUgjoGgadbVOTuXuBwKeUVQogilWBX1+MQI3ofkVIeAlwhhLCFEBbA3/72N2bMmEFJSUncGrOqlFkoFOLCCy/kyy+/pG/fvruJZVWhoSxUQWU0wav5K16JxDRNCivK2FFaSEZux6q6HHuIsocEg0FOOOEEVq5cmageL6tXrz7gscceywJCuyxrzzdDo9FoNBpNcyRWHEoprwAmAq1d13UcxzECgUBUHyxdupTPPvuMjz76iEWLFrF169YG2QbTNOnYsSOHH344J510EsceeyxDhgwhGAxGp1G2iRqiq8rmYAGzgfOEENv2VPT6YkkNbwFnAbbjOJZpmjzxxBP87ne/S6rWrkpaO+SQQ5g5c+ZuzSyURWJDwTZenPNpXTe1SSOEoDIS5tRDB3Fc7364Uvoe5T3Htm0sy+LFF1/k17/+dbzPQALCNE150kknnfjZZ599NXbsWENZG7Tg1Wg0Go1mP8KPZjpSyizgH3iVELBt27EsywQoLCzknXfe4cUXX+Tbb7+lrKwsOr+yNahIZ/XXmlBROeXn9de323QHHXQQp59+OqeddhonnHAC2dnZ0WUr0Vgtwmfjid6FwOn1FL0GnmhKBV4GznNd15ZSWoZhcNppp/H5558nJXrVNCNHjuSDDz4AqiLWSgQuy1vH1IXTCRhWi4jwAhhCUBEJ06djDy4ceFJS/udEqM991apV9O/fn/Ly8niRdseyLLN79+63rF69+p/Dhw+3ZsyYYYMWvBqNRqPR7DfEiN0DgZeAYxzHcUzTNACxefNm/v3vf/Ovf/2LdevWReezLCsqOv3l1HtblPhVQ3UB3KVLF8455xwuueQShg4dGh2v6rHGZOwr0TsfOFUIUVAf0avq/QLPA5c6jmMbhmGtXbuWgQMHUlhYmFQlBpVk9cc//pGHH344GqlUgnf++uW8s/hr0gNB3BbQdAK8z9R2HFqnZ3LV0DNJDaTU19WwC0ceeSTff/99vNJvNmAJIf4ppbwF77ywQSetaTQajUazXxAjdk/Ga+97TCQSsU3TNEtKSsRDDz3EgAEDuPfee1m3bh2maUYTy2KrKjRUR7DqVRuUiFUJcZs2beKpp57i+OOPZ+jQoUyePJn8/HxM04wKHl/0KFEzEPiflDIFr1RanXWWamfs1/u9DJhimqbluq7ds2dP+eSTTybVdhi8CLZpmjzyyCO88sorMUls3vGrtJXfueXEHqWUmIZBQXkJO8uK1ch6L1eJ22OOOQaovTwZ/sGUUvb1/x8NxWvBq9FoNBpNCydG7F4CfAB0sm3bCQQC1vz58znppJO4++672bp1a1RwJqic0BjbiOu62LYdfYxtmiau6/LNN99www030L9/f+666y5+/vnnaNkzx3FwXVeJ3pHAY37Jsj2t3OD6zTcMIcR1wETTNC3btt2LL75Y3nTTTVExmwjXdRFCcN1117FixYro/gBE7AgtoMHabgghsF2XTYU7gIYpQKGO2dFHH+0ts/ZzUn3m/YFsYtLmtODVaDQajaYFI6W0fLF7I/CS67opgGtZlvn4449z3HHHMW/ePCzLikZzk2qP28io6G9sObNNmzbx17/+lUGDBnHttdeyZMmSaMTXcRzLdV0buFFK+VtVfWFP1i2EkHiNKQwhxB+A8b6/WT766KPymGOOSbpUmWEYFBUVcfXVV/u2DU/lhh3bb0DRMuwM1VlfsK3BlqV8wEcccUQ0Up7AG9wWOEzNDlrwajQajUbTYvEju7aU8g/Ak7Ztu4ZhyIKCAmPMmDHceuutVFZWYpomtm3vtWhuXYi1PijLQ2lpKc888wyDBw/mxhtvZPny5Ur4GrZtu67r/l1K2ccXvXsa6VWi1xRCjAPuAIzU1FT5f//3f7JVq1YJu7ABUWE8Y8YMHnnkESxfJDty399UNBaWYbC1JJ+QHcEQot6SXh3jvn370qlTp3iTxtoYjlSzR//RaDQajUbTsoixMfwBeCQSibiWZYnly5eLU045halTp0Y9uomqDtRGbCMK1Uwidqg+viEy9pXf17IsKioqeOqppxg0aBB/+MMfyMvLMyzLkoZh5EQikaeklAHqYZL1Ra/rR8kfw4seGwcffDBPP/20VBHcRPvlui6maXL//fczb/48wLM0tCD7bhTPx2uSX1bCjtIiNbJey1RVGVJTUzn88MOBuD5etbJDY0dqwavRaDQaTQsjRuz+DnjE9+uKzz77TAwfPpwFCxZEHw3XJaqrBK56lK+6pNm2HU1six2qj1frik1O28P920X4lpSUMHHiRI4++mimTJliuq5rBwKBEysrK6+uj7XB32cJOL7ofcp13asikQgXXHABt912m0zGz6v2OxQKcestt4ErMQ2rpboZfB+vw/oCr2ZzQ/p4+/fvH11Hbav3Xw/2X53YkRqNRqPRaFoAvjCzpZSXA/+xbdu1LEu89NJL4oorriASiSRVSzYWJUyre3u7du3KwQcfTPfu3Wnfvj2tW7cmJSUlGpErLi5m27Zt5OXlkZeXx8qVK9m2bXdvpxKMqtFEXVFCXJU2GzJkiHzooYcYPnz4DqC/EGK7lFLuSamyWNSxtW37Minlv13XtU477TQ5Y8YMUZf6vP/3n+c56LQhvDtnOtnpGS2mLJmiMerxqrJuL7zwApdffnm8c9jFC+iuBfoAldAi8wM1Go1Go9k/iYnsnu667rtSStM0TfHUU0+JG2+8ESBeDdPdUFUFlAht1aoVp5xyCieccALDhw/ngAMOiDaHSITruhQVFbFmzRoWLFjAvHnzmDVrFkuXLt1F5KqmFnuSOKcS3Hwh"
        "5Nx5553mdddd9/devXr9/qqrrgo888wzDlWtifcIJXpDodDoYDD40tq1awNHH3203LFjh5GoPq9hGEgp6da5CxP+O4VVFTsJUH+Pa1NDIHCkS3owhauHjSIrJT3aUnlPUV7ob7/9luOOOy6auFbDDZKqzBABDgHWoAWvRqPRaDQtg5imCYcDM1zXzTUMQz788MPGnXfeWWuUtiZUMwg17dChQ7nssss499xz6dix427T+6XBao3iqRJi1amsrGTlypV88sknvP/++8yaNSsapa0mXuuEL+olILt06WILIa7buHHjf/y3TWLqs+4JSvSWl5ePSUtLe/mDDz4wRo0aJUzTFImi1OqGY8SYsxh18+WUF5dimC3PYSqEIBQJc+HAkzi0Y496txlWUeKCggIOPvhgduzYEa/jmhK9ZwIfAqYWvBqNRqPRNHNi2uK2cV13lmEYhwDOxIkTzT/84Q+7RWrjERsBHj58OHfccQdnnHHGLoK5epvfZB5Xx7YgVuK4uvd12bJlvPHGG7zyyissWbIkOr4u26+IEcxK/LwM/BHYjCd6XephL1Wit6Ki4qrU1NRn//SnPzmPPPKIYZqmiCfS1bEKpqXwu8kP0bZrRyKhCMJoWZLMszWEGNyjD6P6Dqm34I3lkEMOYfny5Qk7ruFV1ngMCLS8WwqNRqPRaPYj/I5iSkn8xxe79uTJk+ssdtW0PXv25OWXX2b69OmMGjUKwzCiZctiE85UJDgZ1LRq/ljrgorq9unThz//+c/MmzePt956i7POOgsgmvCmagUneVxi67W6wCXA18C5eBFeiSd89wjfJ22lpaX9y3Xdvzz88MPmoEGDXFU+Ld52CSEIlVcy642PMANBZP1cFk0SCViGxdqdeVTa4QYTuwBt27YFkrrR6q02RwtejUaj0WiaN6bfWexOYBRgv/TSS9YNN9wQjYAlK3Ydx+Hss89m9uzZXHzxxVHRWFexmSyxAjhW/KampnLuuefy7rvv8vXXX3PJJZcQCASiojuZLmcK/4bAwBO5BwBvAU8AGf44qx674Pg+6fuA91555RUzNTXVUfsWZ5tAwPxPvyJvzXoCKSlNsgZyfZBSYpkmO8qKWLczD6DeyXnqGLVp0ybRpOrgd/dfHS14NRqNRqNppsQ0lhjhOM54wJk+fbp59dVXRxOkEgkpZS1wHIdrrrmGN998k44dO0bLfjVE/dxkqC5+ldAeOnQoL730EnPmzOHXv/511Nerpq8DysbgAjcBXwJH4T3+NtiDylWqOYUQguXLl1/fu3fvHRMmTDBc13UTRXkNYRCurGThZzM9wdsEuts1BlK6LM1b10DL8s5lFeGNg/ose+Dd0OgIr0aj0Wg0zREppTFu3DgppWzvOM5/TNMMrFixQlx00UWisrJSTRN3Gcpm4DgODz30EFOmTIlGhS2rPoHP+hErtJVn+Mgjj+SFF15g1qxZjBo1KhoRrqMgN6iK9h4NzACupcrPW2eLgxDCdV3XPOSQQzYBE2677TZx5JFHyoTWBv914eezKNlZgBnYd8e7sZASAqbFmh1bKKksr3fXteqCN4lavN2ATNCNJzQajUajaa6I8ePHu67rTjJNs0dBQYEzevRoIy8vL+rFjTtzTCWGf/zjH9x5553Righ72hCiMVAVHlR74SFDhvDee+/x/vvvc9RRR0UjwXWxOVBVqSELeBp4FWjvj9sTX68rpRQ//fTTv4B1kydPNg3DiPsBSNdFGIL8vG0s+3YBKWlpuHvY8a7pIrEMk8LKUr7ftMobUw9bQx0EryIL6AJa8Go0Go1G0+yIqbd7gWEYvwKc66+/3vz++++jHdTiESt2J0+ezC233IJt20m1yd1XGIaxSwLemWeeyezZs3nwwQfJzMxERVTrsP0mXqDVBcYAM4GT8ESvigQnhW9tMPr06VNi2/YLxx57LOeff76rItBx5gMBP0z/BidiI5rQjUZDIaUkYJos2LCcikjIS16rp5c3Nzc3uuxaEHifrYHn29aCV6PRaDSa5oSfhOVKKTvYtv13QD7xxBPi1VdfxbKsaMWD2lDeV9d1mTRpEtddd120i1VTFbuxKFHrOA6pqancfffdfP3115x++ulRMVyHaK9KaLPxWtF+hpf8p7y+dYn2SimlKCoqegUI/+Uvf7HS0tJkvPrEriNBwqpFS9mxKQ8rEGh5yWtAwLDYWVbE4s2ro+PqQ+vWrYGENaXVm91BC16NRqPRaJobhhBCRiKRxyzL6vTdd9+5f/rTn4xk2wWrpK8JEyZwww03RMVuc0OVNbNtm/79+/PRRx/x9NNP06pVq2i0tw7WDAtPIAngIeBtoCN1sDiotsVt27Zd5jjOnEMPPZRf/vKXrirlVjMSYRiEKypZtfBHgqktr1oDeJFY0zBZuH4FkapScXVGzacEb4KWxUrwdgQteDUajUajaTYoK0MkEvmFaZqXlJSUOFdddZVZUVGRVEUGZXe4/fbbueuuu6LtWpsrQggsy4omtl177bXMmjWLkSNH1tggIwFKEznAOcB0YDB1K11mAti2/Q7AzTffLNX21b4P3uvKRUtxbLsezXebLhIveW1LyU6Wb9sA7FmJMvU5ZmVlRW944k3uv3YALXg1Go1Go2kW+FYGKaVMAx4yDIPx48fz448/kkhUgRcRtW2bc845h0cffXRPPK9NFhXNdRyHww8/nI8//pj333+fYcOG1dXmIPBEqw0cAnwBXOz/bZK4dJkLkJKS8qHruuGjjz7aGjJkiIy3ful6om39kp8pKy7BsKx6P/JvskjBgg3LvchsPaR9ZmYmwWAw0WRa8Go0Go1G0wwxhBBuJBK51rKsI2bMmOH885//NJOxMigx2K9fP1544QWgKnGtJVE9qW3mzJk888wzdO/eva61ey2qqji8DPzZ/1t5fmtECOFKKYUQYqnruvOllIwZMyZ+tQY/Slm4bSfb12/GCljQAmvySikJWhbr8reypXgnXu5a3aS9Ol+TFLyK9kALTAfUaDQajaaFoRLVysrKOpumeU95ebm86aabRCQSUe/XOq8QAiklrVu35tVXXyUnJyf6qL8lEpvUJoTg6quvZs6cOdx0003R2r1JepZjqzj8BXicqnq98Q6eCSCEeE8IwRlnnCHT09Nj2xzvhvCbhGxauRYr2PIS1xSGEEScCIs2rqzXcrKzswkEAglX5792AFJa5tmu0Wg0Gk3LQgghpGVZdxmG0fbJJ590f/zxx6QS1VTHtUmTJtGnTx9s227Wvt1kUfvoOA4dOnTgiSeeYPr06Rx55JGqi5wkccEA4Q828Dvgeao6ttWeieat/xPXde1evXqZhx12mITa68aq8Ts2bPHmbmGRd4VXoszi560bKAtXRm/GkkUdp/T0dFJSUnYZF4c2QKYWvBqNRqPRNGGklAYgt27d2tuyrN9u2LBBPvTQQ4aqoxsPJYivvfZaLrroomZbkaE+qOQmx3EYPnw4M2bM4LrrrosIH6qy+WtD4FkcbOA3eE0qgvjSdLeJhVB3IItc110CiGHDhrkQp6GHL/ryt2wjHAq1yHq84LeyM0yKKkpZsW1jdFxdsSyLrKysZCdPAdq1zCOq0Wg0Gk0LQggh27Zt+3vDMDLuvvtut7CwUKjIbW0o3+5hhx3GY4891qJtDIlQrYodxyE7O5vJkyfb55xzzv1Syq2maapWw4lQovd84AWqor+7iV6/moZr2/b7UkoOPfTQuLpOfY47N+cRCYX95gx13MlmxpIta5EyqQjtbgghyM7OTnZyC2i9f575Go1Go9E0A1R0d9u2bQcbhnH5/Pnz5SuvvGKoxhG1oRLShBBMmjSJjIwM4teD3T/wk9pcKWXam2+++SEwHFhGVWWGRCjReyEwiSprQ42qLRKJfCyEcHv06GECcX28AEU78glXVHqe3haqeKWUBKwA6wu2srO8CL/0SJ3mB6IR3jjHU71hAa327zNfo9FoNJomjhBCtmnT5jYgfezYsa5t2yKR91FFd2+8"
        "8UZOPPHEZl9vtyExDEMKISgvLz8a+Llz586nADOoErOJUBUcrsPryrZbcwpla1i+fPl3wPJevXoJ0zTd2j4zNdaJ2FSWliOMlunhVZhCUBkJs7wetoY6RHgFWvBqNBqNRtM0UdHd9evXdzEM48LZs2fLjz/+2FBitjaUt7dHjx488MADxGttuz+Tnp7eETA2b968GTgTeIfkRa+yQTzoz6vq9EaRUpqDBg2KAO+2adMG0zRrD8n7QtixHSrKyrxIfMsM8EYRAlZs24grXc/CkSTqpiE1NTWZydUx14JXo9FoNJomihBCyM6dO/8GaHXfffe5juOIROJVRX8feOABcnJytJWhdloD7v/+978gUAZcALxJcqJXeXcN4F9AF2qv3PBuaWmpa1lWwhC76zhUlJb5SWstV/G6UhIwA2wu2kF+eQlQ95q8SVZpUAvVglej0Wg0mqaGqru7bNmyLNM0r1i4cCHTp08XiaK7ytt7wgkncNFFF2krQ3xaAYwePVoJ1RBwKfAZVbaFeKgob0c8P68SVwJ2qdbwXV5e3k+BQCCZihA4Ya+9cMuVux6Gb2tYvX0LUPf9TTLCqxark9Y0Go1Go2mCGEII2aNHj7OA3g8//LDjOI6RKLorpUQIwYQJE7TQTUwr/1U1lzCACmAMsBjPopBI9KppzgEuolqUd/To0aYQwv7888/f8qPsyVSD2K9Ys3MzkHy1BhUJVhHeBKiF6jq8Go1Go9E0QSRAWlrabzZs2MDbb78d9ebWhqo3e8455zBs2DAd3U1M9VZdLp6ALQRGAzvwBFMydXolMAHI9qcXAFOnThUATz755OrS0lIAEbcrHhC0ArhS1lz2oQXhNaEw2VS0w2tCAXWqTJFkhDc6uRa8Go1Go9E0IaSUhhDC3bx5cx/ghCeffFKGQiEjXt1dJYYDgQB//vOf9+4GN19qOpgOnp3hZ+AGvGhtoo5sBp7IPQC4nhpaD2/dujVHtYGOh2WYdMxp0+LtDArTMCmpLCevON8bkcSO1zFpTaEFr0aj0Wg0TQwB0KlTp1+GQqHUF1980SVBZFCJ4XPPPZeBAwc2m+iu6oBm2za2beO6Lq7r7vJ3XZOZGgBVcWEq8BxVrYTjoaK8NwM5eMI5NvOsa9yZ/cf5UsCAnoeSZgVx9/5+73UE4Loua3cqH2/y+5xk0pp6UwtejUaj0WiaCipZ7cMPP0wBxnz00Uds2bJF+A0Tap1PlR679dZb99am1gslalUHNMuysCwLwzAwDGOXv4UQUfHbwMRTV6pt8B+BTVRFcWtDvd8Fz8sbOw6gs/8aX50ZBod070XXnLZEHLtO5bqaLYZgc/FOJBJDJC9L6xjhTdu/GmprNBqNRtO0MYQQTnl5+SDgsCeeeGK3x+PVUS1zTznlFIYMGYLruk06uquiz4ZhUFBQwLx585g5cyarVq2iuLgYwzDIzc3l0EMP5bjjjuPII48kIyMjOq8SwQ1Amf9a08KUnzcfGItXeiwZxS3xKj08w64Jat3jrCuKaVm0b9MWl3QWrv25xVsbXCkJGBY7SoooD4fICKYikYgkHMzBYBBIupxZqha8Go1Go9E0MdLS0s7btGmTOXPmTAcw40U31QX/5ptvRggRFYVNESV2N23axOOPP87UqVNZu3Zt3HkOO+wwfvWrX3HllVfSvXv3XZZTT/ITvK8qLrwI3AQcSQ1d1WJQ448F+gI/+H+nAD38/8dVculpaaRmpHNAMJvMlDQijt3im4aYQlASKmdnWREZwdSq2HoC6niOa0uDRqPRaDRNAWVnmDZtWirwizfffJNIJCJU9YWaUHV3Dz30UE477TSklE02umvbNqZp8tVXXzFo0CAeffRR1q5du4utwTTNXf4vhGDp0qXcf//9DBgwgLFjx1JWVhaNateTAv+1thChkl5h4LEkl6kE8ciYcZ2oKoFWI0rUdu3alZTUFNIDQTrntCXiOC1e8OInXG4p8u4/ko1q1/G4NNFbQI1Go9Fo9j8MIYQcOnToYcDBL730kiRRGSv/on/JJZeQmpqK00QFkuM4WJbF9OnTGTVqFHl5eViWFe0KpxLXHMfZ5f+qS5xpmuzcuZP777+fwYMHM23aNJSvuR5JbduTmEaVGHsDWE1yCWwAJ1FlRekJpMebWH1mPXr0iN6wdG3VFkc6ST3eb84IPGvD9tLCOs1X1xs7LXg1Go1Go2lCBIPB49asWWMtWLDAFULUKniVfSEzM5OLL74YqPNj3r2C8hQvXLiQ8847j5KSEgzDwLbtpMSq67pRIW9ZFsuWLeP000/n8ccfj1anqKPoVQpyVRLTKg91BfA/tUlxplcfwCCgjf//A/112tTysD5W8Co6ZOcSMCykbPBkvSaFxDtvC8qLvRucJG/Y6npj1/S+GRqNRqPR7N+c/Mknn2DbNvFq76rkreHDh9OrVy9c121ygldt+/bt27ngggsoLCwkUcWJeMtStohwOMytt97KHXfcsSeiVymldXVYvQDeASJ4Ud7aVqaW3RY4yP//Icmu5IADDvD+40py07MIWFaLT1xDSkxhUFhRRqXt1SquS3myJKlD/QeNRqPRaDSNgpRSCCGcLVu2ZACHvfPOO5AgdUeJvF/96lfRv5sSUsqoCL/22mtZsWIFlmXV23uror2mafLYY49x5ZVX1kX0Kl9uEV4nNTUu7ir9aRYCK6mquVsbSs0f6b/29V9r/TzVMenTp090yqzUdMw4NzwtBQmYhkFpqIJQJFQ1MtF8dTwuWvBqNBqNRrPvEQAdO3bsvXbt2m7ffvutBERtkVDVWa1t27b84he/iI5rSigrw3/+8x/eeustLMvCtu0GWbYS05Zl8e9//5vf/e530QS+RLP6rxuAvGrj4mHiWRK+SmIetRHKnxBX8Cofc2pqKn379lUjMQ3TszTQ8tsMCyDi2BRVlic9T12fEmjBq9FoNBrNvkdpmoOXLFmSUlhY6JqmWat/VyXsnHDCCXTo0KHJlSJTkd2tW7dy9913RwV6MiQr3FWym2VZPPHEE9x///2YpplIVKsDukwIUSmlNIUQyQhetVFzqi0nHu3w6u92qLaMXRfs7+8BBxxA586dq8YDhtHSpa6PL/qLKkqB5A6uFrwajUaj0TRfDvv8888TTqQu9mecccaeJG01OlJKhBA8/PDD5OXlJRV9VV3W1LzJZOHHit6xY8fy5ptvJmubmJf83nir8l9X+v9PxsebDQwGLKqqPew+sS94+/bti2VZ0ePkSJewbSMQLd/Hi1epoSxcmfz0dRO8YS14NRqNRqPZx0ydOhUA27YPmzPHCyLGq87gui4ZGRmcdNJJCCGaXHTXNE3WrVvHc889l1R0Vwli13XJysqKClkhRMKIr7I3CCG48sorWbZsWbzEOLWwuXXcLfVhbAZKiO+vVu9lAkdXm3/3if39GzRoEFDl5y0NVWK7DjQxq0pjIPA+x7KQEryJJb6K5Cf5RKCi6XxDNBqNRqPZTxkzZoxrWRbz5s3r+dNPPwG1C14lbgcMGECPHj2iEdGmgtruKVOmRFsFx4tAK7F7yCGH8MYbbzBnzhymTZvGmDFjovMl2j9loSgsLOTqq6+Oit1q61Wd0zZR1QUt2TChWtAOqloSJ1JlHYAR/v9rje4qgTtkyBBvnG9j2F5SQNi2ky7T1RIojyjBm3ifQ6FQXRZdqQWvRqPRaDT7EL/DmoxEIhlz587Nzc/PJ179XfCE0tChQzEMo0k1m1Cd3kpKSnjppZeA+I+eldg9+OCDmTZtGueffz6HHnooJ554Iq+++iqTJk0CSCrSq9oNf/3110ycOLEmG4X0h++EEDvr4N+NpQxIpLTUhraiqiRZrXpLSkm7du048sgjvZn9AlpbSwpwXLvFN56IIrzENf+/CVGCN4GdR72pBa9Go9FoNPsYdX1vM3/+/NYAIo66U93FTjrpJPxpG38Lk0QJzE8++YQNGzbEje6q7TYMg2eeeYZOnToRDoejjSYcx+GGG27gueee"
        "i9okkrE3GIbB+PHjWb58eU3WBgF8vAe7pnYiglemLBlygdR4E6ho/eDBg8nOzvZuGPxxmwp3YBpmY9SkbZIIBBHlvY7zMatzIBwO12XxIS14NRqNRqNpGuQuW7YsJ94EqoRV27ZtOeKII6LjmgpqW95+++2E3mIVgT333HMZPnw4juMQDAajrYQNwyASifDb3/6WP//5z9GmE/FQXt7y8nLuvvtuIBoBVB3TSgCVFVgXJakOsknihP/YpLVg3Am94+Ucd9xxrmqsAVAWrmRrST6WYTa5hMTGQBVHtl1P8CYT1a6sTD7BDR3h1Wg0Go1m3zJ16lQB8O9//ztn586dFuD6NofdUAKyb9++tGvXLhrRbAqobSktLWXGjBnRxLOaUMJdCMF1111Xo6hTrYRt22b8+PGcddZZSYleVaLtrbfe4vPPP1elylSYd5oQYo2U0hBC7EnP3gwgZQ/mqwm102Xdu3d/WgiB8D/LzYU7KAuH9iv/rp+5lngy/5iUlZUlmHIXdNKaRqPRaDT7kjFjxgDw9NNPp23ZsgVAJorq9evXL+rfbSoo68CCBQvYuHEjkLjSxIEHHsjw4cNrLUOmosSGYfDcc8/Ru3fvpGoOq+VPmDBBRX0FnqR6SU1Sx92L2k7wRO+eLKM60vdfb1mxYsVdwGLLNCXgrt65hYgTifp59wskdapIUV7uNalI0sNbuB8dSY1Go9Fomi6bNm3qqS7itaEu7sceeyzQtOwMatu++eYbgLiRWCVYR44cSTAYjCvclbBv164dL774IsGg5xKIt+9KFM+YMYNp06a5pmkK27Y3AJ/5k9Q1uqtW1gnIquO8teFKKWUgEPh6/PjxxWVlZY8DIuLYrNmxhYBp7Rd2hrqiPvdE3xUfpXO14NVoNBqNpilQWVnZN9E0qvxW//79gaYleJWIXbx4MRB/25SQO/7443f5uzZM08RxHIYMGcJjjz0WTWKLh4ry/v3vf1cW0eeFEIV7WJ1B7cwBeCLKpv4RXgMQtm1/IqUUX3755f+AZSV2pbGjrNCx9quENS8Ua/nnUDJ7naSlQX1GWvBqNBqNRtMUKCkp6R3vfSUoO3fuTKdOnYA9F7wSGe3Q5sqq/+9pRFH5dx3HYeXKlUDt5chU7dm0tDT69esHkJQP2TAMbNvmpptu4le/+lVCP6+/fjljxgxj+fLl2y3L+pcqAVfX/YuZ56g9mLe25RlAnuu6nwSsgDz77LPLgUdW7NhEpR3Zv/y7eOekZVj+H/FL8gEUFRX5k8b9ONVBLNCCV6PRaDSafYfAK3MViEQiXWLG7T6hf6Hv0aMHubm5dRKnEnYRtAIRrW1riKr/q3VEhXAdtWFZWRnr1q2LLiPefnTo0IEDDzwQSE7wKj+vlJInnniCbt26RSPeNe6zt363tLRUXHPNNdOFEOuFEHuarKbmOc5/ra9+Uh6O94Ei+7hhlpSIqdOnvv7l4nmbs9IyTMd194/wrkJCwL+BSWbHCwsLk1gi4H12OsKr0Wg0Gs0+RInbtv5Q+4S+UDzwwAOjUdKEdWn9SK5g1+YNxZVl5BXnsz5/K2t35rEufytbinZSWFGK61dPMIRAIOoU+S0uLmbnzp3xd9jfht69exMIBKJ1hZNBlTLr2LEjEyZMSHq+GTNmdAMsPPFT19CpgSeeegHKdlLf8KsKTb8OCEpnCEMgxzx7R/tv1y1ZbBkme2C7aPakBVTZ4vi7btt2NMKbBDZQaNVjuzQajUaj0TQMbfEqAEAtYkqJu4MOOiipBUZbDguojIRZX7CVlds3saVoJyWhcsrDISKOjStdBIKAZZEeSCEjJZ32ma3o3a4z3Vt3ICfNK0ogkSBrtlGode3cuTNaSzaRGO3evXtUtKsyZSpiG0/IK9F78cUX8/jjjzNv3ryox7emyQEphBggpewDLKZKwCaLgSeUTwYy8aKz8Q3E8VEtjlcDXwGS+TgSaNW579A5635e1q9TrwE5mVntwuGwjNeEpKXgFWgQZKak+WNq3mV1npWVlUXr8CZx0xMBdmrBq9FoNBrNvkNd2TsAAeKIKXVh79mzpzdjLTpIiQIhBIUVpSzYsJylW9axo6wQKcEQAtMwMIRB0ApEE4aklJSFQxRXlrOpcDsLN64gOy2D3m07M7DbwXRr3R4EuFLW6i8tLS1NKEDUdnfu3DkqVIuKisjJyYl6clWVhZr2USWjmabJ5Zdfzrx58+KuDnCklEFgGJ7grauAVEr6gjrOVxtK8L4PVOB97pG2t47sZFpWn/ySovxlW9fNGZrd/8wGWl+zwBCCzJS4jemiFBUV1aXTWimwU1saNBqNRqPZ93RPNIESkr169ap1GmVHCNsRvlr5Pc/N/oDpKxZRUF5CihUkLRAkaAUwhRldphtjWTCFIGgFSA1401aEQyzcsIIXvv2Y1xZMI684Pyp2Y4Wt+n8oFEp6h03T5NFHH+Woo45iwIABDBs2jIkTJ7Jly5ZoG2HHcXZJfosm2vnth1V5tiStDUOS3riYzfRf++P5d1WyWX1Qy3wTgOHDvVZwgZTBtpSpGamp1uzVSxbnFxfuDAaDImFR5paAfxPVKi0TqP2ORB2K/Px8KioqEi7Vf90GVOgIr0aj0Wg0+57O8d5Uj/xTU1Np3759dFwsEk80bCzYxodLvmNj4XZSrADpgRSk9MVigo3wlJeM/t8UAisQxJWSJVvWsmrHJo454DCO790/Wic2djuSST5TloeJEyfuUkt19erVzJ49m8cee4xrrrmGG264gQ4dOkTfj01QCwQCACxdurTGY1ELh6pFUVUJKxkkcA1ehzUbzwu8p6jo7k/At4Bgxgw7Z+w5rUS5c6R0jVDQClg7SgvLft664bsh2f3OqH6MWyISCJoBstMy4k8XI3gjkUgyiwVP8EZ0hFej0Wg0mn1Ph8STQJs2bcjMzNxtvJeYJpi/fjn/N+dTthTnkx5MxRBij6otRJeLFzUGSAsEkRKmL1/I/835lG0lBZ69IEaQqaYQyVBeXh7tomYYBqZpYpomeXl53H///QwYMIB77rknWtdXVWgoKChg+fLlTJkyhdtuuw2ovQRaNVRIMFmxq7y7PYFL/Hnq490lZr2vAyEGDrQArErneISRDa5tO47MSk1P+WbNksX5xYX5LT3KKxA4rkt2agZpAf/8iWPXAU/wQtU5kYCtUL+7FI1Go9FoNA1DXMGrIrytW7cmKysrOg6qPLszVnzPl8sXkGJZpFgWrtyT6lu1o4RvejCNjQXbeOG7Tzir31AO7dAdxxecrVq1im5rIpQXtzrKu7t582YmTJjA3/72Nw466CCysrIoKiqitLSUgoICiouLo/MksT4J/KxWncTuqulc4C6gFfVPVlOCOQK8AsD8+XbXsaNzK8orB0rhhsDrJRwwLXNbSUHp8u0bvz02u+8Z4XC45UZ5BTjSJSs1jRQrqEbFJVbwxrnZiY3w1tuHotFoNBqNpv60j/emEju5ubmkpaVFBZ6Krk5fsYgvfp5PaiCIwIiK08bAlS4pVoCQHWHqwuks2LAc0zSRUpKbm0tqauou21wbtYlU5c9VUd/KykoWL17M7NmzWbJkCevWraO4uDjq800C4Q/T6rCbBp7A7Q9cRpUVoT64eCJsBrCEsV61iPLy8pOFEJnIaHIcjuu6WanpKbNX/bi4oLiooCVHeQXeZ57t+3eTOXfz8vLqsoqtoAWvRqPRaDT7EhWeim9e9FHRXdd1vewpIZi77iemLV9AWjAF6mFfqAuulJiGgSkM3l08mzlrlyGEICMzo1aPcZ3X4Qtf1XDCNM2o/UEltCUZ2RV4oufzmHHxUALZBP4BpMYspz6o5f4XAYzHzbnzzF5gDJBSVqrorkJFeX/etuFbdUPRElElyVqn727VqY7ycG/atCmZRavjuSn2D41Go9FoNHsX5SW18ESVGrf7hL54zM7OBvwELiFYvWMLnyybQ4oVTCoprSFRVoqgafHhkm9ZtHElWZlZdO/efZdtboj1KPHr"
        "um6dGlXgRWkF8DSwA0/EJvJ6mP58NwIjqL+VAaoixFuBd3ARjB1rWLYxEoFZ012K47puTlpm6jdrliwuKClq0RUbDCHIzchOOJ06pzZv3gwktLIojbsu9g+NRqPRaDT7hjR/qBV1YW/VqhXglfQqDVfy/o+zcWtpBrE3UKI3YFm8vWgm20Il9DvMa0bWBDynDt7NxBLg71QlocXDxKvE0B+4n4axMkBVVPklhMhHIFuVzxsmDNlbSFGJ2P1GxxBChB3b2bRtY/7a/K3f+zWKW5zglVJiCIPcdN+bHmfaOghe9UYYWAs6aU2j0Wg0mn1NKlUR3rioCK9hGHz583x2lBWRHkxNtkpBoyClxBQGYSTvL/mG9j27AklXTmgsVPmwIjwPbhGJO6wp324O8F//dU9aEVdH1e4NZWRk/Ke0tNTocc/5HSoc51TpirABRqzgFUKIiGM7JaHK0FFdD+ox+NjTB7fLzOkVCoUQQrS4QKUrJRnBVLJS0r0RtRxtdXNVWVlZFw/vDn/Qglej0Wg0mn2EsjQEqLoexxVXQT8hbMXWDSzauJK0QHBfC0vAEy0pVpBthfmUtg5EE5GSxTTNuloVat0UqmwiecAYYCFVNoXaEDGv/wEOp2GsDACuYRimlPKNsrKyJUIIjJtGnJoSsEyBqBS+d1fZFSojYbt9VuvMXw048bSebTsNTAmmmJFIpLbWyc0az4tt0zoji/RgijcuQVvhzZs3Ryt0JIjwCjw7QyVowavRaDQazb7GJElhlZbiiYKvVn7vX+zr0j+hcZFIcFwyOuSS27kDOzdvTapEmWEYUTGnWgvXUfwqkSuosh98BPwOWElyYldFdycD51H/BhNVCxfCcF23FPirlPLQO9+cfP76kp1n5aZlpQVMM8MyzKArpeO6rl0RCZUUVpTu7JHb8bCOuW27VFRUUFFR4XqL2fcekYZGAI7rkpuejSGMuG2r1fmwceNGysrKEi1anTzr8G0pWvBqNBqNRrNvSVrwZmdmkVdRxNqdeWSkpjVq+bG6IoTAsW2yc1vRs38fT/AaAunUvI2qhuoxxxzDwIEDee6552LbxUqqEs7U37usjiqhGvuYfyHwJPBv/+9EYlfZHBxgInAdDSh2DcOQruuKX5xxRtGHH3wwefXOzUd0bN8xs31uWyS7RyiFEAghsG2b8vJyVwhhtEQbg0JVaGiXmZN42hjBK6XENM14Ue9YwQuQRA9AjUaj0Wg0jYlJkolREsmc9T/5/296SH84bOhA72+39q2UUmIYBvPnz+ess85i5cqVPPDAAwwfPlzm5uaK1NRUyzAMdTNgVRtij9ka4P+Ac4Bj8cRubNS2NlTFBgE8AdxOw9kYVIKVyMjI4Nmnp3RxYdh7P8zOLCouciorQ25FRYUMhUK7DJWVlW5FRYUbiURkSxa6USSYwqB9VmsgObP0+vXrgYRtrNWbK9QIHeHVaDQajWbfE/dar6Jb63ZuReZvJWgFmmRdVmEYhCtD9DriMHI7tSd/y7ZabQ3KkxkOh7n00kv5/vvvueeee7jnnnvEmjVrNr333ntfPProo6s2btyYA7TBS+yz8VoEb8ITut/jZeEXxixaCdl4JmIV+c3C8+z+kuTFblI1eU3TxLZtxo8fT5duXeW/Zr0v88tLRGogaKrKBDXQ4mwL8ZC4pASCtE0iwqsE7sqVK715E5ckk8AytSoteDUajUaj2be4xI9ERqO5Gwq20cFuum1mBXi2hjatOfLk4/jypTfj+nhd18U0TbZv3y7POecc8cknnxS1bt361z179vzmlltu2V6HVSuhmuhYKpXpAH2BfwHH0Ehi94QThnP77bfz7uKvxbrCrSIjmNbgLZ+bK0IIwrZNh6zcaIWGeOe1eu/nn70O0UkkrJVSFeF1W364XKPRaDSapo1D4vqwABRVlNYWGWwyCCEIh0IMPPV4UtLTEkaiHcfBNE3mzp0rzzvvPGvu3Lk/CiG2Dx8+PJXdrQw12RoE3jF0qN3pYbBr5PdqYDae2LWpm9iNxJvIsiwcx6Fz58689r//MWv1YuasWUZGMFWL3RhUwlq7rFaYRvx22LElydasWRMdV9vk/utqoECNbNrfGo1Go9FoWj6JHr9HccK+1mp6boYoQggioTAde3Vn4MgTol7deDiOIwzDkDNmzMg47rjjXk1JSTlgxowZlVTZgu0aBnWjEO9oKFGsIr9HAO8Az+DZGVRzikSoBLq1wFl4QkoKIWzV6tgwDCzLwrZtcnNz+ej9D9geiPDxj981uQTDpoBqjd0pp23iaZWlZ906CgoKdhlXy6IBfsY7TwxAasGr0Wg0Gs2+RQk4qEW8qQe94crQXtmg+iKEIBIOc/yvziQtKzMaoYuH67qGEEKGw+FBoVBoJnA+VZFbgSdMVUS31lWza9ULJYoPBf4JfAecTZVQTiay6/rTFQAXAJ8AfwCElNJSJdRc18W2bQ47/HCmff4F6T07MvXbz0lPSW2Sfut9jpSYhkmXnDZA/A9VHb9Vq1ZRXl6e6AZKHezl/qsR/Uej0Wg0Gs1eR12YK/0hIZFwuFmkNQkhiFSG6NC9CydeeLYneI3EGy6lVPaErsAbwGt4lRdUlLd6zV01xJYvUyIZYBheNHcucDOQ4r+XSDgrVGvhMrzEtjlAQAjxHHB++/btZ/Xu3Vu2b99eDhs2jIcfeZhF8xeQ2qM9L836kJRgsNZGCvszArBdl1ZpmdGWwvEOkxK8P/3kVSgxTTPeTYTStktiR+qkNY1Go9Fo9i1JC95wRajZCCjDMKgoK+f4X53B0q/nsm7pCoRhIBN3YDOpuhkYjRfpnQZMBb4BfsLz0dakeDoAR+GJ5FHAAKqklPLqJlt2TCWyFQO/8rfBAiL33Xef9eCDD761devW35aUlIiKigq3Tdu2wjQMluSt5fX50wkGPLErm7L/ZB8hhMB2HTpm55IaSEEi457XKqL7448/AgntDKoc3UJ/nAta8Go0Go1Gs68J+UPt+FqgorTUH9EMRJQQSNclkJbKebdexdO3jiNUUZlU9zWqRKoSnaf4QyVey+A1wFa842YBrYEeQEe8EmaxqIhuXTSPaj6xHS+yO9P/25ZSGkIIu7Ky8hbgrKysLDcrK8sAmLtmGR8tm4Nlmcnu536JOipdW7fz/pYQz/GiBO/3338PJNW2eh2e3zq6Oi14NRqNRqPZN6holItXW1aNq3lKoLKsEjsSabJlyaojDINQeSXd+vTmnN/9llf/+hSGaXpR3uTEoIr2KitDKnCAP9TG/7d373FyVnWexz/nnKcufUknnc79nhASwgBRIQpeRqKg4lzQZUDBdVxZRxxnnJEZ1tVVidF1GF/eeDnMDDs6I8ygi2DQWbkYIRBNBAJJIAkhpkNCEtK5dDqdTt+qq+p5ztk/nuepqu50V1WHXLqT3/tFUd1VT1dXd1de9a1f/c7vxG0NcdvCcDeSiBeybQduADZTDLsKcM658cAXrLUu3vXr6V0v8fjv1uMZDy1htzwHnjbMapwMVO7fVUrR1tbG7t27C5cNIe63fpHwxVD870t6eIUQQogzKH6uby93UPz0nsv0kc/mUKNoo1RtNJnOHpa8/1289+M3YIMArdVwQnscWj2K4TcY5FQ6saGaBW4DlS5kewy4kpKwG/84SilHuAXxZK21NcboJ7dv5FfbnifheWgk7JYT9u8GNNaOoam+oXjhEOJq7pYtW+jo6Kj0uIl/8Zui88KLndHzL0YIIYQ4e7WWvzp8Hs/09JDL9IW9sKMoVCmj6evp4T0fv4GP/vknsIFFqWGF3sJNUZypO/AUB9zh3mhpRRjgq8D7CVsnDMWwi1IqcM7VWLgZIB/46ueb17L6lRdJJ5Ioh/TsVhD3704dO560lwwruGX+ZPHjfMuWLYWNSso89uOAuzE6L/Q+SOAVQgghzry2stdGz++9nd309WYqzrUdibQyHD12jJs/91d84+/uIB7nZcxwOw5OqnjkmQG2AO8DllEM1oVd"
        "25xzBiDr++/WcF5Hptv++Pkn9Ma9zdQkUlgnUbca8YiNuU1TC5+XEz/Wn3/++Wpvuo9ihbdw86PvX4wQQghx9jlU7sq4otXX1UOuty8MAaMwXSWM4cVdv+NzX/g8P7rvR4wbN66w09pp7EuOR5xBGHR7gDuAK4Bf0b9v+Dgpz7vmcG8n9z77mH21/QC1SZmzOxzWOdKJJLPHV+7fhTDwOufYsGFD+PVDL1iL/wgvAfsHXCaBVwghhBgBXqt0gFIKay09nV3RTNvRFbIcjqTx6Mh08/L+V7npIzexevVq3vGOdxAEQaHaewqr15Yw6MabWEA46uytwP8iDL6GkqpubNmyZTpqZ1Brd25e+uONq+jI9Oi0l5TtgodBK4Uf+ExpGE9jNH+33Aud+IXEjh07ql2wBvAsxYWHEniFEEKIEaQlOh/62T+6pv1ga9jDe+rv0ylhgV1HDmKtZfHixfzmN7/h7rvvZvbs2QRBgLVhf6/necTb9p6guEpbWHBGGIJyhBtavJPiFAZD+Bs+LuyybJn+6vLl9sLrr09e9b3b/mzltvULe3p6SBpPyXbBw2etZVbjZLTSFbdbjqu569evJ5PJVLvhxNPlrhRCCCHE6Rc/e7cT9h5qhizdhsGvfX9rGAJHYdZyOBLGsLv9AL6zhUBzyy238MILL/Ctb32LRYsW4ZzD9/1Cn28cgD3PK1SBS0/GGIwxhWOisBz34cbV3O3A1wk3o/gQ8BuKO7UFlPxGnXNq2bJl+gH3gFHLl9umz7536sv1uz7ck+1dWp+uCUeTjYo970YWR7hL2ryJ04Hqf4HPPPNMePzQL37iEX9Zws1JYMCLF5nDK4QQQpx5R6LT9KEOiJsYjh5sHTVzeAdyDjylOZrpZn/HYWaPn1IItY2Njfzt3/4tf/EXf8HKlStZuXIla9asYfv27eTzeXzfr/wNQgHhXOMssCOVSj0DPDx+/PhnDx482BveD6cB5VyhHyGe7qABp5QKA/By4JOXv6m2vuH6t86cP/vSmQsuUkop55wbrX+DM0WhyAc+E+vHMW1sU3xhWfGCxrVr1wIV+3cV4cLDQduDJPAKIYQQZ44lfKI+QjipYTrF4fmDOtp6JBpNNjoDl1KaXD7Hax2HmdM0FetcuBmFc1hrSafTXHvttVx77bXkcjl2797Nli1b2Lp1K3v37qW1tZW2tjYymUyh77ehoYFx48Yxbdo0Zs6cmRs/fvzTzrmvf+pTn/pNNhtuYnfgwAEovrPtADsgtBYWqjnnzHO7tp33b88+ejNGfWBSfeO0umRqjPE8stksStLu8CkIbMCcpskkjRf+3Sv07yql2LFjB83NzYXLhmAJ/7brKO7OJxVeIYQQYgSJ31LfAywe6qD4yf5YaxtdxzqpHzuGwA9GXbXXOUvK89jcsot5TdOYPm4CECWWkuALkEwmWbBgAQsWLOC6664ruQ1XWOgWtzuUqAHeA1x+yy23PJvJZJ4Ansnlci+PHTu2Uynll9yOBhJAQ578ggSJRcDilmNtb92wv/nC86bPShsU+XyeIAjws1knYffEGWVYMHEWULmdIZ7e8fTTT9Pb24sxhiA4vsU6Er+QWROdH3fzEniFEEKIkaG53JVx4O1sO0pPRycNTY0Efjx0YPRwgFGa9p5O/n3dSi6bvYDL517EmFQNEAXfaLGac67fSSlVuG5AyB0Ygp3neQ3Ae2pqat4DuJqamg5gv3OujeK2s2MtTMTapoROjAHY2dXKYy89w9GeYxh0kAu3E1ax0/V7OpsowA98muoamNE4Mbyswm8y/juvWrUqOr5i/24n8OvosuN6HyTwCiGEECPD9koHKK0IgoD2A61MP39usXNxlHGApw0Ox9qdL7H1wG7eOPN8Lp25kPoo+MbHKa3QA9bYD/bW9oAQHLc8x9sFe0BjdOpHA2hNy7E2nmp+MfjdwT0u5SW0p7VyYCTjvn7x7mqzx08h5SUKL17K0VrT29vLr38dZtgy/btxC9AzhLvjKSTwCiGEECNWHHiHnKCkUDgc+3fsZvHSK0b1qIB4X7KaRJLubB+rtm/khdd28HtT5nLx9HlMaRjfr8fTOhf+rKr87NYS8Q5qAM5aiwOntHYlv2B14NgRnt39stp2cDc53zd1qTTIrmknlQOMNlw4dXZVx8ftDOvXr2ffvn2FGdQVPBmd99sOOiaBVwghhDiz4my1h3A82XiKi3AGdWDXnrNmdy/rHEYpahMpurMZ1uzczPrXtjO7cTILJ89i/sTpjK2pO26BkyOu9A5V5nYoVByOVbShhQLI+nl2te1nc8sudh3ZT18+R8pLkEokz5rf60ihgHzgM6m+kZmNk8LLqqyaP/7441hr8Tyv3JQOQ/jv5VfR54MmYwm8QgghxJkVT2poAfYRBt5BU1ecxY60HKTnaCeJVBJrXcV+yJEuDq9GaWqTKaxzNLe+xvbW16hPpZna0MTM8ZOZMXYCk8Y0UptMYwqbUgz1wxcvzwc+HZke9nccZnf7Ifa0H+RobxfOOZJegppEEhv1CYuTK25nOH/iDBJVTGeAcByZtZaHH34YqNjOoIFtwNboskH/iBJ4hRBCiDMvntTQDFwy1EHx2Ni2loN0th1l0pzp2L5c5RVAo0SxagupRBKAnO/zStt+trfuI+V5JL0E42rqGV87hvpUDfWpWtJeEs8YjNb4QYBvA/ryObqyvXT1ZWjr7qAz20vWz2OtJWHC2yH6frJj2qljcaRMotDOUO10hs2bN7N1a5hhK4wjU8AqIM8QW0ODBF4hhBBiJHkO+JNyByil8HN59u/aw9TzZpGL3ro/28QhRylFykuEK5GcI+/7HOxsp6WjDRu9ANAqrPaGK9UczoF1FoVCK4XR4e5s8e04qeaeFkopcn6eWY2TmdTQWNVitdivfvUr8vl8Ne0MCvhFpduTwCuEEEKMHBuj8yFTgdIKFzj2bN3OkvddWexzOIu5kkVkSikS2iNpgELE7U8x4DoXB2FZjHa6Wee4aNpcjNLFhYdlxLurPfTQQ+HXV25n2An8tuSyQUngFUIIIc68OIdtJ9x1rYmhVmNFR+5/ZQ/ZTAZtTBh6z5K2hmoUg+vg8bXcdeL0UCgCG9CQruOCydFmExUeo3E7w8aNG9m4cWNhFvMQ4naGRwm3kh6ynQHKrAAVQgghxGkTP3nvB3aVXHb8gVHF6+DOPXQcbMMkPIl2YuRRkA8CFkyewZh0bdjOUOWXPvTQQ+TzeUy0894Q4naG/6zmNiXwCiGEECODJgy5z1U6UClFX2+Glh2v4nme9KOKkceFm4tcMu28qr/EGEMul2PFihVAxXYGRdjO8HTJZUOSwCuEEEKMLE9VOkDpsFb2yotb0Z45J/p4xeihlCIX5Jk+bgIzxk3EUXmxWrwt9K9//Wuam5vRWlcKvNC/naHsPwIJvEIIIcTIED9hvwh0U+ZJPM63e7Y203OsC+N5knnFiGKd4w0zzsdoXdVjU6lwk5B7770Xay3RRiFDiXfQu7/a+yOBVwghhBgZ4rdpd1PcZnjwwGstKDi0ex+te1rwkgkqvKMrxGmhlMIPAibUjeXCKeHs3UobTcQBd9++fYXNJoJgyPVn8b+T54F10ccVH/wSeIUQQoiRI96AYnX0+ZC1Ma00Ngh4ZeNLmEQCJ3lXjADxVsKXTJ9HOtrBrpK4deGBBx7g2LFjlfrS4yv+L+G/lYrtDCCBVwghhBiJHo/OKy5s37FhM/lsrtDXK8SZEo4is4xJ17J4xvzwsgoPS+dcYbHaPffcA5RdrOYIA24X8GB0WVUv9STwCiGEECNH/OS9HjhAcXLD8QdGoWDP1mbaXttPIpmUaQ3izFKQC/JcMHkW42rqo40mKrczKKV47LHH2LJlS6XFagFh6H0U2EcYfiXwCiGEEKNMXME6Qtif6Cjzdq3Smnw2R/P6zSSSCZyVwCvOHOccqUSSN8+5AKji7QmKm1Hcdddd/T4fQjx7977h3jcJvEIIIcTI9J9UyAzxldue"
        "3kA+n5e2BnHGaKXI+nkWTZ7N5DHjw40mqlisppTiueeeY/Xq1Silyi1Wi6/YStjyU9VitcL9q/ZAIYQQQpwWcZn2SaCTMm/bxm/97n5pOwdffY1EStoaxJlhnSNpPN4y58JhfZ1SijvvvDPwfb/SKDIIQ+73gSxhhq36wS6BVwghhBhZ4rFLe4FnqdDWoLUmn8vx8m/XS1uDOCPi6u7CyTOZNrapqupuEAROa82mTZu6V6xYERhjqlmsdhj4cXTZsOaSSOAVQgghRp74+flnVGhrcFEW3vKbdWS6e9FGFy4T4nSIq7tXzP29qo6P3oVQQPett976Wi6XS4YXD/m4jcPtjwhDb1WjyEpJ4BVCCCFGnvjJ/FHgGGXaGpwNq2kHdu5h90vbSabTIFVecZqE1d0ci6bMZvq4iVVVd51zgTGGVatWrXjqqaeajDHlencdYV7NEbYzxJcN734O9wuEEEIIccpZwufovcBT0WVlpjUonHNsfHxN+PHpuIdCAIFz1CRSvGP+JVV/idbaAP9x9dVXZ40xk6y1AUO/kxG3+DwCvEyZUX3lSOAVQgghRqY4ANxf6cC49/Hlp9fT1nJQZvKK00JF1d03zDififXjqqnuWsJ3K15cunTpvc65/+pC5fJovDjtu/G3PZH7KoFXCCGEGJniKtbjQAvlhuy7cPFapquHDSt/TSKdxA29AEiI1y3eVa0hXccVcy8M+8YrdDJE593Ah1evXv0pY0ytDV+tDfWVceV3FbCW4tbbwyaBVwghhBiZ4pXp7cBPSy4blHUOUDz985Uc2XdQRpSJU0opyPl53jz7AsbW1OMclXZVi9t0blRKNXme9ydB2Lhryn2b6PybhI/9Ex40LYFXCCGEGLnixPrvQJ5ys0edQ2tFT0cnv3nwERKplIwoE6eEUopc4DN1bBNvmXNhuKqsulaGryqlHjbG3O37PpQPsAHh4/0p4AleR3UXJPAKIYQQI1lcFXsR+C0VdpeyLty5at0vnmDP1u2katPlZpsKccKccyxd8EZSXgLKv5MQADqfz2/yPG9ZMpm8zVp7McVAO5Q4DN9BceHaCZPAK4QQQoxsccitvGjHhdW3fC7HY9+/H6oYESXEcCil6MvnuGDKbBZOnlXtGDISicTfBEEwNQiC251z8Qu5ocRh+HFOQnWXCt9MCCGEEGde/OT/CFUs3LHWorRmx4bNPPP/nqCmvg479IxTIYbFOktNIsm7FryxmpJrABhr7S+11k96nvfdIAjGULkfN36Rt7yKY6sigVcIIYQY+RRhePjfVR0dVd1W/tv9HNq9j2Q6Lf284nXTSpHN53n7vIurGUPmAOX7vjXG/JVz7g+stR8iCsFlvk38Am8FYRvP667uggReIYQQYjSIQ8BK4BeEgWHoramcA6Xo7ezmZ3f+ABf19sqOFOJEhTN388xpmsIV836v2pm72vO8f1FKvWaM+RdrbaVqbXx9BvjKSbvzSOAVQgghRpsvEAYCKBNhnbVordmxYQuP3/tTahrqZQGbOGHOORLG45oL34LR5Qq04eGACoKgE/iSUuofnXPTKC7CHEp8/T8T7qo29OzpYZLAK4QQQowO8WinrcBdVBEGbBR6n/rxz3lx1VpqG+oIpJ9XDJNWmr58jrefdzFTxzZVXd01xixXSr1Za31ztH1wudzpousPAN+g3Ai+E/kZTtYNCSGEEOKUi0PBV4Fmqgi9zjlsYPnpN+9m345XqamrxQZS6RXV0dH2wXMnTOVt8y6qKuw65wzw8o033viE1vrfSloZyn1hPHrsdqCVCiP4hktmlQghhBCjS9y/+17gl1ReBITSGmctk+fM4JPf/jJ14xrI9+XQRupeYmiKcAe/hPH4+BXXVLNQDYqPx48ppW4wxvxBFTuqxdevBd4ZXeaQCq8QQghxzorDwUrgn6iwgA3Cfl6lNYd27+NHy+/Ez+bxkh5OenpFOUqRtz7vu/DNVYVdGzaJG+DRdDq9SGtdTdiNq7954FaKld6TusRSAq8QQggx+sSLe/4HsJkqQ6/Wml2bt/Gjr96JDSzGk9ArBqeVJpPLsmTWBVwy/bxqKrtxcD1y8803b81ms39D5UVqlBzzHWA9VTyWT4S0NAghhBCjkyYMC5cBa4Aklfsk0VpjreWity/hpts/i1IQ5H2UlhqYCOloBNn0sRP407e8j4RnUBUiYxAEzhijnnjiicevvvrqi40xU4IgqDSGLA6724C3AD2c5FaGmAReIYQQYvSKq2H/HfgB4EeXlQ+9RmMDy4VvvZQbv/RXJFJJ8tkcWkLvOU8pRRAEpBMp/tvl72VCda0MTmutDh48uP+8887z+/r6ZrlQpZm78dsL7wGe5BRVd0FaGoQQQojRLO6P/FfgHwGPKla228CijeHlpzfw71/+FpmuHlI1admC+BynUDjncMAHFr+92rCL1lrl8/n8tddem+rt7Z0FVAq7UByzdydh2PU4RWEXpMIrhBBCjHZxG4PRWq+01i5VSgXRaKiy4vaGaefP4SNf/iyT58ygt6sbYyp+qTgLKaXoy+f4w4uuYMnsC7DOocuEXecc1lqUUu6GG27IrFixosYYQxAE1YRdTdh/fgXQxylqZYhJhVcIIYQY3eKQkJ8/f/4N9fX1rzjnjFKqcqXXWrTR7N+xm/9z63Kan3uRurENOGvD7YnFOUMrRSaf5R3zL6k67AZBgDGGW2+91a1YsaLW8zxVRdiNH1hZ4ONAL6dgKsNAEniFEEKI0c9ef/31prm5ue2v//qv/2zWrFnWOee01hVDhA3C6Q2dR47yr1/4e9Y8+DDp+lqMZ2SDinOEVoreXJbLZi7kqoWX4qoMu57n8fWvf53vfe972vO8anfxi3dc+xywkVPYt1tKWhqEEEKIs8Q73/lOb+3atf7999//jc9//vOf27lzZ2CMMdUEEaVUoap72XvfyR/8+UepbxxLX3cPWhtJDGcprTW92T4unjaP6974+2hVvhYatzEYY/j2t7/NbbfdhjEGW927AnHP+U+B6yn27Z7ytxPk4SuEEEKcJZxzSimlnXOJVatWPfuxj31scUtLS9WhF6XC4GstU+bO4oOfvZn5l15MtruXIJrjK84eWmsyuT4WTZnD9W+8EqM15eaIlVZ2TyDslo4geztwNL7Zk/GzVCKBVwghhDiLOOe0Uso655Zs3Ljxt9dcc41ubW3VxhhV5VvOhbFlxvNYetO1LL3xWpK1NfR196K0qrQBgRgFtNb05rIsnDSTG960lIQxONyQ83ZfZ2U3PqAX+H1OYytDTB6xQgghxFkmWrQWOOe+/Oqrr371bW97m3/gwAEvWkFf1W0orQu7sM284Dyu+eRHWLhkMflslnw2jzZS7R2t4sruvAnT+fCl7yLlJcqOH7PWOhXia1/7mrv99tvVMMNuQNi+cCNwP6c57IIEXiGEEOKsFIVenHOP7Nmz573XXHNNsG3bNuN5Hr7vV3cjSqG1KsztvfyPr+bKD/0RE2ZMpa+nh8C3EnxHmTDsZpnTNJkPX3oVNYlk2bDr+z7RgrTgL//yL/N33313ehhhF8LNUDzga8Dt0cdVPgBPHgm8QgghxFnIOae01m7r1q1TFy1a9ExbW9vsD37wg3bt2rWFFfXVjh5TWuFseOyYxrG866PXseR9V1JTX0tfbybefOBU/jjiJNBK05fPMqNxEjdd9m5qk+lBw27cvqC1RilFPp9v+cAHPqAfffTRScYYcwJh9z+AP+UMVHZjEniFEEKIs1Tc2tDV1bW0vr5+ZSaTMZ/+9KfVPffco1S0QM3aKkePqWijimhU2bTz57L0xmu56B1vJpFK0tfTG46zkuA7Imml6fOzTGuYwE1LrqI+VXNc2I2DbLzxiLWWFStWBLfddlvb3r17J3ieZ6p+d6A4keEJ4P0Uq7pnZMCzBF4hhBDiLOac85RSfj6f/4bneZ8DgjvuuMN86UtfKixCqravF8LxZaVBed7iRbzj+j/kgsvfSCKZJNubwQZW"
        "FreNIFop+vw8k8c0ctNlVzG2pq5f2A2CAKVU4cVKX18fK1as4K677uLZZ58FGO7jJK7sbgDeB7QRTmg4Y4Od5ZEohBBCnMWcc/HWw/XW2g1a6/MA98gjj+hbbrmFlpYWhtviAGGbA47C18xbvIgr/vg9LHrrpdTU15Hr68PP++G+x1L1PWO0UmT9PE11Y/nIkqtorB2DdbbwtyvdRnrXrl08+OCD3HvvvWzbti38eq1xoWozYxx2dwBXAXs5w2EXJPAKIYQQZ72SqQ3XAj/P5/NBIpEwe/fu5ZZbbuGXv/wlELUsVNviEFFag3OF4Dt13iyWvH8pi5e+jbGTxmN9S66vr1BRlKrv6ROH3fG1Ddx42buZUNtA3gYkPK9wjHOO1atXc8899/CLX/yCo0fD8bjGmEIv7zDEYfcl4L8Qht4z1rdbSh51QgghxDmgJPTeD3woDr3WWr797W/zla98hd7e3uHMVu0nrOK6wuK2hqbxXPT7b+aN734bMy+YTyKVJNeXJcjncYRhDAm/p4xWikwuR2NtPTctuYpJYxr7Xb9u3TpWrlzJQw89xKZNmwqXe56HtXa4QdcRVnANsJpw/NhBRkjYBQm8QgghxDmh5C3pCcA6YG4QBNaYcK7YunXr+MxnPsPzzz8PDLtns0BphaLY46uNYe7FC3nDu9/OosvfxLhJTaAU+b4sge+HO3tJ5fekcC58waF1WNmd3jSJT1zxh2ggl8uxfv16Vq9ezUMPPcQLL7xQ+BvF/bsn8kKH4uI0gH8BPgPkGEFhFyTwCiGEEOeMkirvu4GVANZaba1VnueRy+X4zne+wx133EFnZ2dhEdNw2xzg+MVtAGPGj2PhksUsuuJS5l6yiIYJ4wBFPpsl8AOcdbLYrUrOuaiVJCyUK60xCY9EKonvB0yoGcPlTfN4Zcs2Hvnlo6x+ajU7duzoF2jjaq4raUkZzl2gWNXtBv4n8E/RdWe8Z3cgeUQJIYQQ55B4aoNz7ivAMqK+y9JZups3b+aLX/wiDz/8MMAJtzkAoEApjaJ/cB47sYmFSxZzwRVvYvaF59PQ1Ij2DH4uj5/L46wNw5wE4GIgdfELCdCeh5dMhJV438f3fdr3H2L/zj20NL9K1679tLy6l4Oth/rd1usMubHSqu7zwKeB9YRB13GGRo+Vc24/goQQQohzTNTaoKNK74PAnxCFXuccQRDgRYuafvKTn7B8+fLCiv3XFXwpVn1dSa8vwJimRmYvms/5l13CnIsWMnHWdJLpFFor8tkcft4vBmClwgkRZ5E4zLpwdAJEvydF+LOahIeX8FBKE/g+gR/Q1X6U1tcOcHhvC/tf2U1L86u0H2gl093T77bjKQxxwH0dIReKVVsN9ALfAP6esIXhjOygVq2z6xEjhBBCiIpK+nnrCTcGeDPFFfZYawvhtKenh7vvvps777yTffv2Aa8/+EJJ+B0QwpLpFJNmTWfeGy5kxoJ5TJs/h6bpU0ikEmht8PNhBbjw/UuqniNqEZxzxTJn/LFzQPG+KgCt0NpgjEF7Bm3CzT1sEGCDgGymj6MHD3P04GEOv3aA1r0tHNr9GkcPtdF99Nhx31ZpHd6+42QE3JgNb7FQ1V0JfJFwzi6MwBaGgUbQI0MIIYQQp4tzTiulrHNuDuHK+tn0f6uaIAgKFcJDhw7x3e9+l+9///u0t7cDJyf4QsmiNUVhJ7dYuq6WCTOmMOfiC5g+fy6TZk9j4oxppOpq8RIGpTR+Pk+Q98P7YoshrxAshwrCYQod/P33gdcd94GK/ysE7dKPtdZh+NQabTRa62imrQ0DrQ3P/XyenmNd9HQco+tIB8fa2mk/eJj2A620H2ilq72DTFc3ub7soD+CNiYK1K7fXOSTpLSiC7AJ+Dvggehzj/AxM+JaGAaSwCuEEEKco0oWsV1EWLWbxoDQO7DNYc+ePfzDP/wDP/zhD/sF3xOY2TqoOPiG1V9wA27TSyaoHzeWKXNnMmPhPCbNmk7jlEk0TZtEbUM9xvMw0X2N3/4PgiCqrg7yfcJvFuVU1S8kF4P4wM+JgnUYsK1zYcuFDc+ttWQzGbI9Gfp6eunt6iHT1U1vVw89HZ10dxyj80gHnYfb6WzvIJfJkM304efyZX8vhZnH4R/mZIfbWLwYTVPMiS8BdwI/AvqiyxUjvKpbSgKvEEIIcQ4rCb2XAL8EplLS3lByXGErYoC9e/fygx/8gB/+8IeFVofXOd5qUKUBGDf4xAiT8KgdU8+4SROYPGcGE2dOZdykJsZODE9jxo8rVEJLf5bA97G+JQiK50HeDxfORa0TxfP+l2d7w5Ca7c3Q15sJz3t6yXT1kOnuKSy+8/N58tlc1T9rPM84+q/ffT7FBrYtADxLOGrsJ4Q9u0TXx8eOGhJ4hRBCiHNcyeSGxcDPgTkMEnqBQpiNg++RI0e47777uO+++1i/fn3huJNZ9S1VaE8o7QEe4nsorUjX1lLTUE+6Jh1We30/DLpRRbYwsaDweVDSQ3sSg7tWKBV3BhwfaMMPT3uGtBRHi8WZMAs8BvyAsOofL0QblUFXCCGEEKLAOWei8znOuXUulHfOWTeIIAhcPp8vfJ7NZt3PfvYzd91117na2tp4NJVTSjnP85xSqnDZST8p5ZRSTmnttNZOG+O01ift9sPbVsXb7neKvqfWTmntlI7uS8kJdYp+7uGfLGGA9Qe5biPwJeD8AQ+N0jAshBBCCDG6lYTesc65n0VZ1jrngsFCr3POWWv7BV/nnGtubnbLly93F1988XHB0fO8kxpGy54UxeA5SBAtd2JkBdXXE3ADIE9xcVnpaTPwTeBtQLLkoaCRoCuEEEKIs5VzTpd8/GXnnB/lWN+VYa11vu+7IChm40wm41atWuU++clPupkzZx4XyOLwe0qrv+fWKa7g5im2H5SeeoEngduBtwDpAX9+j+JEBiGEEEKIs5dzTrloVq9z7j3OuZ3VVHtjA9sdnHOuo6PDPfroo+4Tn/iEmz179nFhTWt9equ/o/tUWrkdqnobB9zngX8GbiLszR4oDrlndTX3rP7hhBBCCHHiXHGCw2TCbYj/PLoqIMwQZauBrmTRWrzIDaCzs5MXX3yRRx55hCeffJJNmzaRz/cfyWWMQSl1MrbBHY1cyXl8in/fQ2U3B+wj7MXdQrjV7wbgIMfvgObRvwJ81pPAK4QQQoghxaE3+vhqwo0HLouurir4Rl87aPj1fZ+9e/fy5JNPsmbNGtatW0dzc/OgAXdgCI5vd9QonfsLKKWctdY650rn3lbTUtAFvAI0R6dthCF3L9A5yPEex1d/zykSeIUQQghRVtTeoKKd2TzgY8BngYuiQ+JqYVVvjcfh1zlX2NAilslkaG5uZs2aNbzwwgts2LCB7du309fXN+htFXcwc8VT+E1O8KetIA6t0ceFz4r/C5VUpUtCefxBQPi70lActVZynA+0A/sJQ+xeYA+wKzq1AB2E7QyD3EMM53jAHUgCrxBCCCGqMqDaWwP8GXALcGHJYT6V334vvc1CAFZK9av+AuRyOQ4fPsy2bdvYuHEjmzZtYvv27ezcuZNjx46Vr/CW7o4Wfrf4v2IgVsMKrMNV2nOrUJhxE5uYNXMWUyZMJJlIZg8cOPDc5s2bn1VKteZyub2EIbeFMPBmgHK7VsShWcJtBRJ4hRBCCFG1qNprlFJ+9Hka+GPgo8D76L9ZRbyYKg6/rzsAQ9gG0dnZye49e3j55a1s276d5zdvpPmVV2hrbaXvWA/OD07qphdKKbTRaGNIptOkatOkamvCU03apWrSJNIpm0inXLquRqXH1JlUfR1NE5uYNnUa0ydOZmbTZDt38oyWyWPHrxkzpuHJpDFPjBs3bs+xY8fKfmuKu58NDLUSbqskgVcIIYQQwxYFXx1XfKPLLgL+CHgv8Hb6b1MLxZ29SsPvwCxS+rmDcHe3+BRPcxjsNvZ1HGbTnma27N7Bnn2vcbS1jczRLjLHuujr6aW3q5tMdw+Z7l4yXd3Y"
        "wJKqqyFdV0u6tib8uLaWdF0NNfW1pOrqqKkLP0/F5zU1eJ5x2hiU0VZp7YzneYl0klRNmrq6OrCOlNVMrBvbPbFu3I55E6c9P1anNgDPAa8opboH/MwJhg6zEmpPAgm8QgghhDhhcfAFbDRPN778YmApsIQw/M7k+AA85M1SZUYJggBrbWC0cdpoBag8gXr1yEG2HdrDno7DKmt9/MAHa3GBwwUW6wfgHCbh4Xke2hiM52G88C5GAdu56DywgQsC66wNlHMYpRV1tXXUJFNkezMklGnRio0NXu3v5k+YtnPBxBlbGxsbdwBH4mp4bNmyZfrKK6/Uq1evtsuXL5c2hNNAAq8QQgghTgoXblqhBwY851wtMJ9wkdvC6DQHmAxMINwAof/qtZIvJ6wK54CjwCGgDZgKzANqy92n9u5O9rYfcoc6292h7g7X0dvlevNZfBeowFkCawlsgLM46wKlUCiUNlorYwwJzyOVTJFOpaivraPWJPGsIm28Ns94z9Wnap6f0jj+ifpG78WpanL3YE0UDzhnrg8zlwVc6QsDcXpI4BVCCCHESRVPdaBY+T0uB0bhOAXUABOBMYTb26aik0+4cUKWcPHWEaAb6ItmAyeA8cD5wCLCQD2DsJI801rbpLVOEbYLHJd3umyW3lyW3nyWfODjcPjWYgMfFziCvI9SOpf2EpmaZPJQyiR21qVqm9PG255IJF4Afucp3RH0L86qZU89Za4ErrzyynhyBRJwzzwJvEIIIYQ4pQYEYAUEg4Xgk/i9aoAmYKrv+01KqfEWxhtjJugwTCcJK8rJ6D75QB/QQzjHth1oJdy0YS9wTCk12Fw09ZRz5vCDD7rrr7/eSrAduf4/TgCnMmFwxIEAAAAASUVORK5CYII="
    ),
    "photon_red": (
        "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"
    ),
    "photon_green": (
        "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"
    ),
    "slab_logo": (
        "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"
    ),
}

_ART_RAW_CACHE = {}  # name -> PIL.Image (RGBA, decoded once from the embedded bytes)


def _load_art_image(name):
    """Load (and cache) the source PIL image for one art asset, decoded
    from the base64 bytes embedded in this file (_ART_DATA_B64) -- no
    filesystem access, so there is no separate art/ folder to go missing."""
    if name in _ART_RAW_CACHE:
        return _ART_RAW_CACHE[name]
    if name not in _ART_DATA_B64:
        raise RuntimeError(
            f"Unknown art asset '{name}' -- no embedded data for it. "
            f"Known assets: {sorted(_ART_DATA_B64)}.")
    raw = base64.b64decode(_ART_DATA_B64[name])
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    _ART_RAW_CACHE[name] = im
    return im


_ART_OUTLINED_CACHE = {}  # name -> PIL.Image, with a white silhouette stroke added


def _add_white_outline(im, thickness=3):
    """Trace a solid white stroke `thickness` px wide around the *outer*
    silhouette of `im` (an RGBA PIL image) -- the whole character's alpha
    boundary, not each individual ink line inside it. The artwork's own
    pixels (black ink strokes, green/red fill) are left completely
    untouched; the ring only ever occupies pixels that were fully
    transparent in the source, so there's no risk of it bleeding into or
    covering any linework. This replaces the old approach of inverting the
    black ink to white -- Filip wanted the ink to stay black and instead
    have an outline carry the contrast against the dark background.

    The canvas is padded by `thickness` (+2px slack for anti-aliasing) on
    every side so the outline never gets clipped; callers that rely on
    fractional anchor offsets (eye/snout position) are unaffected in
    practice since the padding is small and symmetric."""
    arr = np.array(im)
    mask = arr[..., 3] > 16
    pad = thickness + 2
    h, w = mask.shape
    padded_mask = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    padded_mask[pad:pad + h, pad:pad + w] = mask
    dilated = binary_dilation(padded_mask, structure=np.ones((3, 3), dtype=bool),
                               iterations=thickness)
    ring = dilated & ~padded_mask

    # Build directly with numpy rather than PIL's paste()/alpha_composite()
    # -- paste(im, box, im) blends partially-transparent (anti-aliased edge)
    # source pixels against whatever's already there, which very slightly
    # changes their RGB/alpha values. A raw masked array copy instead
    # guarantees the artwork's own pixels come out byte-identical.
    out = np.zeros((h + 2 * pad, w + 2 * pad, 4), dtype=np.uint8)
    out[ring] = (255, 255, 255, 255)
    dest = out[pad:pad + h, pad:pad + w]
    dest[mask] = arr[mask]
    return Image.fromarray(out, mode="RGBA")


def _load_art_image_outlined(name, thickness=3):
    """Same artwork as _load_art_image, but with a white silhouette
    outline added around it (see _add_white_outline) -- reads clearly
    against a dark background while the ink/fill colors themselves stay
    exactly as drawn."""
    cache_key = (name, thickness)
    if cache_key in _ART_OUTLINED_CACHE:
        return _ART_OUTLINED_CACHE[cache_key]
    im = _load_art_image(name)
    out = _add_white_outline(im, thickness=thickness)
    _ART_OUTLINED_CACHE[cache_key] = out
    return out


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
                       rotate_deg=0.0, distort=0.0, phase=0.0, outlined=False):
    """Load, resize (to target_h px tall, aspect-preserved), optionally
    rotate and/or scanline-distort, then draw the given art asset centered
    at (cx, cy) on `canvas` via create_image. Keeps a strong reference to
    the resulting PhotoImage on the canvas itself (Tk requires this --
    otherwise it gets garbage-collected and the image silently vanishes).
    `outlined=True` uses the white-silhouette-outlined version (ink/fill
    colors unchanged, just a white stroke traced around the outer edge),
    for display on a dark background. Returns (disp_w, disp_h), the actual
    on-screen pixel size, so callers can compute anchor points (eye,
    snout, ...) relative to it.
    """
    canvas.delete(tag)
    im = _load_art_image_outlined(name) if outlined else _load_art_image(name)
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
    """Draw Filip's actual neuron sketch (one of 4 expressions) centered at
    (cx, cy). `scale` sets the display size (kept as the same "px per local
    60-unit" convention the old vector version used, so callers didn't need
    to change their numbers -- multiplied by a fixed constant to get a
    pixel height). `distort` in [0, 1] applies a scanline glitch that the
    motion-correction overlay decays to zero as NormCorre progresses.
    Returns (disp_w, disp_h) for anchor-point math."""
    name = {"neutral": "neuron_neutral", "surprised": "neuron_surprised",
            "annoyed": "neuron_annoyed", "sleepy": "neuron_sleepy"}.get(expr, "neuron_neutral")
    target_h = scale * 4.2
    return _render_art_image(canvas, name, cx, cy, target_h, tag,
                              distort=distort, phase=phase, outlined=True)


def draw_chameleon(canvas, cx, cy, scale, tag="chameleon", outline="#111111"):
    """Draw Filip's actual chameleon sketch (the animation's 'light
    source') centered at (cx, cy). Returns (disp_w, disp_h)."""
    target_h = scale * 3.4
    return _render_art_image(canvas, "chameleon", cx, cy, target_h, tag, outlined=True)


def _load_photon_variant(tint="red"):
    """The photon squiggle art for a given leg of the volley: 'red' is the
    chameleon's outbound shot, 'green' is the neuron's return shot. Each is
    real, independently hand-drawn artwork (photon_red.png / photon_green.png)
    with its own black ink wave, with a white silhouette outline added like
    the other assets (ink/fill colors untouched, just outlined). Previously
    the green variant was synthesized by swapping the red art's R/G
    channels; Filip has since supplied a real green-inked squiggle, so that
    synthesis is no longer needed."""
    name = "photon_green" if tint == "green" else "photon_red"
    return _load_art_image_outlined(name)


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


def _rounded_rect_points(x1, y1, x2, y2, radius):
    """Point list for a rounded rectangle, meant to be drawn via
    canvas.create_polygon(points, smooth=True, ...) -- the standard
    tkinter recipe for smooth corners (Tk's Canvas has no native
    rounded-rectangle primitive). `radius` is clamped so it never exceeds
    half the shorter side, which would otherwise self-intersect on very
    small/thin cards."""
    radius = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    return [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]


class PhotonNeuronBar(tk.Canvas):
    """Busy-state animation replacing the plain progress strip: a chameleon
    (light source) launches red photon squiggles at the neuron; each impact
    increments a running counter and swaps the neuron's expression for a
    moment. Purely decorative -- ticks continuously while Run Analysis is
    doing anything (reg_tif read / PTC compute / CNMF), not tied to a real
    progress metric, since most of those steps don't expose one either.

    Dark background matching the rest of the app, neuron on the left,
    chameleon on the right -- both drawn with their original black ink
    strokes/colored fill untouched, plus a white silhouette outline traced
    around the outside so they read against the dark canvas instead of
    needing a white one."""

    # Sized for display in the main output area (replacing the plot canvas
    # while Run Analysis works) rather than the old slim strip under the
    # status line -- bumped up proportionally from the original 190/42/46
    # so the characters read at a reasonable size in that much larger space.
    HEIGHT = 300
    NRN_SCALE = 66
    CHAM_SCALE = 72

    # Each volley: the chameleon fires two red photons in quick succession
    # (STAGGER seconds apart, both genuinely in flight together for most of
    # the trip -- a "burst" rather than one-at-a-time), then once both have
    # landed, the neuron fires exactly one green photon back. Flight A
    # ("_flight_frac"/"_flight_dir") carries the first outbound leg and
    # later the single return leg; flight B ("_flight2_frac", always
    # outbound) only ever carries the second, closely-following shot.
    VOLLEY_STAGGER = 0.15

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
        self._flight2_frac = None   # the 2nd, closely-following outbound shot
        self._flight2_t0 = 0.0
        self._next_second_launch = None
        self._out_landed = None     # outbound shots landed so far this volley (0/1/2), None = idle
        self._return_pending = False
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
        self._flight2_frac = None
        self._next_second_launch = None
        self._out_landed = None
        self._return_pending = False
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
        dur = 0.55

        # -- launch shot A: the first outbound photon of a new volley --
        if (self._flight_frac is None and self._flight2_frac is None
                and not self._return_pending and t >= self._next_launch):
            self._flight_frac = 0.0
            self._flight_t0 = t
            self._flight_dir = "out"
            self._out_landed = 0
            self._next_second_launch = t + self.VOLLEY_STAGGER

        # -- launch shot B shortly after shot A ("right after each other"),
        #    while shot A is typically still mid-flight --
        if (self._flight2_frac is None and self._flight_dir == "out"
                and self._next_second_launch is not None
                and t >= self._next_second_launch):
            self._flight2_frac = 0.0
            self._flight2_t0 = t
            self._next_second_launch = None

        # -- advance shot A (outbound leg, or later the single return leg) --
        if self._flight_frac is not None:
            frac = (t - self._flight_t0) / dur
            if frac >= 1.0:
                if self._flight_dir == "out":
                    self._hits += 1
                    self._out_landed += 1
                    self._expr = random.choice(["surprised", "annoyed", "sleepy"])
                    self._expr_until = t + 0.5
                    self._flight_frac = None
                else:
                    # the single return shot landed -- volley's fully done
                    self._flight_frac = None
                    self._flight_dir = "out"
                    self._return_pending = False
                    self._out_landed = None
                    self._next_launch = t + random.uniform(0.5, 1.1)
            else:
                self._flight_frac = frac

        # -- advance shot B (always outbound) --
        if self._flight2_frac is not None:
            frac2 = (t - self._flight2_t0) / dur
            if frac2 >= 1.0:
                self._hits += 1
                self._out_landed += 1
                self._expr = random.choice(["surprised", "annoyed", "sleepy"])
                self._expr_until = t + 0.5
                self._flight2_frac = None
            else:
                self._flight2_frac = frac2

        # -- once both outbound shots have landed, the neuron fires its one
        #    return shot back (reusing flight A's slot) --
        if (self._out_landed is not None and self._out_landed >= 2
                and self._flight_frac is None and self._flight2_frac is None
                and not self._return_pending):
            self._return_pending = True
            self._flight_dir = "back"
            self._flight_frac = 0.0
            self._flight_t0 = t

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

        cham_pt = (cham_cx + CHAM_SNOUT_OFFSET_FRAC[0] * cham_w,
                   cham_cy + CHAM_SNOUT_OFFSET_FRAC[1] * cham_h)
        nrn_pt = (nrn_cx + NRN_EYE_OFFSET_FRAC[0] * nrn_w,
                  nrn_cy + NRN_EYE_OFFSET_FRAC[1] * nrn_h)

        if self._flight_frac is not None:
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

        # shot B -- the 2nd, closely-following outbound photon of the
        # burst -- is always red/outbound, drawn under its own tag so it
        # doesn't get clobbered by draw_photon's internal delete(tag) call
        # for shot A above.
        if self._flight2_frac is not None:
            (x0, y0), (x1, y1) = cham_pt, nrn_pt
            fx = x0 + (x1 - x0) * self._flight2_frac
            fy = y0 + (y1 - y0) * self._flight2_frac
            angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
            draw_photon(self, fx, fy, length=30, amplitude=8,
                        angle_deg=angle, phase=t * 14 + 2.5, tag="photon2", tint="red")
        else:
            self.delete("photon2")

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

    def _load_slab_logo_photo(self, height=26):
        """Load the slab wordmark, resized to `height` px tall, as a cached
        ImageTk.PhotoImage. The source art is a plain mid-grey mark (no
        color to invert), so it's used as-is against the dark chrome."""
        cache_attr = f"_slab_logo_photo_{height}"
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached
        im = _load_art_image("slab_logo")
        scale = height / im.height
        w = max(1, round(im.width * scale))
        photo = ImageTk.PhotoImage(im.resize((w, height), Image.LANCZOS))
        setattr(self, cache_attr, photo)  # keep a strong ref -- Tk drops GC'd images
        return photo

    def _build_slab_logo(self, parent):
        """The slab wordmark, top-left of the window -- click opens the lab
        site in the default browser. SLAB_LAB_URL is a placeholder until
        Filip gives the real link."""
        photo = self._load_slab_logo_photo(26)
        lbl = tk.Label(parent, image=photo, bg=BG_MID, cursor="hand2", bd=0)
        lbl.pack(side=tk.LEFT, padx=(10, 14), pady=5)
        lbl.bind("<Button-1>", lambda e: webbrowser.open(SLAB_LAB_URL))
        return lbl

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

        def _lbtn(parent, text, command, bg=GREY_BTN, fg=TEXT_BRIGHT, font_size=12,
                   bold=True, side=tk.LEFT, padx=6, fill=None):
            """Label-based button — colour always renders on macOS. Greyscale
            by default now (bg=GREY_BTN); pass bg=GREY_BTN_ACTIVE + fg=GREY_BTN_TEXT_ACTIVE
            for the one "primary action" button in a given view. Pass
            fill=tk.X for sidebar buttons that should stretch full-width
            (avoids the winfo_children() re-pack dance, which also chokes
            under the _fake_tkinter test shim)."""
            weight = "bold" if bold else "normal"
            lbl = tk.Label(parent, text=text, bg=bg, fg=fg,
                           font=("Helvetica", font_size, weight),
                           padx=16, pady=7, cursor="hand2", relief=tk.RAISED, bd=1,
                           highlightbackground=GREY_BORDER)
            if fill:
                lbl.pack(side=side, padx=padx, pady=5, fill=fill)
            else:
                lbl.pack(side=side, padx=padx, pady=5)
            lbl.bind("<Button-1>", lambda e: command())
            lbl.bind("<Enter>",    lambda e: lbl.config(relief=tk.SUNKEN))
            lbl.bind("<Leave>",    lambda e: lbl.config(relief=tk.RAISED))
            return lbl
        self._lbtn = _lbtn  # reused by the gain sidebar

        self._build_slab_logo(left)

        # Kurtosis-tab chrome (Load Data / Settings toggle / Plot button /
        # Sort-by) -- the Gain Estimation tab replaced all of this with the
        # always-visible left sidebar (see _build_gain_sidebar), so these
        # are hidden whenever tab_var == "gain" (see _switch_tab).
        self._load_data_btn = _lbtn(left, "📂  Load Data", self._load_dispatch, padx=(4, 6))

        self._settings_open = False
        self._settings_btn = _lbtn(left, "⚙  Settings", self._toggle_settings, bold=False)

        # ── center tab toggle ────────────────────────────────────────────
        self._tab_kurtosis_btn = _lbtn(center, "Kurtosis", lambda: self._switch_tab("kurtosis"),
                                        bg=GREY_BTN_ACTIVE, fg=GREY_BTN_TEXT_ACTIVE,
                                        font_size=11, padx=(0, 2))
        self._tab_gain_btn = _lbtn(center, "Gain Estimation", lambda: self._switch_tab("gain"),
                                    bg=GREY_BTN, font_size=11, padx=(2, 0))

        self._action_btn = _lbtn(right, "▶  Plot", self._on_action_click,
                                  bg=GREY_BTN_ACTIVE, fg=GREY_BTN_TEXT_ACTIVE,
                                  font_size=13, side=tk.RIGHT, padx=10)

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

        # ── settings panels (kurtosis-tab only now, hidden by default) ───
        # The old always-toggled gain-settings panel is gone -- gain-mode
        # settings live in the always-visible left sidebar instead (see
        # _build_gain_sidebar), matching Filip's "options stuff should be
        # visible before we run gain estimation... on the left column".
        self._settings_shared = tk.Frame(self.root, bg="#141414", pady=6)
        self._settings_kurtosis = tk.Frame(self.root, bg="#141414", pady=6)
        self._build_settings_shared(self._settings_shared)
        self._build_settings_kurtosis(self._settings_kurtosis)

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

    # ── Gain Estimation sidebar (always-visible left column) ────────────
    #
    # Replaces the old toggle-hidden "Settings" panel for this tab: Filip
    # wanted the options visible before running the analysis, grouped into
    # areas rather than one long wrapping row, with the Load actions as
    # explicit buttons (Suite2p folder / .mat movie) instead of one
    # auto-detecting "Load Data" button. Built in _build_canvas (needs
    # self._gain_sidebar, created there) once self.fs_var etc. already
    # exist from _build_settings_shared.

    CARD_RADIUS = 10  # corner radius for the rounded sidebar cards, in px

    def _sidebar_card(self, title=None):
        """A rounded-corner, padded group -- the "areas" the settings/
        buttons are grouped into, echoing the boxed-panel look Filip
        pointed to (rounded per his "smoother angles" follow-up). Plain
        tk.Frame has no rounded-corner support, so the card background is
        a rounded-rectangle drawn on a backing Canvas, with the actual
        content (title + returned body frame) embedded on top via
        create_window; the canvas auto-resizes to fit the content and
        redraws the rounded rect to match on every size change."""
        canvas = tk.Canvas(self._gain_sidebar_inner, bg=BG_MID, highlightthickness=0, bd=0)
        canvas.pack(fill=tk.X, padx=12, pady=(0, 10))

        content = tk.Frame(canvas, bg=CARD_BG)
        win_id = canvas.create_window(0, 0, window=content, anchor="nw")
        state = {"bg_item": None}

        def _redraw(event=None):
            w = max(content.winfo_reqwidth(), 20)
            h = max(content.winfo_reqheight(), 20)
            canvas.config(width=w, height=h)
            canvas.itemconfig(win_id, width=w)
            pts = _rounded_rect_points(1, 1, w - 1, h - 1, self.CARD_RADIUS)
            if state["bg_item"] is None:
                state["bg_item"] = canvas.create_polygon(
                    pts, smooth=True, fill=CARD_BG, outline=CARD_BORDER, width=1)
                canvas.tag_lower(state["bg_item"])  # keep it behind the content window
            else:
                canvas.coords(state["bg_item"], *pts)

        content.bind("<Configure>", _redraw)

        if title:
            tk.Label(content, text=title, bg=CARD_BG, fg=TEXT_DIM,
                     font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        body = tk.Frame(content, bg=CARD_BG)
        body.pack(fill=tk.X, padx=10, pady=(2, 10))
        return body

    def _make_tooltip(self, widgets, text):
        """Attach a shared hover tooltip to one or more widgets -- a small
        floating borderless window with `text`, appearing just below the
        first widget on <Enter> (of ANY of them) and disappearing once the
        pointer has left ALL of them. Bound to every widget in a
        parameter row (label, entry, suffix, "?" glyph) rather than just
        the tiny "?" icon, per Filip's "only show descriptions when
        someone hovers over the parameter" -- a bare Enter/Leave on the
        row's own Frame doesn't fire reliably once child widgets cover its
        whole area, so each child gets its own binding into this shared
        state instead."""
        if not isinstance(widgets, (list, tuple)):
            widgets = [widgets]
        state = {"win": None, "hover_count": 0}
        anchor = widgets[0]

        def _show(event=None):
            state["hover_count"] += 1
            if state["win"] is not None:
                return
            win = tk.Toplevel(anchor)
            win.wm_overrideredirect(True)
            try:
                win.wm_attributes("-topmost", True)
            except Exception:
                pass
            x = anchor.winfo_rootx() - 6
            y = anchor.winfo_rooty() + anchor.winfo_height() + 4
            win.wm_geometry(f"+{x}+{y}")
            tk.Label(win, text=text, bg="#1c1c1c", fg=TEXT_BRIGHT,
                     font=("Helvetica", 9), justify="left", wraplength=260,
                     padx=8, pady=6, bd=1, relief=tk.SOLID).pack()
            state["win"] = win

        def _hide(event=None):
            state["hover_count"] = max(0, state["hover_count"] - 1)
            if state["hover_count"] == 0 and state["win"] is not None:
                state["win"].destroy()
                state["win"] = None

        for w in widgets:
            w.bind("<Enter>", _show)
            w.bind("<Leave>", _hide)
        return state

    def _help_icon(self, parent, help_text=None):
        """A small "?" glyph -- purely a visual cue now (shrunk down per
        Filip's "make the ? icons smaller" feedback); the actual hover
        trigger is the whole parameter row, wired up by the caller via
        _make_tooltip(list_of_row_widgets, ...) rather than this icon
        alone, so `help_text` here is optional/for simple standalone
        call sites only."""
        icon = tk.Label(parent, text="?", bg=CARD_BG, fg=TEXT_DIM,
                         font=("Helvetica", 6, "bold"), width=1,
                         relief=tk.RIDGE, bd=1, cursor="question_arrow")
        icon.pack(side=tk.LEFT, padx=(3, 0))
        if help_text:
            self._make_tooltip(icon, help_text)
        return icon

    def _sidebar_row(self, parent, label_text, var, width=7, suffix=None, help_text=None):
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill=tk.X, pady=2)
        lbl = tk.Label(row, text=label_text, bg=CARD_BG, fg=TEXT_MAIN,
                        font=("Helvetica", 10))
        lbl.pack(side=tk.LEFT)
        hover_widgets = [row, lbl]
        if help_text:
            hover_widgets.append(self._help_icon(row))
        if suffix:
            suf = tk.Label(row, text=suffix, bg=CARD_BG, fg=TEXT_DIM,
                            font=("Helvetica", 9))
            suf.pack(side=tk.RIGHT)
            hover_widgets.append(suf)
        entry = tk.Entry(row, textvariable=var, width=width, bg=BG_PANEL, fg="white",
                          insertbackground="white", relief=tk.FLAT,
                          font=("Helvetica", 10), justify="right")
        entry.pack(side=tk.RIGHT, padx=(0, 6))
        hover_widgets.append(entry)
        if help_text:
            self._make_tooltip(hover_widgets, help_text)
        return row

    def _sidebar_check(self, parent, text, var, help_text=None):
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill=tk.X, pady=3, anchor="w")
        chk = tk.Checkbutton(row, text=text, variable=var, bg=CARD_BG, fg=TEXT_MAIN,
                       selectcolor=CARD_BG, activebackground=CARD_BG,
                       activeforeground=TEXT_BRIGHT, font=("Helvetica", 10),
                       anchor="w", justify="left", wraplength=190)
        chk.pack(side=tk.LEFT, fill=tk.X)
        hover_widgets = [row, chk]
        if help_text:
            hover_widgets.append(self._help_icon(row))
            self._make_tooltip(hover_widgets, help_text)

    def _build_gain_sidebar(self):
        # Tk variables the old row-based _build_settings_gain used to create
        # inline as it built each widget -- now created up front since the
        # sidebar's cards are built in a different order (Load first).
        self.enf_var = tk.DoubleVar(value=1.2)
        self.fit_lo_var = tk.DoubleVar(value=0.0)
        self.fit_hi_var = tk.DoubleVar(value=50.0)
        self.spatial_bin_var = tk.IntVar(value=1)
        # 4px matches the Lees et al. 2025 reference script's own default
        # (see PTC method notes in README) -- a contaminated edge left
        # uncropped can visibly bias the fitted slope, so cropping a small
        # margin by default is the safer out-of-the-box choice than 0.
        self.margin_var = tk.IntVar(value=4)
        self.max_frames_var = tk.IntVar(value=2000)
        self.frame_start_var = tk.StringVar(value="")
        self.cnmf_gsig_var = tk.IntVar(value=6)
        self.exclude_roi_var = tk.BooleanVar(value=False)
        # Default ON + 30th pctile: raw F (Suite2p or CNMF, both a per-ROI
        # pixel SUM) includes the digitizer's black-level offset, which
        # otherwise gets miscounted as photons and scales with ROI size --
        # see photon_flux_per_cell's docstring. On by default since the
        # un-subtracted numbers are the ones that looked implausibly huge.
        self.subtract_baseline_var = tk.BooleanVar(value=True)
        self.flux_baseline_pct_var = tk.DoubleVar(value=30.0)
        self.pw_rigid_var = tk.BooleanVar(value=False)
        self.skip_mc_var = tk.BooleanVar(value=False)
        self.save_mc_var = tk.BooleanVar(value=True)

        # ── Load ──────────────────────────────────────────────────────
        load_body = self._sidebar_card("LOAD")
        seg = tk.Frame(load_body, bg=CARD_BG)
        seg.pack(fill=tk.X)
        self._lbtn(seg, "📁  Suite2p Folder", self._load_gain_folder,
                   side=tk.TOP, padx=0, fill=tk.X)
        self._lbtn(seg, "🎬  Load .mat Movie", self._load_gain_mat_movie,
                   side=tk.TOP, padx=0, fill=tk.X)

        # ── Frame rate + fit parameters ──────────────────────────────
        fit_body = self._sidebar_card("FIT PARAMETERS")
        self._sidebar_row(fit_body, "Frame rate fs", self.fs_var, suffix="Hz",
                           help_text="Acquisition frame rate in Hz. Used to convert "
                                     "photons/frame into photons/cell/s for the flux "
                                     "panel, and read automatically from ops.npy/the "
                                     ".mat file when available.")
        self._sidebar_row(fit_body, "ENF", self.enf_var,
                           help_text="Excess noise factor for a GaAsP PMT (default "
                                     "1.2). gain_true = fitted slope / ENF². Only "
                                     "matters for photon-detector setups with excess "
                                     "multiplicative noise; leave at 1.0 for a plain "
                                     "shot-noise-limited camera.")
        fitrange_help = ("Which intensity-percentile bins go into the shot-noise "
                          "linear fit (0-50 by default, matching the Lees et al. "
                          "2025 reference protocol). Raise the high end toward 98 "
                          "to use almost the full intensity range instead -- a PMT "
                          "has no strong inherent need for a read-noise floor "
                          "exclusion the way a camera sensor does.")
        fitrange_row = tk.Frame(fit_body, bg=CARD_BG)
        fitrange_row.pack(fill=tk.X, pady=2)
        fitrange_lbl = tk.Label(fitrange_row, text="Fit range (% of mean)", bg=CARD_BG,
                                 fg=TEXT_MAIN, font=("Helvetica", 10))
        fitrange_lbl.pack(side=tk.LEFT)
        fitrange_icon = self._help_icon(fitrange_row)
        fitrange_entries = tk.Frame(fit_body, bg=CARD_BG)
        fitrange_entries.pack(fill=tk.X, pady=(0, 2))
        fitrange_lo = tk.Entry(fitrange_entries, textvariable=self.fit_lo_var, width=5, bg=BG_PANEL, fg="white",
                 insertbackground="white", relief=tk.FLAT, font=("Helvetica", 10),
                 justify="center")
        fitrange_lo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        fitrange_dash = tk.Label(fitrange_entries, text="–", bg=CARD_BG, fg=TEXT_DIM)
        fitrange_dash.pack(side=tk.LEFT)
        fitrange_hi = tk.Entry(fitrange_entries, textvariable=self.fit_hi_var, width=5, bg=BG_PANEL, fg="white",
                 insertbackground="white", relief=tk.FLAT, font=("Helvetica", 10),
                 justify="center")
        fitrange_hi.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))
        self._make_tooltip([fitrange_row, fitrange_lbl, fitrange_icon, fitrange_entries,
                             fitrange_lo, fitrange_dash, fitrange_hi], fitrange_help)
        self._sidebar_row(fit_body, "Spatial bin", self.spatial_bin_var, suffix="px",
                           help_text="Superpixel binning applied before the PTC fit "
                                     "(pixels are summed, not averaged, so the "
                                     "recovered gain estimate is invariant to this "
                                     "setting). Raising it trades spatial resolution "
                                     "for fewer, less noisy bins. 1 = no binning.")
        self._sidebar_row(fit_body, "Edge margin", self.margin_var, suffix="px",
                           help_text="Crops this many pixels off every edge of each "
                                     "frame before computing anything, to exclude "
                                     "blanking/vignetting artifacts near the frame "
                                     "border that resonant-scan systems often have. "
                                     "A contaminated edge left uncropped can visibly "
                                     "bias the fitted slope. Default 4px matches the "
                                     "reference protocol; set to 0 to disable, or "
                                     "raise it if your system's blanking band is "
                                     "wider.")
        # Fit range / ENF / fs only feed the linear regression over bins
        # that a full run has already computed -- so once Run Analysis has
        # happened once, tweaking these and clicking here re-fits instantly
        # without redoing motion correction or CNMF. (Spatial bin / edge
        # margin above bake into the bins themselves, so those still need a
        # full Run Analysis to take effect.)
        self._lbtn(fit_body, "↻  Re-fit", self._refit_gain, bg=GREY_BTN,
                   font_size=10, bold=False, side=tk.TOP, padx=0, fill=tk.X)

        # ── Frame window ──────────────────────────────────────────────
        win_body = self._sidebar_card("FRAME WINDOW")
        self._sidebar_row(win_body, "Max frames", self.max_frames_var,
                           help_text="Size of the contiguous frame window used for "
                                     "the PTC estimate. The window must be "
                                     "contiguous (not evenly-spaced samples) because "
                                     "the frame-differencing noise estimate needs "
                                     "genuinely adjacent frames. Larger = more "
                                     "stable estimate, more memory/time.")
        self._sidebar_row(win_body, "Start frame", self.frame_start_var, suffix="blank=auto",
                           help_text="Explicit starting frame index for the analysis "
                                     "window. Leave blank to auto-center the window "
                                     "on the recording (the default). Set it to skip "
                                     "a noisy start-of-session period, or to target "
                                     "a specific known-bad stretch. Invalid/"
                                     "out-of-range values fall back to auto.")

        # ── CNMF ──────────────────────────────────────────────────────
        cnmf_body = self._sidebar_card("CNMF")
        self._sidebar_row(cnmf_body, "Cell radius", self.cnmf_gsig_var, suffix="px",
                           help_text="Expected cell radius (gSig) passed to CaImAn "
                                     "CNMF, used only when CNMF is offered to fill "
                                     "an empty photon-flux panel (no Suite2p F.npy "
                                     "available). The mean-projection viewer shown "
                                     "right before that prompt lets you measure a "
                                     "cell's diameter directly and auto-fills this.")
        self._sidebar_check(cnmf_body, "Exclude cell ROIs", self.exclude_roi_var,
                             help_text="Restrict the PTC fit to background/non-cell "
                                       "pixels only, using Suite2p's stat.npy masks. "
                                       "Useful if cell bodies show extra variance "
                                       "(e.g. residual motion) that biases the gain "
                                       "estimate away from pure background shot "
                                       "noise.")

        # ── Photon flux ──────────────────────────────────────────────
        flux_body = self._sidebar_card("PHOTON FLUX")
        self._sidebar_check(flux_body, "Subtract baseline (F0)", self.subtract_baseline_var,
                             help_text="F (Suite2p or CNMF) is a raw per-ROI pixel SUM "
                                       "that includes the digitizer's constant black-"
                                       "level/dark offset -- present even with zero "
                                       "real photons (visible as the nonzero floor on "
                                       "the PTC plot's Mean-ADU axis). Left in, that "
                                       "offset gets miscounted as detected photons and "
                                       "scales with ROI size, which is the usual cause "
                                       "of implausibly large flux numbers. When "
                                       "checked, each cell's own baseline (see "
                                       "'Baseline pctile' below) is subtracted before "
                                       "the photon conversion. Doesn't affect the gain "
                                       "fit itself -- only the flux panel.")
        self._sidebar_row(flux_body, "Baseline pctile", self.flux_baseline_pct_var, suffix="%",
                           help_text="Which percentile of each cell's own raw trace "
                                     "is treated as its baseline (F0) and subtracted "
                                     "when 'Subtract baseline' is checked. Lower = "
                                     "closer to the trace's true minimum (more "
                                     "aggressive); higher = closer to the median. 30 "
                                     "is a reasonable default for calcium traces that "
                                     "spend most of their time near baseline.")
        # Same story as ENF/fit range above: this only affects the flux
        # panel's conversion of already-cached F_for_flux, not the PTC fit
        # itself, so Re-fit picks it up instantly too.

        # ── Motion correction ────────────────────────────────────────
        mc_body = self._sidebar_card("MOTION CORRECTION")
        self._sidebar_check(mc_body, "Piecewise-rigid (NormCorre)", self.pw_rigid_var,
                             help_text="Use non-rigid (piecewise-rigid) motion "
                                       "correction instead of rigid. Only applies "
                                       "when NormCorre actually needs to run (no "
                                       "usable reg_tif export found). Slower, but "
                                       "corrects local/non-uniform drift that a "
                                       "single rigid shift can't.")
        self._sidebar_check(mc_body, "Skip motion correction\n(manual TIFF already registered)", self.skip_mc_var,
                             help_text="Manual-TIFF mode only (no Suite2p ops.npy "
                                       "found at all). Check this if the TIFF you "
                                       "point the tool at is already motion-"
                                       "corrected, to skip an unnecessary NormCorre "
                                       "pass.")
        self._sidebar_check(mc_body, "Save motion-corrected TIFF\n(reg_tif)", self.save_mc_var,
                             help_text="After NormCorre runs, save the result as "
                                       "reg_tif/ chunks next to the source data, "
                                       "matching Suite2p's own naming convention. "
                                       "Next time you load the same folder the tool "
                                       "finds this export and skips NormCorre "
                                       "entirely -- it only ever needs to run once "
                                       "per recording.")

        # ── Run Analysis (primary action, bottom of the sidebar) ────────
        # Was silently getting clipped off the bottom of the window before
        # the sidebar became scrollable -- there was no way to reach it.
        run_frame = tk.Frame(self._gain_sidebar_inner, bg=BG_MID)
        run_frame.pack(fill=tk.X, padx=12, pady=(4, 14))
        self._lbtn(run_frame, "▶  Run Analysis", self.run_gain_analysis,
                   bg=GREY_BTN_ACTIVE, fg=GREY_BTN_TEXT_ACTIVE, font_size=12,
                   side=tk.TOP, padx=0, fill=tk.X)

    def _toggle_settings(self):
        """Kurtosis-tab only now -- the Gain tab's settings are always
        visible in the left sidebar, no toggle needed there."""
        if self._settings_open:
            self._settings_shared.pack_forget()
            self._settings_kurtosis.pack_forget()
            self._settings_btn.config(text="⚙  Settings")
            self._settings_open = False
        else:
            self._settings_shared.pack(fill=tk.X, after=self._bar)
            self._settings_kurtosis.pack(fill=tk.X, after=self._settings_shared)
            self._settings_btn.config(text="⚙  Settings ▲")
            self._settings_open = True

    # ── tab switching ────────────────────────────────────────────────────

    def _switch_tab(self, new_tab):
        if new_tab == self.tab_var.get():
            return
        self.tab_var.set(new_tab)

        if new_tab == "kurtosis":
            self._tab_kurtosis_btn.config(bg=GREY_BTN_ACTIVE, fg=GREY_BTN_TEXT_ACTIVE)
            self._tab_gain_btn.config(bg=GREY_BTN, fg=TEXT_BRIGHT)
            self._action_btn.config(text="▶  Plot")
            self._action_btn.pack(side=tk.RIGHT, padx=10, pady=5)
            self._sort_frame.pack(side=tk.LEFT, padx=(16, 0))
            self._load_data_btn.pack(side=tk.LEFT, padx=(4, 6), pady=5)
            self._settings_btn.pack(side=tk.LEFT, padx=6, pady=5)
            self._gain_sidebar.pack_forget()
        else:
            self._tab_kurtosis_btn.config(bg=GREY_BTN, fg=TEXT_BRIGHT)
            self._tab_gain_btn.config(bg=GREY_BTN_ACTIVE, fg=GREY_BTN_TEXT_ACTIVE)
            self._sort_frame.pack_forget()
            # Gain tab has no top-bar Load/Settings/Plot -- all of that
            # lives in the always-visible sidebar now.
            self._action_btn.pack_forget()
            self._load_data_btn.pack_forget()
            self._settings_btn.pack_forget()
            if self._settings_open:
                self._toggle_settings()  # close the kurtosis panel if it was open
            self._gain_sidebar.pack(side=tk.LEFT, fill=tk.Y, before=self._main_area)

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
        # _content_row holds the (Gain-tab-only) left sidebar next to the
        # always-present main output area, so the plot/animation and the
        # sidebar can sit side by side per Filip's "options on the left
        # column" request.
        self._content_row = tk.Frame(self.root, bg=BG_DARK)
        self._content_row.pack(fill=tk.BOTH, expand=True)

        # The sidebar's card stack (LOAD / FIT PARAMETERS / FRAME WINDOW /
        # CNMF / PHOTON FLUX / MOTION CORRECTION / Run Analysis) is taller
        # than the window at the default 990px height, which silently
        # clipped "Run Analysis" off the bottom with no way to reach it --
        # so the sidebar is a scrollable Canvas+Frame rather than a plain
        # Frame. self._gain_sidebar is still the thing _switch_tab packs/
        # unpacks; card-building (_sidebar_card) targets the inner
        # self._gain_sidebar_inner frame that actually scrolls.
        self._gain_sidebar = tk.Frame(self._content_row, bg=BG_MID, width=260)
        self._gain_sidebar.pack_propagate(False)
        # not packed here -- _switch_tab packs/unpacks it depending on tab

        sidebar_canvas = tk.Canvas(self._gain_sidebar, bg=BG_MID, highlightthickness=0, bd=0)
        sidebar_scrollbar = tk.Scrollbar(self._gain_sidebar, orient="vertical",
                                          command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._gain_sidebar_inner = tk.Frame(sidebar_canvas, bg=BG_MID)
        inner_win = sidebar_canvas.create_window((0, 0), window=self._gain_sidebar_inner, anchor="nw")

        def _on_inner_configure(event=None):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
        self._gain_sidebar_inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event):
            sidebar_canvas.itemconfig(inner_win, width=event.width)
        sidebar_canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse-wheel scroll, only while the pointer is actually over the
        # sidebar (bound/unbound on Enter/Leave) so it doesn't hijack
        # scrolling anywhere else in the app. Covers Windows/macOS
        # (<MouseWheel>, event.delta) and Linux (<Button-4>/<Button-5>).
        def _on_mousewheel(event):
            if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
                sidebar_canvas.yview_scroll(1, "units")
            elif getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                sidebar_canvas.yview_scroll(-1, "units")

        def _bind_wheel(event=None):
            sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            sidebar_canvas.bind_all("<Button-4>", _on_mousewheel)
            sidebar_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(event=None):
            sidebar_canvas.unbind_all("<MouseWheel>")
            sidebar_canvas.unbind_all("<Button-4>")
            sidebar_canvas.unbind_all("<Button-5>")

        sidebar_canvas.bind("<Enter>", _bind_wheel)
        sidebar_canvas.bind("<Leave>", _unbind_wheel)

        self._main_area = tk.Frame(self._content_row, bg=BG_DARK)
        self._main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(facecolor=BG_DARK)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self._main_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # No toolbar — removed intentionally

        # Busy animation lives in the same main output area, swapped in for
        # the canvas by _progress_start/_progress_stop while an analysis is
        # running -- not packed until then.
        self.gain_progress_bar = PhotonNeuronBar(self._main_area)

        self._build_gain_sidebar()
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
        """Swap the plot canvas out for the photon-hits-neuron busy
        animation in the main output area, and start it.
        Safe to call from any thread (marshals onto the main loop)."""
        def _go():
            self.canvas.get_tk_widget().pack_forget()
            self.gain_progress_bar.pack(fill=tk.X, expand=True)
            self.gain_progress_bar.start()
        self.root.after(0, _go)

    def _progress_stop(self):
        """Stop the busy animation, swap the plot canvas back in.
        Safe to call from any thread (marshals onto the main loop)."""
        def _go():
            self.gain_progress_bar.stop()
            self.gain_progress_bar.pack_forget()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.root.after(0, _go)

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
                # Inlined rather than calling self._flux_baseline_pct(): a
                # few tests bind just this method onto a lightweight fake
                # app object via types.MethodType, which doesn't have
                # _flux_baseline_pct available as a *callable* attribute
                # (only whatever was explicitly bound) -- getattr on the
                # *variable* here degrades gracefully instead.
                subtract_var = getattr(self, "subtract_baseline_var", None)
                baseline_pct = (float(self.flux_baseline_pct_var.get())
                                 if subtract_var is not None and subtract_var.get()
                                 else None)
                flux = photon_flux_per_cell(F_for_flux, fit["gain_true"], fs,
                                             baseline_pct=baseline_pct)
                med, sem = bootstrap_median_sem(flux)
            else:
                flux, med, sem = None, np.nan, np.nan

            self.g_results = dict(movie_source=movie_source, movie_note=movie_note,
                                   movie_shape=movie_shape, has_cells=has_cells,
                                   cnmf_used=cnmf_used, n_cells=(len(F_for_flux) if has_cells else 0),
                                   mu_flat=mu_flat, var_flat=var_flat,
                                   mu_bin=mu_bin, var_bin=var_bin, n_bin=n_bin,
                                   fit=fit, flux=flux, flux_med=med, flux_sem=sem,
                                   # cached only so "Re-fit" (below) can redo the cheap
                                   # part -- fit_gain() + photon_flux_per_cell() -- from
                                   # a changed Fit range / ENF without re-running
                                   # motion correction or CNMF. F_for_flux is small
                                   # (n_cells x n_frames), nowhere near the size of the
                                   # movie array this function already dropped above.
                                   F_for_flux=(F_for_flux if has_cells else None))

            self._progress_stop()
            self.root.after(0, self._render_gain_results)
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            self._progress_stop()
            self._motion_overlay_stop()  # safety net -- should already be closed via the finally in _normcorre_then_maybe_save
            self.root.after(0, lambda: messagebox.showerror("Analysis failed", msg))
            self.root.after(0, lambda: self.status_var.set("Analysis failed — see error dialog"))

    def _flux_baseline_pct(self):
        """The percentile-baseline to subtract from raw F before the
        photon-flux conversion, or None if 'Subtract baseline (F0)' is
        unchecked (keeps raw, un-subtracted F -- see photon_flux_per_cell's
        docstring for why raw F alone tends to read implausibly high).

        Uses getattr rather than a direct attribute access because several
        tests build a lightweight fake app (types.SimpleNamespace, not a
        real KurtosisChecker) and bind just _run_gain_worker/_refit_gain
        onto it without setting every sidebar var -- falling back to None
        (the original raw-F behavior) keeps those tests focused on what
        they're actually exercising instead of forcing every one of them
        to also stub out this unrelated setting."""
        var = getattr(self, "subtract_baseline_var", None)
        if var is not None and var.get():
            return float(self.flux_baseline_pct_var.get())
        return None

    def _refit_gain(self):
        """Re-run just the cheap part of the analysis -- the linear PTC fit
        and (if there are cell traces) the photon-flux conversion -- using
        whatever Fit range / ENF / frame rate / baseline settings are
        currently in the sidebar, against the mean/variance bins +
        F_for_flux already cached in g_results from the last full Run
        Analysis.

        Deliberately does NOT touch the movie, motion correction, or CNMF --
        those bins are a function of the raw pixel data (spatial bin, edge
        margin, exclude-ROI mask, frame window all bake into them at
        compute_ptc() time), so changing *those* settings still requires a
        full Run Analysis. Fit range, ENF, and the baseline-subtraction
        settings, by contrast, only rework numbers that already exist (the
        cached bins, and the cached raw F_for_flux respectively) -- this is
        instant and needs no thread."""
        r = self.g_results
        if r is None:
            self.status_var.set("Nothing to re-fit yet — click Run Analysis first")
            return

        fit = fit_gain(r["mu_bin"], r["var_bin"], r["n_bin"],
                        self.fit_lo_var.get(), self.fit_hi_var.get(), self.enf_var.get())
        r["fit"] = fit

        if r["has_cells"] and r.get("F_for_flux") is not None:
            fs = max(0.1, float(self.fs_var.get()))
            flux = photon_flux_per_cell(r["F_for_flux"], fit["gain_true"], fs,
                                         baseline_pct=self._flux_baseline_pct())
            med, sem = bootstrap_median_sem(flux)
            r["flux"], r["flux_med"], r["flux_sem"] = flux, med, sem

        self._render_gain_results()
        self.status_var.set(
            f"Re-fit with ENF={self.enf_var.get():.2f}, fit range "
            f"{self.fit_lo_var.get():.0f}–{self.fit_hi_var.get():.0f}% -- "
            f"gain_true={fit['gain_true']:.2f} ADU/photon")

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
