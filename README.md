# Kurtosis / Gain Estimation Checker

A desktop tool for two-photon calcium imaging QC, built around Suite2p output. It has two modes, toggled from the top bar:

- **Kurtosis** — sorts and visualizes ROI traces by kurtosis or SNR to spot noisy or non-cell-like extractions.
- **Gain Estimation** — runs a photon-transfer-curve ("hockey stick") analysis on a Suite2p recording to recover true PMT/camera gain (ADU/photon) and convert each cell's fluorescence to photons/cell/s.

This README focuses on **Gain Estimation** mode, since that's the part with real setup decisions.

## Installation

```
pip install numpy scipy matplotlib pillow tifffile
python kurtosis_checker.py
```

That's everything you need if you already have registered TIFFs saved (see Case 1 below). CaImAn is an optional, heavier dependency only needed if Suite2p's `reg_tif` export is missing (Case 2).

## Gain Estimation: two setup cases

Click **Gain Estimation** in the top bar, then **Load Data** and select any folder at or above your Suite2p output. The tool recursively searches *downstream* of whatever folder you select for a directory containing `ops.npy` — there's no fixed list of expected folder names, so it works whether you're parked at the session folder, a custom intermediate folder (e.g. a `small/` subfolder for a downsampled test run), the `suite2p` folder, or `plane0` itself directly.

If the tree contains **more than one** `ops.npy` — for example an older full-resolution run sitting alongside a newer downsampled one — the search specifically prefers whichever one has a *usable* `reg_tif` export over one that doesn't, rather than just grabbing the first/shallowest match. That matters: picking a plane with cell traces but no registered movie silently sends you down the slow CaImAn path for no reason.

### Case 1 — You have registered TIFFs saved (recommended, no CaImAn needed)

If you ran Suite2p with the `reg_tif` option enabled, your plane folder has a `reg_tif/` subfolder containing the motion-corrected movie as TIFF chunks (e.g. `file000_chan0.tif`, `file001_chan0.tif`, ...). This is the tool's preferred source and requires nothing beyond the core dependencies above.

1. **Load Data** → select the folder. The status bar shows the *full resolved path* plus an honest `reg_tif found (N file(s))` / `no usable reg_tif` hint, computed with the same file-matching logic Run Analysis uses — so this can never claim reg_tif is available and then fall back anyway.
2. Click **Run Analysis**.
3. The tool reads only the chunk files it needs for a contiguous window of frames (see *Memory behavior* below), computes the PTC gain fit, and shows:
   - Left panel: the hockey-stick plot (mean vs. variance, shot-noise fit, read-noise floor)
   - Right panel: per-cell photon flux histogram (median ± bootstrap SEM)

**Notes:**
- Suite2p's own `data.bin` binary is deliberately *never* used as a fallback, even if present — it's a scratch/working file Suite2p can overwrite on a later run, so it isn't a trustworthy record of the frames that produced your `F.npy`. If there's no `reg_tif` export, the tool goes straight to Case 2 rather than silently trusting `data.bin`.
- File matching inside `reg_tif/` is case-insensitive and doesn't require `chan0` in the filename. If a `reg_tif/` folder exists but genuinely has no readable TIFFs inside it, the tool raises an error listing that folder's contents rather than silently falling back to Case 2.

### Case 2 (optional) — You don't have motion-corrected TIFFs saved

If `reg_tif` wasn't exported, the tool falls back to the raw acquisition TIFFs and motion-corrects them on the fly using **NoRMCorre**, via **CaImAn**. This is slower and heavier, so it's opt-in territory rather than the default path.

Install CaImAn (compiled dependencies mean mamba/conda is the supported route — plain `pip install caiman` is not):

```
conda install -n base -c conda-forge mamba
mamba create -n caiman caiman
conda activate caiman
```

Windows note: if `conda install` fails with `SSL module is not available`, that's a broken/mismatched OpenSSL DLL pairing, not a real certificate problem. Try the [fix-conda-ssl](https://github.com/davidfokkema/fix-conda-ssl) tool, or do a clean Miniforge/Miniconda reinstall if the manual fix doesn't resolve it.

Once CaImAn is installed:

1. **Load Data** → select the folder. The status bar shows `no reg_tif export — will need NormCorre on raw TIFFs`.
2. Click **Run Analysis**. The tool tries to resolve the raw TIFF paths from `ops.npy` automatically; if those paths don't exist on this machine, it will prompt you to locate the raw TIFF folder manually.
3. NoRMCorre runs (rigid by default; enable **piecewise-rigid** in Settings for non-rigid correction), then the PTC analysis proceeds as in Case 1.

**Saving the motion-corrected movie:** the **"Save motion-corrected TIFF (reg_tif)"** setting (checked by default) writes the NormCorre output as `reg_tif/file000_chan0.tif`-style chunks next to the source data, matching Suite2p's own naming convention. Next time you load the same folder, the tool finds this `reg_tif/` export and uses it directly — NormCorre only ever needs to run once per recording.

**CNMF segmentation prompt:** any time there's no `F.npy` available for the loaded dataset (so the photon-flux panel would otherwise stay empty) — whether the movie came from a fresh NormCorre run or was loaded straight from an existing `reg_tif` export that just happens to be missing its Suite2p segmentation — the tool asks whether to run CaImAn's CNMF on the registered movie to detect cells and extract traces on the spot. Accepting fills in the flux panel using CNMF-derived footprints; the results panel labels these cells "CaImAn CNMF (unverified — sanity-check!)" in place of the usual "Suite2p F.npy" label, as a reminder to check the footprints/traces before trusting the numbers. The **"CNMF cell radius (px)"** setting (default 6) controls the expected cell size (`gSig`) passed to CNMF.

Right before that prompt, the tool shows a zoomable/pannable **mean-projection viewer** of the just-loaded registered movie — a standard matplotlib zoom/pan toolbar with a live pixel-coordinate readout — specifically so you can count how many pixels a cell spans. The window then asks directly for the cell diameter (px) you measured; entering one converts it to a radius (`diameter / 2`, floored at 2) and writes it straight into the **"CNMF cell radius (px)"** setting before the CNMF prompt fires, so you don't have to go find that setting yourself. Leave it blank to keep whatever the setting already was.

CNMF's `fit()` call has been observed to return `None` on some CaImAn versions/installs while still mutating its own state in place; the tool now falls back to that mutated object instead of crashing with a bare `'NoneType' object has no attribute 'estimates'` error, and only raises a real error if CNMF genuinely produced no usable components.

**Component-quality filtering:** after CNMF finds candidate components, the tool calls CaImAn's own `evaluate_components()` and keeps only the **top 60%** by whichever quality score that produces — CaImAn's CNN-classifier probability (`cnn_preds`, a real 0–1 probability) if a CNN model is installed and usable, otherwise the `r_value` spatial-consistency score as a fallback (this is **not** a true probability, just a ranking proxy — labeled as such wherever it's shown). If quality evaluation isn't available at all (older CaImAn install, no CNN model), all components are kept rather than losing the run over an optional metrics step.

Right after CNMF finishes, a **footprint sanity-check popup** shows the mean projection with the surviving masks outlined in green and the quality-filtered-out ones in red, so you can actually look at whether the green outlines look like real cells before trusting the flux numbers — instead of just trusting a histogram. It's non-blocking (doesn't hold up the rest of the analysis) and has the same zoom/pan toolbar as the mean-projection viewer.

**Progress feedback (Gain Estimation tab only — not shown in Kurtosis mode):** below the status line, a chameleon (the light source) lobs photon squiggles at a cartoon neuron for the whole duration of Run Analysis — each impact ticks a counter (drawn as classic tally/gate marks — four verticals plus a diagonal fifth stroke per group of five) and swaps the neuron's expression, purely decorative so the app never looks frozen during reg_tif reads, PTC computation, or CNMF. The artwork is vectorized directly from Filip's own reference sketches (uploaded as real SVG files). During NormCorre specifically, a separate big centered popup takes over: a progress bar plus the neuron redrawn with a scanline "distortion" that resolves into the clean drawing as CaImAn's own log output comes in. That bar is a step-count heuristic, not a true percentage — CaImAn's `motion_correct()` is one blocking call with no frame-level progress exposed, so each distinct log line nudges the bar forward (capped short of 100% until the call actually returns).

### No Suite2p output at all

If the selected folder has no `ops.npy` anywhere, the tool offers to run the PTC gain estimate directly on manually-selected TIFF file(s), bypassing Suite2p entirely. Without `F.npy`, only the gain estimate is available — the per-cell photon-flux panel is unavailable in this mode. A **"Skip motion correction"** setting lets you tell it the TIFF is already registered, avoiding an unnecessary NormCorre pass.

## Memory behavior

Gain Estimation mode is deliberately conservative about RAM:

- The registered movie is **only** loaded when you click **Run Analysis** — never on folder load, never on tab switch.
- Above ~1.5 GB estimated usage, a confirmation dialog shows the estimate before loading anything.
- For chunked `reg_tif` exports, only the chunk files overlapping the needed frame window are decoded — file headers are probed first (no pixel data) to figure out which chunks are actually needed.
- Frame subsampling (`Max frames` setting) always picks a **contiguous** window of frames, not evenly-spaced samples — the PTC's frame-differencing noise estimate requires genuinely adjacent frames. By default that window is auto-centered on the recording; the **"Start frame"** setting lets you pin it to an explicit starting index instead (e.g. to skip a noisy start-of-session period, or to specifically analyze a known-bad stretch) — leave it blank to keep the auto-centered default. An out-of-range or invalid value is treated the same as blank rather than erroring.
- `reg_tif`/raw-TIFF arrays are kept in their native dtype (typically int16) rather than eagerly upcast to float32; the PTC computation streams frame-pairs in chunks so peak memory doesn't scale with total recording length.
- The raw movie array is dropped from memory immediately after the PTC bins are computed.

## Key settings (Gain Estimation)

| Setting | Default | Meaning |
|---|---|---|
| ENF | 1.2 | Excess noise factor (GaAsP PMT); `gain_true = slope / ENF²` |
| Fit range (pct of mean) | 0–50 | Which intensity-percentile bins are used for the shot-noise linear fit. Matches the Lees et al. 2025 reference protocol's default (lower half of the intensity range only). Raise the high end toward 98 if you want to use (almost) the full range instead — a PMT has no strong inherent reason to need a read-noise floor excluded like a camera sensor would, so the wider range is also a defensible choice; see *PTC method notes* below. |
| Spatial bin (px) | 1 | Superpixel binning before the PTC fit (summed, not averaged — gain estimate is invariant to this setting) |
| Max frames | 2000 | Size of the contiguous frame window used for the PTC estimate |
| Start frame | blank (auto-center) | Explicit starting frame index for that window; blank keeps the original auto-centered behavior. Invalid/negative/out-of-range values fall back to auto rather than erroring |
| Edge margin (px) | 0 | Crops this many pixels off every edge of each frame before computing anything, to exclude blanking/vignetting artifacts near the frame border (see *PTC method notes* below). 0 preserves the original no-cropping behavior. |
| Exclude cell ROIs | off | Restrict the PTC fit to background/non-cell pixels only |
| NormCorre: piecewise-rigid | off | Use non-rigid motion correction instead of rigid (Case 2 only) |
| Skip motion correction | off | Manual-TIFF mode only — skip NormCorre if the TIFF is already registered |
| Save motion-corrected TIFF (reg_tif) | on | After NormCorre runs, save the result as `reg_tif/` chunks next to the source so future runs skip motion correction entirely |
| CNMF cell radius (px) | 6 | Expected cell radius (`gSig`) passed to CaImAn CNMF, used only when it's offered to fill an empty flux panel after a fresh NormCorre run |

Photon flux uses **raw F only** — no neuropil subtraction — consistent with the Wilt et al. convention that F0 includes all detected photons (signal + background).

## PTC method notes (vs. the Lees et al. 2025 reference protocol)

Checked against a MATLAB reference implementation of the Lees et al. 2025 (*Nature Protocols*, Procedure 7 / Box 7-8) method. The core statistics already match exactly: adjacent-frame mean `M = 0.5*(X + X')`, adjacent-frame variance `D = 0.5*(X - X')²` (averaged per-pixel over all frame pairs, so slow structural signal cancels and only shot noise survives), and the `gain_true = slope / ENF²` correction are identical formulas in both.

Two places the tools genuinely differed, one now fixed and one left as an open choice:

- **Edge cropping (fixed):** the reference script crops a fixed margin off every frame edge before computing anything, to exclude blanking/vignetting artifacts that resonant-scan two-photon systems can have near the frame border. This tool had no equivalent at all until the **"Edge margin (px)"** setting above — default 0 (no change to prior results), set it to match your acquisition system's blanking width (the reference script used 4px for its dataset). A contaminated edge with the wrong gain, left uncropped, can visibly bias the fitted slope (confirmed in `test_ptc_margin.py`); a 4px crop restored the correct value in that test.
- **Fit-range default (changed to match):** the reference script fits only the **lower 50%** of the intensity range. This tool's default now matches (`Fit range (pct of mean)` = 0–50, changed from a prior 0–98 default) — raise the high end back toward 98 per-run if you want the wider range instead (see the settings table above for the tradeoff).
- **Binning (left as-is, flagged here):** the reference script bins by unique rounded ADU value (mean-aggregated, weighted by `sqrt(pixel count)`). This tool bins into 40 percentile-spaced bins instead (median-aggregated, weighted by raw pixel count — standard weighted least squares). Both are defensible; median binning is more robust to the kind of localized artifact pixels the new edge-margin setting targets. Not changed, since it's a genuinely different (not clearly worse) design choice rather than a gap.

## Repo contents

- `kurtosis_checker.py` — the app (single file, tkinter + matplotlib)
- `hockey_stick_requirements.txt` — dependency list, including the CaImAn/NormCorre setup notes above
