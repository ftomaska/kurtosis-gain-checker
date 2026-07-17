# Kurtosis / Gain Estimation Checker

A desktop tool for two-photon calcium imaging QC, built around Suite2p output. It has three modes, toggled from the top bar:

- **Kurtosis** — sorts and visualizes ROI traces by kurtosis or SNR to let the user assess whether one or the other can sort their data better
- **Gain Estimation** — runs a photon-transfer-curve analysis on a Suite2p recording to estimate PMT/camera gain (ADU/photon) and convert each cell's fluorescence to photons/cell/s. Option to use raw recordings that can be motion corrected (requires caiman package) and segmented.
- **Neuropil Sweep** — Suite2p only. Sweeps the neuropil-subtraction coefficient alpha (`Fcorr = rawF - alpha*Fneu`) and plots the mean ± SEM pairwise cell-cell correlation at each alpha, to help pick a good alpha for your recording.

## Installation

```
pip install numpy scipy matplotlib pillow tifffile
python kurtosis_checker.py
```

That's everything you need if you already have registered TIFFs saved (see Case 1 below). CaImAn is an optional, heavier dependency only needed if Suite2p's `reg_tif` export is missing (Case 2), if a Raw Movie needs motion correction, or if you choose CNMF segmentation. `h5py` is optional, only needed if you're loading a `.mat` movie saved in MATLAB's v7.3/HDF5 format.

## Demo

*(placeholder for a demo GIF)*

![demo](demo.gif)

## Interface

**Layout:** the Gain Estimation and Neuropil Sweep tabs each have an always-visible left sidebar (Load buttons, then settings grouped into rounded-corner cards, then Run Analysis), scrollable independently once its cards outgrow the window height. The Kurtosis tab has a top-bar Load Data + collapsible Settings, plus a **▶ Plot** button. Every sidebar parameter has a small "?" glyph next to it — click it to toggle a popup explaining what the setting does.

**De-whimsify the GUI:** a button at the far right of the top bar, visible on every tab. Click it and every 'creative' 100% not ai generated animation in the app is removed or replaced with a generic loading bar. I did go a little bit overboard. If you want it to go away that's totally fair. But if you change your mind just click again to bring all the whimsy back.

**Character "art":** the artwork is hand-drawn, rasterized from vector source files and embedded directly in `kurtosis_checker.py` as base64 PNG data — no separate `art/` folder is needed to run the app. 

**PTC reference citation:** the Gain Estimation sidebar's Export card links directly to the Lees et al. 2025 photon-transfer-curve reference protocol citation, with the DOI as a clickable link. (See Box 8 and Fig.16).


## Gain Estimation: two setup cases

Click **Gain Estimation** in the top bar, then in the left sidebar's **LOAD** card click **📁 Suite2p Folder** and select any folder at or above your Suite2p output. The tool recursively searches *downstream* of whatever folder you select for a directory containing `ops.npy` — there's no fixed list of expected folder names, so it works whether you're parked at the session folder, a custom intermediate folder, the `suite2p` folder, or `plane0` itself directly.

If the tree contains **more than one** `ops.npy`, the search prefers whichever one has a *usable* `reg_tif` export over one that doesn't, rather than just grabbing the first/shallowest match.

### Case 1 — You have registered TIFFs saved (recommended, no CaImAn needed)

If you ran Suite2p with the `reg_tif` option enabled, your plane folder has a `reg_tif/` subfolder containing the motion-corrected movie as TIFF chunks (e.g. `file000_chan0.tif`, `file001_chan0.tif`, ...). This is the tool's preferred source and requires nothing beyond the core dependencies above.

1. **Suite2p Folder** → select the folder. The status bar shows the full resolved path plus a `reg_tif found (N file(s))` / `no usable reg_tif` hint.
2. Click **Run Analysis** (bottom of the sidebar).
3. The tool reads only the chunk files it needs for a contiguous window of frames (see *Memory behavior* below), computes the PTC gain fit, and shows:
   - Left panel: the Photon Transfer Curve (mean vs. variance, shot-noise fit)
   - Right panel: per-cell photon flux histogram (median)

**Notes:**
- Suite2p's own `data.bin` binary is never used as a fallback, even if present — it's a scratch/working file Suite2p can overwrite on a later run. If there's no `reg_tif` export, the tool goes straight to Case 2.
- File matching inside `reg_tif/` is case-insensitive and doesn't require `chan0` in the filename. If a `reg_tif/` folder exists but has no readable TIFFs inside it, the tool raises an error listing that folder's contents. If you have multiple channels maybe copy them out to a separate file. Or email me. If this video gets 10 likes I will add the channel selection option.

### Case 2 (optional) — You don't have motion-corrected TIFFs saved

If `reg_tif` wasn't exported, the tool falls back to the raw acquisition TIFFs and motion-corrects them on the fly using **NoRMCorre**, via **CaImAn**.

Install CaImAn (compiled dependencies mean mamba/conda is the supported route — plain `pip install caiman` is not):

```
conda install -n base -c conda-forge mamba
mamba create -n caiman caiman
conda activate caiman
```

Windows note: if `conda install` fails with `SSL module is not available`, that's a broken/mismatched OpenSSL DLL pairing, not a real certificate problem. Try the [fix-conda-ssl](https://github.com/davidfokkema/fix-conda-ssl) tool, or do a clean Miniforge/Miniconda reinstall if the manual fix doesn't resolve it.

Once CaImAn is installed:

1. **Suite2p Folder** → select the folder. The status bar shows `no reg_tif export — will need NormCorre on raw TIFFs`.
2. Click **Run Analysis**. The tool tries to resolve the raw TIFF paths from `ops.npy` automatically; if those paths don't exist on this machine, it will prompt you to locate the raw TIFF folder manually.
3. NoRMCorre runs (rigid by default; enable **piecewise-rigid** in the sidebar's MOTION CORRECTION card for non-rigid correction), then the PTC analysis proceeds as in Case 1.

**Saving the motion-corrected movie:** the **"Save motion-corrected TIFF (reg_tif)"** setting (checked by default) writes the NormCorre output as `reg_tif/file000_chan0.tif`-style chunks next to the source data, matching Suite2p's own naming convention. Next time you load the same folder, the tool finds this `reg_tif/` export and uses it directly.

**Cell segmentation when there's no `F.npy`:** any time there's no `F.npy` available for the loaded dataset — Case 2, the Raw Movie flow below, or the no-Suite2p fallback further down — the tool asks how to fill the photon-flux panel, with three choices: run CaImAn **CNMF** on the registered movie to detect cells and extract traces automatically; draw **Manual ROIs** yourself; or **Skip** and leave the flux panel empty (the gain estimate itself doesn't need cell traces).

Right before that choice, the tool shows a zoomable/pannable **mean-projection viewer** of the just-loaded registered movie, so you can count how many pixels a cell spans. It then asks directly for the cell diameter (px), converts it to a radius, and writes it into the **"CNMF cell radius (px)"** setting.

**CNMF:** fills the flux panel using CNMF-derived footprints; the results panel labels these cells "CaImAn CNMF (unverified — sanity-check!)" as a reminder to check footprints/traces before trusting the numbers. The **"CNMF cell radius (px)"** setting (default 6) controls the expected cell size (`gSig`) passed to CNMF. After CNMF finds candidate components, the tool calls CaImAn's own `evaluate_components()` and keeps only the top 60% by whichever quality score that produces — CaImAn's CNN-classifier probability if a CNN model is installed and usable, otherwise the `r_value` spatial-consistency score as a fallback. If quality evaluation isn't available at all, all components are kept.
Note: CNMF might run for a while. I was considering adding cellpose, but we already have CaImAn from motion correction and I want to keep the number of dependencies minimal. I do like cellpose though. 

**Manual ROIs:** opens a polygon-drawing tool over the same mean projection — click to place points, click the first point again (or press Enter) to close the polygon, then **Add ROI** to glue it on and start the next one, and **Done** when finished. Each banked polygon becomes a cell, with its photon flux computed the same way as every other path: an unweighted sum of raw pixel values over the ROI mask. Results are labeled "Manual ROIs (user-drawn)" in the flux panel.

A **footprint sanity-check popup** shows the mean projection with the resulting masks outlined in green (and, for CNMF, the quality-filtered-out ones in red) — non-blocking, with the same zoom/pan toolbar as the mean-projection viewer, for both CNMF and Manual ROIs.

The animation runs on Tk's main-thread event loop (~30fps) while the actual PTC/segmentation computation runs in a separate background thread. That doesn't mean the animation isn't slowing down the compute time, but it is so worth it.

### Raw Movie — load a TIFF, .mat, or .npy movie directly

The **"🎬 Raw Movie"** button (in the sidebar's LOAD card, next to Suite2p Folder) loads a movie straight from a file instead of a Suite2p folder. It asks two questions:

1. **Already motion corrected, or raw?** "Already motion corrected" loads the movie as-is and skips NormCorre entirely — useful when motion correction was already done somewhere else (e.g. in MATLAB), or when NormCorre keeps failing with `OSError: [Errno 28] No space left on device`. "Raw — needs motion correction" runs it through the same NoRMCorre/CaImAn pipeline as Case 2, exactly as if it were a manually-selected raw TIFF folder — for a raw `.mat`/`.npy` source, the movie is first written out to temporary TIFF chunk files so it can go through that same pipeline unchanged.
2. **File format:** TIFF, `.mat`, or `.npy`.

For `.mat` files, both classic (pre-v7.3) and v7.3/HDF5 formats are supported (the latter needs `pip install h5py`); the expected matrix shape is **(x, y, time)**, MATLAB's own `size(mov)` convention, and if the file has more than one 3D array in it, the largest one is picked automatically. For `.npy` files, the array is expected already shaped **(time, y, x)** — numpy's own natural frame-stacking order, no axis transpose applied. Either way, a confirmation dialog shows the detected shape before committing.

The **"Scratch folder"** setting in the sidebar's MOTION CORRECTION card (click **"📁 Choose…"**, or **"Reset"** to go back to CaImAn's default) points NormCorre's memmap writes at a folder on a drive with more room, for the disk-space case above.

A Raw Movie load has no Suite2p `F.npy`, so it goes through the same three-way segmentation choice (CNMF / Manual ROIs / Skip) described above to fill in the photon-flux panel.

### No Suite2p output at all

If the selected folder has no `ops.npy` anywhere, the tool offers to run the PTC gain estimate directly on manually-selected TIFF file(s), bypassing Suite2p entirely. A **"Skip motion correction"** setting lets you tell it the TIFF is already registered, avoiding an unnecessary NormCorre pass. This path also has no `F.npy`, so it goes through the same three-way segmentation choice (CNMF / Manual ROIs / Skip) described above.

## Memory behavior

Gain Estimation mode is deliberately conservative about RAM as it kept crashing my laptop:

- The registered movie is only loaded when you click **Run Analysis** — never on folder load, never on tab switch.
- Above ~1.5 GB estimated usage, a confirmation dialog shows the estimate before loading anything.
- For chunked `reg_tif` exports, only the chunk files overlapping the needed frame window are decoded.
- Frame subsampling (`Max frames` setting) always picks a contiguous window of frames, not evenly-spaced samples. By default that window is auto-centered on the recording; the **"Start frame"** setting lets you pin it to an explicit starting index instead.
- `reg_tif`/raw-TIFF arrays are kept in their native dtype (typically int16) rather than eagerly upcast to float32; the PTC computation streams frame-pairs in chunks so peak memory doesn't scale with total recording length.
- The raw movie array is dropped from memory immediately after the PTC bins are computed.
- Spatial binning is possible to reduce size, but is set to 1 by default (no binning).

## Key settings (Gain Estimation)

| Setting | Default | Meaning |
|---|---|---|
| ENF | 1.2 | Excess noise factor (GaAsP PMT); `gain_true = slope / ENF²` |
| Fit range (pct of mean) | 0–50 | Which intensity-percentile bins are used for the shot-noise linear fit. Matches the Lees et al. 2025 reference protocol's default (lower half of the intensity range only). Raise the high end toward 98 to use (almost) the full range instead — see *PTC method notes* below. |
| Spatial bin (px) | 1 | Superpixel binning before the PTC fit (summed, not averaged — gain estimate is invariant to this setting) |
| Max frames | 2000 | Size of the contiguous frame window used for the PTC estimate |
| Start frame | blank (auto-center) | Explicit starting frame index for that window; blank keeps the auto-centered behavior |
| Edge margin (px) | 4 | Crops this many pixels off every edge of each frame before computing anything, to exclude blanking/vignetting artifacts near the frame border (see *PTC method notes* below). Matches the Lees et al. 2025 reference protocol's own default; set to 0 to disable cropping entirely. |
| Exclude cell ROIs | off | Restrict the PTC fit to background/non-cell pixels only |
| NormCorre: piecewise-rigid | off | Use non-rigid motion correction instead of rigid (Case 2 only) |
| Skip motion correction | off | Manual-TIFF mode only — skip NormCorre if the TIFF is already registered |
| Save motion-corrected TIFF (reg_tif) | on | After NormCorre runs, save the result as `reg_tif/` chunks next to the source so future runs skip motion correction entirely |
| CNMF cell radius (px) | 6 | Expected cell radius (`gSig`) passed to CaImAn CNMF, used only when it's offered to fill an empty flux panel after a fresh NormCorre run |
| Use baseline (F0) for flux | on | Converts each cell's own low-percentile baseline value directly to photons/s, instead of the trace's plain mean |
| Baseline pctile | 30 | Which percentile of each cell's own raw trace counts as its baseline (F0) when the above is checked |

Photon flux uses **raw F only** — no neuropil subtraction — consistent with the Wilt et al. (2013) convention that F0 includes all detected photons (signal + background light). F itself is never modified for the flux calculation; what changes is which per-cell scalar gets converted to photons/s. With **Use baseline (F0) for flux** unchecked, that's the plain per-frame mean of the whole trace. Checked (the default), it's each cell's own baseline — the `Baseline pctile`-th percentile of its raw trace — converted directly.

Both Suite2p's `F.npy` and CaImAn CNMF traces are put on a comparable per-cell-total scale before this conversion, but they get there differently: CNMF traces (via `footprints_to_raw_traces`) are already a true unweighted pixel sum over each ROI mask. Suite2p's `F.npy` normalizes its per-pixel ROI weights (`lam`) to sum to 1 before extraction, so it's actually a weighted mean pixel value. The Suite2p loader corrects for this at load time by multiplying `F` (and `Fneu`) by each ROI's pixel count (`npix`, from `stat.npy`), recovering a per-cell total that matches the CNMF path's convention.

**Re-fit without a full re-run:** after Run Analysis has completed once, the **↻ Re-fit** button (under Fit range/ENF in the sidebar) redoes just the linear PTC fit — and, if there's a flux panel, rescales it — using the mean/variance bins already computed. It's instant (no thread, no movie re-read). Frame rate `fs` is also picked up live by Re-fit. Spatial bin, Edge margin, Exclude cell ROIs, and anything CNMF/motion-correction related bake into the bins themselves during the raw-pixel pass, so changing those still needs a full Run Analysis.

## Exporting results for cross-condition comparison

Once Run Analysis has completed, the **💾 Export Results…** button (EXPORT card in the sidebar, just above Run Analysis) writes the current run's numbers to disk so results from different conditions/sessions/animals can be loaded into one table and plotted against each other outside the app. Clicking it asks for a short label for this run and a destination folder, then writes:

- **`gain_results_summary.csv`** — one row per export, appended across calls: timestamp, label, source, cell count, `gain_true`/slope/intercept/r², flux median ± SEM, and every sidebar setting that affected the run.
- **`{label}_{timestamp}_ptc_bins.csv`** — that run's PTC mean/variance bins (`mu_bin`, `var_bin`, `n_bin`, and whether each bin was used in the fit), enough to redraw the hockey-stick plot exactly.
- **`{label}_{timestamp}_flux_percell.csv`** — one row per cell's photon flux, only written if that run had a flux panel at all.

The summary row records each detail file's name, so a given condition's summary row can always be matched back to its own bins/flux files.

## Neuropil Sweep

The third tab, Suite2p-only (needs `F.npy`, `Fneu.npy`, `stat.npy`, and `iscell.npy` in the same folder — no registered movie, NormCorre, or CaImAn involved at all). Click **📁 Suite2p Folder** in its sidebar's LOAD card, then **Run Analysis**.

**What it does:** sweeps the neuropil-subtraction coefficient alpha across a configurable range (default 0–2, step 0.05) in `Fcorr = rawF - alpha*Fneu`, and at each alpha computes the mean ± SEM of the off-diagonal pairwise Pearson correlation between cells (only cells with `iscell == 1`). The plot marks the alpha whose mean correlation is closest to zero with a dashed line, as a starting-point suggestion.

**Exclusion distance (default 4px):** cell pairs whose ROI masks come within this many pixels of each other are dropped from every alpha's pairwise-correlation average. Distance is the exact nearest-pixel (edge-to-edge) gap between the two ROIs' `stat.npy` pixel masks, not centroid-to-centroid. This is due to non-cells often existing at FOV borders in suite2p segmented datasets. 

**Cell filter (optional):** the **"Filter by kurtosis"** checkbox restricts the sweep to cells with Pearson kurtosis (`fisher=False`, normal ≈ 3) above a threshold (default 5), computed on the raw, un-subtracted trace before anything else runs.

The sidebar's alpha range/step and exclusion-distance fields, and the kurtosis-filter checkbox/threshold, are the only settings — see `neuropil_alpha_sweep`, `build_exclusion_mask`, and `roi_pair_min_distance` in `kurtosis_checker.py` for the implementation.

## PTC method notes (vs. the Lees et al. 2025 reference protocol)

Checked against a MATLAB reference implementation of the Lees et al. 2025 (*Nature Protocols*, Procedure 7 / Box 7-8) method. The core statistics match exactly: adjacent-frame mean `M = 0.5*(X + X')`, adjacent-frame variance `D = 0.5*(X - X')²` (averaged per-pixel over all frame pairs, so slow structural signal cancels and only shot noise survives), and the `gain_true = slope / ENF²` correction are identical formulas in both.

- **Edge cropping:** the reference protocol crops a fixed margin off every frame edge before computing anything, to exclude blanking/vignetting artifacts that resonant-scan two-photon systems can have near the frame border. The **"Edge margin (px)"** setting matches this — default 4px, matching the reference protocol's own default; set to 0 to disable.
- **Fit-range default:** matches the reference protocol's lower-50%-of-intensity-range default (`Fit range (pct of mean)` = 0–50) — raise the high end toward 98 per-run if you want the wider range instead.
- **Binning:** the reference protocol bins by unique rounded ADU value (mean-aggregated, weighted by `sqrt(pixel count)`). This tool bins into 40 percentile-spaced bins instead (median-aggregated, weighted by raw pixel count — standard weighted least squares). Both are defensible; median binning is more robust to localized artifact pixels.

## Repo contents

- `kurtosis_checker.py` — the app (single file, tkinter + matplotlib). Fully self-contained: all embedded artwork and the slab wordmark logo are base64 PNG data inside this file, so there's nothing else to copy alongside it.
- `art/` — the source PNGs the embedded art data was generated from. Not needed to run the app; keep around only if the art ever needs regenerating.
- `hockey_stick_requirements.txt` — dependency list, including the CaImAn/NormCorre setup notes above.
