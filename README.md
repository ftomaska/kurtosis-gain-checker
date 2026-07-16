# Kurtosis / Gain Estimation Checker

A desktop tool for two-photon calcium imaging QC, built around Suite2p output. It has three modes, toggled from the top bar:

- **Kurtosis** — sorts and visualizes ROI traces by kurtosis or SNR to spot noisy or non-cell-like extractions.
- **Gain Estimation** — runs a photon-transfer-curve ("hockey stick") analysis on a Suite2p recording to recover true PMT/camera gain (ADU/photon) and convert each cell's fluorescence to photons/cell/s.
- **Neuropil Sweep** — Suite2p only. Sweeps the neuropil-subtraction coefficient alpha (`Fcorr = rawF - alpha*Fneu`) and plots the mean ± SEM pairwise cell-cell correlation at each alpha, to help pick a good alpha for your recording. See its own section below.

This README focuses on **Gain Estimation** mode, since that's the part with real setup decisions.

**Layout:** the Gain Estimation and Neuropil Sweep tabs each have an always-visible left sidebar (Load buttons, then settings grouped into rounded-corner cards, then Run Analysis) so every option is visible before you run anything — no toggle needed. Each sidebar scrolls independently (mouse wheel, or the black scrollbar on its right edge) once its cards outgrow the window height, so **Run Analysis** at the bottom stays reachable regardless of window size. The Kurtosis tab keeps the old top-bar Load Data + collapsible Settings, since it has far fewer options. The busy animation (chameleon lobbing photons at the neuron) takes over the main output area — where the plot normally is — while a Gain Estimation or Neuropil Sweep run is in progress, and swaps back to the plot when it's done. GUI chrome (buttons, sidebar cards) uses a flat greyscale palette; plot colors (fit line, histogram, exclusion masks) are unchanged. Every sidebar parameter has a small "?" glyph next to it — click it to toggle a small popup explaining what the setting does (click again, or click the popup itself, to dismiss it). This used to open on hovering anywhere over the parameter's whole row, which Filip found too intrusive; it's now click-only and scoped to just the "?" glyph, so reading a label or typing into a nearby entry never pops it open by accident.

**Rounded corners:** every button and entry field in the app — the Load/Re-fit/Run Analysis buttons, every sidebar number field, dialog OK/Cancel/Continue/Close buttons — is drawn as a rounded pill rather than a square-cornered native widget (Tk has no built-in rounded-rectangle primitive, so each is a small Canvas that draws the shape and embeds a borderless Label/Entry on top, resized to fit on every layout change). Hover gives a flat color-shift instead of the old 3D relief bevel, which didn't read right on a rounded shape. The card backgrounds themselves also got a contrast bump (`#161616`→`#232323` fill, `#2a2a2a`→`#3d3d3d` border) — the original colors were close enough to the sidebar's own background that the rounded corners were technically there but essentially invisible in practice.

## Installation

```
pip install numpy scipy matplotlib pillow tifffile
python kurtosis_checker.py
```

That's everything you need if you already have registered TIFFs saved (see Case 1 below). CaImAn is an optional, heavier dependency only needed if Suite2p's `reg_tif` export is missing (Case 2). `h5py` is an optional extra only needed if you're loading a pre-motion-corrected `.mat` movie saved in MATLAB's v7.3/HDF5 format (Case 3).

## Gain Estimation: two setup cases

Click **Gain Estimation** in the top bar, then in the left sidebar's **LOAD** card click **📁 Suite2p Folder** and select any folder at or above your Suite2p output. The tool recursively searches *downstream* of whatever folder you select for a directory containing `ops.npy` — there's no fixed list of expected folder names, so it works whether you're parked at the session folder, a custom intermediate folder (e.g. a `small/` subfolder for a downsampled test run), the `suite2p` folder, or `plane0` itself directly.

If the tree contains **more than one** `ops.npy` — for example an older full-resolution run sitting alongside a newer downsampled one — the search specifically prefers whichever one has a *usable* `reg_tif` export over one that doesn't, rather than just grabbing the first/shallowest match. That matters: picking a plane with cell traces but no registered movie silently sends you down the slow CaImAn path for no reason.

### Case 1 — You have registered TIFFs saved (recommended, no CaImAn needed)

If you ran Suite2p with the `reg_tif` option enabled, your plane folder has a `reg_tif/` subfolder containing the motion-corrected movie as TIFF chunks (e.g. `file000_chan0.tif`, `file001_chan0.tif`, ...). This is the tool's preferred source and requires nothing beyond the core dependencies above.

1. **Suite2p Folder** → select the folder. The status bar shows the *full resolved path* plus an honest `reg_tif found (N file(s))` / `no usable reg_tif` hint, computed with the same file-matching logic Run Analysis uses — so this can never claim reg_tif is available and then fall back anyway.
2. Click **Run Analysis** (bottom of the sidebar).
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

1. **Suite2p Folder** → select the folder. The status bar shows `no reg_tif export — will need NormCorre on raw TIFFs`.
2. Click **Run Analysis**. The tool tries to resolve the raw TIFF paths from `ops.npy` automatically; if those paths don't exist on this machine, it will prompt you to locate the raw TIFF folder manually.
3. NoRMCorre runs (rigid by default; enable **piecewise-rigid** in the sidebar's MOTION CORRECTION card for non-rigid correction), then the PTC analysis proceeds as in Case 1.

**Saving the motion-corrected movie:** the **"Save motion-corrected TIFF (reg_tif)"** setting (checked by default) writes the NormCorre output as `reg_tif/file000_chan0.tif`-style chunks next to the source data, matching Suite2p's own naming convention. Next time you load the same folder, the tool finds this `reg_tif/` export and uses it directly — NormCorre only ever needs to run once per recording.

**CNMF segmentation prompt:** any time there's no `F.npy` available for the loaded dataset (so the photon-flux panel would otherwise stay empty) — whether the movie came from a fresh NormCorre run or was loaded straight from an existing `reg_tif` export that just happens to be missing its Suite2p segmentation — the tool asks whether to run CaImAn's CNMF on the registered movie to detect cells and extract traces on the spot. Accepting fills in the flux panel using CNMF-derived footprints; the results panel labels these cells "CaImAn CNMF (unverified — sanity-check!)" in place of the usual "Suite2p F.npy" label, as a reminder to check the footprints/traces before trusting the numbers. The **"CNMF cell radius (px)"** setting (default 6) controls the expected cell size (`gSig`) passed to CNMF.

Right before that prompt, the tool shows a zoomable/pannable **mean-projection viewer** of the just-loaded registered movie — a standard matplotlib zoom/pan toolbar with a live pixel-coordinate readout — specifically so you can count how many pixels a cell spans. The window then asks directly for the cell diameter (px) you measured; entering one converts it to a radius (`diameter / 2`, floored at 2) and writes it straight into the **"CNMF cell radius (px)"** setting before the CNMF prompt fires, so you don't have to go find that setting yourself. Leave it blank to keep whatever the setting already was.

CNMF's `fit()` call has been observed to return `None` on some CaImAn versions/installs while still mutating its own state in place; the tool now falls back to that mutated object instead of crashing with a bare `'NoneType' object has no attribute 'estimates'` error, and only raises a real error if CNMF genuinely produced no usable components.

**Component-quality filtering:** after CNMF finds candidate components, the tool calls CaImAn's own `evaluate_components()` and keeps only the **top 60%** by whichever quality score that produces — CaImAn's CNN-classifier probability (`cnn_preds`, a real 0–1 probability) if a CNN model is installed and usable, otherwise the `r_value` spatial-consistency score as a fallback (this is **not** a true probability, just a ranking proxy — labeled as such wherever it's shown). If quality evaluation isn't available at all (older CaImAn install, no CNN model), all components are kept rather than losing the run over an optional metrics step.

Right after CNMF finishes, a **footprint sanity-check popup** shows the mean projection with the surviving masks outlined in green and the quality-filtered-out ones in red, so you can actually look at whether the green outlines look like real cells before trusting the flux numbers — instead of just trusting a histogram. It's non-blocking (doesn't hold up the rest of the analysis) and has the same zoom/pan toolbar as the mean-projection viewer.

**Progress feedback (Gain Estimation tab only — not shown in Kurtosis mode):** in the main output area (replacing the plot for the duration of the run), a chameleon (the light source) lobs photon squiggles at a cartoon neuron for the whole duration of Run Analysis — each impact ticks a counter (drawn as classic tally/gate marks — four verticals plus a diagonal fifth stroke per group of five) and always shows the same "smashed" reaction pose the instant a photon lands; once that reaction's hold period ends, the neuron settles into a random one of its other 5 poses (sourced from Filip's own multi-pose sketch sheet) rather than a plain neutral face, purely decorative so the app never looks frozen during reg_tif reads, PTC computation, or CNMF. The neuron's single return shot now flies to and lands at the midpoint between the chameleon and the neuron. A slab-hand catch animation is built around that landing point (pop-in/hold/pop-out state machine), but the hand itself is **temporarily not rendered** — Filip is redrawing that asset — so for now the return shot just lands at the midpoint with no catch visual; the state machine keeps ticking harmlessly in the background and the hand will reappear once the new asset is wired in. Before any analysis has run, the plot area shows a small centered quote ("no power supply no gain") instead of a placeholder chart.

The bar's canvas is 340px tall (was 300px) with the neuron centered higher up (55% down vs. the old 68%) and scaled down slightly — at the old size/position, the neuron's legs and the "photons:" tally counter directly beneath it both landed past the canvas's bottom edge and were silently clipped, since a full-height character there ran to ~y=343 in a canvas that only went to y=300. Verified clear across all 10 poses (every pose renders to the same normalized height regardless of its source aspect ratio, so the fix applies uniformly).

The artwork is Filip's own hand-drawn sketches, sourced from his uploaded true-vector SVG files (the chameleon, the ten neuron expressions — 4 original plus 6 randomized alternates — the red/green photon squiggles, and the slab-hand catch asset are each real Illustrator vector art, rasterized once at build time rather than hand-traced) and **embedded directly in `kurtosis_checker.py` as base64-encoded PNG bytes** — there used to be a separate `art/` folder that had to be copied alongside the script, but that step kept getting missed in practice (copying just the `.py` file to a new machine), which raised `RuntimeError: Missing art asset '...'` and silently killed the busy animation's redraw loop. The script is now fully self-contained: a single `kurtosis_checker.py` file is everything you need, nothing else to ship or forget. The neuron's source sketch sheet had no color baked in (pure black outline), so its green body fill is added programmatically at build time (flood-filling the closed triangle interior, matching the chameleon's own green) rather than being part of the original art. Five of the alternate poses (alt1, alt2, alt4, alt5, alt6) originally had small unfilled gaps left over from the auto-fill step — most commonly the apex triangle above the eyebrow where a thin antenna-stalk line bottlenecked the flood-fill, plus a couple of eye-interior gaps that should have been white sclera — fixed with a seed-based region fill (label the dilated-and-closed interior, then paint the specific connected component containing a chosen interior pixel) rather than widening the fill heuristic globally. alt2 also had a second, separate issue: its right leg's foot loop was drawn as a disconnected shape with no leg line ever joining it to the body (a genuine gap in the original linework, not a fill bug) — patched by drawing a new connecting stroke that matches the other leg's curve and thickness. Its mouth (originally left white, same as the eye) is now dark-filled, and a light-green iris ring was added around the pupil (green wasn't in the original ink at all — both are small deliberate embellishments Filip asked for, not bug fixes). Both bars use a dark background (matching the rest of the app); ink strokes and fill colors stay exactly as drawn (black ink, green/red fill), with a **white silhouette outline traced around the outer edge** of each character/photon so they still read clearly against the dark canvas — the outline never touches or overlaps the artwork's own pixels, it only occupies what was fully transparent in the source. During NormCorre motion correction, the neuron image gets a scanline "glitch" effect (a pixel-shift, not a redraw) that fades out as progress comes in, keeping the original resolving-into-focus metaphor.

Each volley, the chameleon fires **two red photons in quick succession** (staggered ~0.15s apart, so both are genuinely airborne together for most of the trip — a rapid double-shot rather than one-at-a-time) — once **both** have landed, the neuron fires back **exactly one** green return shot, drawn with its own real green-inked squiggle art (`photon_green.png`, independently hand-drawn — not a recolor of the red one) rather than the chameleon's red one (`photon_red.png`), so the two directions are visually distinct at a glance.

The animation runs on Tk's main-thread event loop (`~30fps`) while the actual PTC/CNMF computation runs in a separate background thread, so the animation doesn't block or meaningfully slow down analysis — resizing a couple of small PNGs per frame is cheap relative to the 33ms budget, and NumPy/CaImAn's heavy lifting mostly happens at the C level, which releases Python's GIL anyway. During NormCorre specifically, a separate big centered popup takes over: a progress bar plus the neuron redrawn with a scanline "distortion" that resolves into the clean drawing as CaImAn's own log output comes in. That bar is a step-count heuristic, not a true percentage — CaImAn's `motion_correct()` is one blocking call with no frame-level progress exposed, so each distinct log line nudges the bar forward (capped short of 100% until the call actually returns).

### Case 3 (optional) — Load an already motion-corrected movie from a .mat file

If motion correction was already done somewhere else (e.g. in MATLAB), or NormCorre keeps failing with `OSError: [Errno 28] No space left on device`, skip NormCorre entirely with the **"🎬 Load .mat Movie"** button (in the sidebar's LOAD card, next to Suite2p Folder).

That disk-space error specifically comes from `motion_correction_piecewise`'s `np.memmap(..., mode='w+')` call — NormCorre writes its intermediate registered frames to a multi-GB scratch file on disk before joining them, so the error means the disk that scratch directory lives on is full, not that anything is wrong with your movie or settings. Freeing space (or pointing CaImAn's temp/cache dir at a drive with more room) fixes it the normal way; loading a pre-registered `.mat` movie instead sidesteps NormCorre's disk write altogether.

Pick a `.mat` file containing a 3D numeric matrix shaped **(x, y, time)** — MATLAB's own `size(mov)` convention. Both classic (pre-v7.3) and v7.3/HDF5 `.mat` files are supported (the latter needs `pip install h5py`); if the file has more than one 3D array in it, the largest one is picked automatically. A confirmation dialog shows the detected variable name and the frame count / pixel dimensions it inferred before committing, so a swapped-axis file is caught before it silently produces garbage.

This mode has no Suite2p `F.npy`, so — like the manual-TIFF fallback below — only the gain estimate is available by default; the per-cell photon-flux panel needs the CNMF fallback (see above) to get cell traces.

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
| Edge margin (px) | 4 | Crops this many pixels off every edge of each frame before computing anything, to exclude blanking/vignetting artifacts near the frame border (see *PTC method notes* below). Matches the Lees et al. 2025 reference protocol's own default; set to 0 to disable cropping entirely. |
| Exclude cell ROIs | off | Restrict the PTC fit to background/non-cell pixels only |
| NormCorre: piecewise-rigid | off | Use non-rigid motion correction instead of rigid (Case 2 only) |
| Skip motion correction | off | Manual-TIFF mode only — skip NormCorre if the TIFF is already registered |
| Save motion-corrected TIFF (reg_tif) | on | After NormCorre runs, save the result as `reg_tif/` chunks next to the source so future runs skip motion correction entirely |
| CNMF cell radius (px) | 6 | Expected cell radius (`gSig`) passed to CaImAn CNMF, used only when it's offered to fill an empty flux panel after a fresh NormCorre run |
| Use baseline (F0) for flux | on | Converts each cell's own low-percentile baseline value directly to photons/s, instead of the trace's plain mean (see below) — gives a stable resting-state estimate instead of one inflated by activity transients |
| Baseline pctile | 30 | Which percentile of each cell's own raw trace counts as its baseline (F0) when the above is checked |

Photon flux uses **raw F only** — no neuropil subtraction — consistent with the Wilt et al. convention that F0 includes all detected photons (signal + background *light*). F itself is never modified for the flux calculation; what changes is which per-cell scalar gets converted to photons/s. With **Use baseline (F0) for flux** unchecked, that's the plain per-frame mean of the whole trace, which activity-driven transients pull upward. Checked (the default), it's each cell's own baseline — the `Baseline pctile`-th percentile of its raw trace, 30 by default, representing its resting/least-active state — converted directly, so the flux number reflects the cell's steady baseline photon rate (background light + dark counts) rather than a mean that mixes in activity.

Both Suite2p's `F.npy` and CaImAn CNMF traces are put on a comparable per-cell-total scale before this conversion, but they get there differently: CNMF traces (via `footprints_to_raw_traces`) are already a true unweighted pixel **sum** over each ROI mask. Suite2p's `F.npy`, however, normalizes its per-pixel ROI weights (`lam`) to sum to 1 before extraction, so it's actually a weighted **mean** pixel value — its magnitude barely depends on ROI size, which is easy to mistake for a sum. The Suite2p loader corrects for this at load time by multiplying `F` (and `Fneu`) by each ROI's pixel count (`npix`, from `stat.npy`), recovering a per-cell total that matches the CNMF path's convention. Without this, Suite2p-derived flux numbers would read far smaller than CNMF-derived ones for the same underlying photon rate, and the flux magnitude would depend almost entirely on which segmentation path was used rather than on the cell itself.

**Re-fit without a full re-run:** after Run Analysis has completed once, the **↻ Re-fit** button (under Fit range/ENF in the sidebar) redoes just the linear PTC fit — and, if there's a flux panel, rescales it — using the mean/variance bins already computed. It's instant (no thread, no movie re-read) because Fit range, ENF, and the baseline settings only rework numbers that already exist (the cached bins, and the cached raw per-cell F respectively). Frame rate `fs` is also picked up live by Re-fit. Spatial bin, Edge margin, Exclude cell ROIs, and anything CNMF/motion-correction related bake into the bins themselves during the raw-pixel pass, so changing those still needs a full Run Analysis.

## Exporting results for cross-condition comparison

Once Run Analysis has completed, the **💾 Export Results…** button (EXPORT card in the sidebar, just above Run Analysis) writes the current run's numbers to disk so results from different conditions/sessions/animals can be loaded into one table and plotted against each other outside the app. Clicking it asks for a short label for this run (used in filenames and the summary row — e.g. an animal/session/condition name) and a destination folder, then writes:

- **`gain_results_summary.csv`** — one row per export, **appended** across calls, so exporting several conditions into the same folder builds up a single growing table: timestamp, label, source, cell count, `gain_true`/slope/intercept/r², flux median ± SEM, and every sidebar setting that affected the run (fs, ENF, fit range, spatial bin, edge margin, baseline settings). This is the file to load into pandas/Excel/MATLAB for plotting one metric against another across conditions.
- **`{label}_{timestamp}_ptc_bins.csv`** — that run's PTC mean/variance bins (`mu_bin`, `var_bin`, `n_bin`, and whether each bin was used in the fit), enough to redraw the hockey-stick plot exactly.
- **`{label}_{timestamp}_flux_percell.csv`** — one row per cell's photon flux, only written if that run had a flux panel at all.

The summary row records each detail file's name, so a given condition's summary row can always be matched back to its own bins/flux files. Exporting doesn't change anything on screen or require a specific settings state — it's a pure read of whatever `g_results` Run Analysis (or a subsequent Re-fit) last produced.

## Neuropil Sweep

The third tab, Suite2p-only (needs `F.npy`, `Fneu.npy`, `stat.npy`, and `iscell.npy` in the same folder — no registered movie, NormCorre, or CaImAn involved at all, so it's fast). Click **📁 Suite2p Folder** in its sidebar's LOAD card, then **Run Analysis**.

**What it does:** sweeps the neuropil-subtraction coefficient alpha across a configurable range (default 0–2, step 0.05) in `Fcorr = rawF - alpha*Fneu`, and at each alpha computes the mean ± SEM of the **off-diagonal pairwise Pearson correlation** between cells (only cells with `iscell == 1`). The idea: two cells with no genuinely shared activity should settle toward ~zero correlation once neuropil contamination is properly removed. Systematically positive correlation across many pairs at a given alpha suggests under-subtraction (shared neuropil signal still bleeding into both traces); systematically negative correlation suggests over-subtraction. The plot marks the alpha whose mean correlation is closest to zero with a dashed line, as a starting-point suggestion — not a guaranteed "correct" answer, since real inter-cell correlation from genuine shared activity isn't zero either.

**Exclusion distance (default 4px):** cell pairs whose ROI masks come within this many pixels of each other are dropped from every alpha's pairwise-correlation average. Distance is the exact nearest-pixel (edge-to-edge) gap between the two ROIs' `stat.npy` pixel masks, not centroid-to-centroid — closely adjacent or overlapping ROIs can share optical bleed-through independent of alpha, which would bias the sweep the same way at every step regardless of how well-chosen alpha is. Computing that distance for every pair would be expensive at typical Suite2p cell counts, so a cheap centroid + equivalent-radius prefilter (`sqrt(npix/pi)`) skips the exact check for pairs that are obviously too far apart to matter, and only runs the real nearest-pixel check on the remaining candidates.

**Cell filter (optional):** the **"Filter by kurtosis"** checkbox restricts the sweep to cells with Pearson kurtosis (`fisher=False`, normal ≈ 3 — same convention as the Kurtosis tab's own metric) above a threshold (default 5), computed on the raw, un-subtracted trace before anything else runs. Useful for focusing the sweep on clearly active/spiky cells rather than flat or noise-dominated ROIs, which can otherwise dilute the correlation signal with near-zero-variance traces that correlate with everything and nothing.

The sidebar's alpha range/step and exclusion-distance fields, and the kurtosis-filter checkbox/threshold, are the only settings — see `neuropil_alpha_sweep`, `build_exclusion_mask`, and `roi_pair_min_distance` in `kurtosis_checker.py` for the implementation.

## PTC method notes (vs. the Lees et al. 2025 reference protocol)

Checked against a MATLAB reference implementation of the Lees et al. 2025 (*Nature Protocols*, Procedure 7 / Box 7-8) method. The core statistics already match exactly: adjacent-frame mean `M = 0.5*(X + X')`, adjacent-frame variance `D = 0.5*(X - X')²` (averaged per-pixel over all frame pairs, so slow structural signal cancels and only shot noise survives), and the `gain_true = slope / ENF²` correction are identical formulas in both.

Two places the tools genuinely differed, one now fixed and one left as an open choice:

- **Edge cropping (fixed):** the reference script crops a fixed margin off every frame edge before computing anything, to exclude blanking/vignetting artifacts that resonant-scan two-photon systems can have near the frame border. This tool had no equivalent at all until the **"Edge margin (px)"** setting above — default 0 (no change to prior results), set it to match your acquisition system's blanking width (the reference script used 4px for its dataset). A contaminated edge with the wrong gain, left uncropped, can visibly bias the fitted slope (confirmed in `test_ptc_margin.py`); a 4px crop restored the correct value in that test.
- **Fit-range default (changed to match):** the reference script fits only the **lower 50%** of the intensity range. This tool's default now matches (`Fit range (pct of mean)` = 0–50, changed from a prior 0–98 default) — raise the high end back toward 98 per-run if you want the wider range instead (see the settings table above for the tradeoff).
- **Binning (left as-is, flagged here):** the reference script bins by unique rounded ADU value (mean-aggregated, weighted by `sqrt(pixel count)`). This tool bins into 40 percentile-spaced bins instead (median-aggregated, weighted by raw pixel count — standard weighted least squares). Both are defensible; median binning is more robust to the kind of localized artifact pixels the new edge-margin setting targets. Not changed, since it's a genuinely different (not clearly worse) design choice rather than a gap.

## Repo contents

- `kurtosis_checker.py` — the app (single file, tkinter + matplotlib). Fully self-contained: the Gain Estimation progress animation's artwork and the slab wordmark logo are embedded directly in this file as base64 PNG data, so there's nothing else to copy alongside it.
- `art/` — the source PNGs the embedded art data was generated from (Filip's own sketches, plus the slab wordmark logo). Not needed to run the app; keep around only if the art ever needs regenerating.
- `hockey_stick_requirements.txt` — dependency list, including the CaImAn/NormCorre setup notes above
