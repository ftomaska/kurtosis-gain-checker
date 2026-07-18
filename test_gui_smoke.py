import _fake_tkinter  # noqa: F401 -- must be imported before kurtosis_checker
import kurtosis_checker as app
import inspect

# No existing test constructs the *real* KurtosisChecker / builds the real
# widget tree -- every other test binds unbound methods onto lightweight
# fake objects instead. That leaves zero coverage of the actual
# _build_controls/_build_canvas/_build_gain_sidebar/_toggle_settings/
# _switch_tab code paths, which is exactly where the Gain-tab redesign
# (sidebar, greyscale palette, moved animation) touched things. This test
# exercises that real construction + the state transitions a user would
# actually trigger by clicking around, under the _fake_tkinter shim.

fake_root = app.tk.Tk()
checker = app.KurtosisChecker(fake_root)

# All the settings vars the old row-based _build_settings_gain used to
# create must still exist (they moved into _build_gain_sidebar).
for name in ["fs_var", "enf_var", "fit_lo_var", "fit_hi_var", "spatial_bin_var",
             "margin_var", "max_frames_var", "frame_start_var", "cnmf_gsig_var",
             "exclude_roi_var", "subtract_baseline_var", "flux_baseline_pct_var",
             "pw_rigid_var", "skip_mc_var", "save_mc_var"]:
    assert hasattr(checker, name), f"missing sidebar var: {name}"
print("all gain-sidebar *_var attributes present after real __init__: OK")

# Structural attributes the new layout depends on.
for name in ["_content_row", "_gain_sidebar", "_main_area", "canvas",
             "gain_progress_bar", "_load_data_btn", "_settings_btn",
             "_action_btn", "_sort_frame", "_tab_kurtosis_btn", "_tab_gain_btn"]:
    assert hasattr(checker, name), f"missing layout attribute: {name}"
print("all new layout attributes present: OK")

# Exercise _toggle_settings (kurtosis tab, default) -- open then close.
checker._toggle_settings()
assert checker._settings_open is True
checker._toggle_settings()
assert checker._settings_open is False
print("_toggle_settings open/close on kurtosis tab: OK")

# Exercise _switch_tab both directions -- this is exactly the code path
# that was broken (AttributeError on _settings_gain/_mat_movie_btn/
# _progress_row) right after the _build_controls rewrite.
checker._switch_tab("gain")
assert checker.tab_var.get() == "gain"
checker._switch_tab("kurtosis")
assert checker.tab_var.get() == "kurtosis"
print("_switch_tab kurtosis -> gain -> kurtosis: OK")

# Toggling settings open, then switching tabs while open (another path that
# touched the removed attributes).
checker._toggle_settings()
assert checker._settings_open is True
checker._switch_tab("gain")
checker._switch_tab("kurtosis")
print("tab switch while settings panel open: OK")

# Progress start/stop swaps the canvas <-> animation in _main_area.
checker._progress_start()
checker._progress_stop()
print("_progress_start/_progress_stop swap: OK")

# Logo button exists and is wired to open SLAB_LAB_URL (placeholder for now).
assert hasattr(checker, "_load_slab_logo_photo")
photo = checker._load_slab_logo_photo(26)
assert photo is not None
print("slab logo loads as a PhotoImage: OK")

# Edge margin now defaults to 4px (matches the Lees et al. reference
# protocol's own default), changed from the prior 0px.
assert checker.margin_var.get() == 4, f"expected margin_var default 4, got {checker.margin_var.get()}"
print("margin_var defaults to 4px: OK")

# Every sidebar parameter has a "?" help icon wired to a tooltip.
assert hasattr(checker, "_help_icon") and hasattr(checker, "_make_tooltip")
print("help-icon/tooltip infrastructure present: OK")

# The real slab lab URL is wired in (no more placeholder).
assert app.SLAB_LAB_URL == "https://slslab.org", \
    f"expected the real slab lab URL, got {app.SLAB_LAB_URL}"
print("slab logo links to the real lab URL: OK")

# Baseline subtraction defaults to ON at 30th percentile (the fix for
# Filip's "these numbers are huge" feedback).
assert checker.subtract_baseline_var.get() is True
assert checker.flux_baseline_pct_var.get() == 30.0
print("baseline subtraction defaults: on, 30th percentile -- OK")

# Export Results infrastructure exists and no-ops gracefully with no prior run.
assert hasattr(checker, "_export_gain_results") and hasattr(checker, "_prompt_text")
checker.g_results = None
checker._export_gain_results()
assert "Run Analysis" in checker.status_var.get()
print("_export_gain_results present, graceful no-op with no prior run -- OK")

# PTC reference citation is linked under the Export button (Filip: "I'd
# like to add the original PTC reference to the gain estimate tab paste
# the citation as a hyperlink under the export button"). Real DOI, real
# citation text (Lees et al. 2025, the same reference the README's "PTC
# method notes" section already checks the math against), and the
# sidebar actually wires a click on it to open that DOI.
assert app.PTC_REFERENCE_URL == "https://doi.org/10.1038/s41596-024-01120-w", \
    f"expected the real Lees et al. 2025 DOI, got {app.PTC_REFERENCE_URL}"
assert "Lees" in app.PTC_REFERENCE_CITATION and "2025" in app.PTC_REFERENCE_CITATION
src_sidebar = inspect.getsource(app.KurtosisChecker._build_gain_sidebar)
assert "PTC_REFERENCE_URL" in src_sidebar and "webbrowser.open(PTC_REFERENCE_URL)" in src_sidebar, \
    "expected the Export card to bind a click on the citation to open PTC_REFERENCE_URL"
print("PTC reference citation + hyperlink wired into the Export card: OK")

# "Estimating gain" title shown directly above the Gain Estimation busy
# animation, per Filip: "id like to add a title above the gain estimation
# animation saying 'Estimating gain'." Shown only while a Gain Estimation
# run is actually in progress (packed by _progress_start, torn down by
# _progress_stop), and only for the Gain tab -- this same bar widget is
# reused for a Neuropil Sweep re-run, where this title doesn't apply.
# (_fake_tkinter's dummy Label doesn't actually track widget state, so
# this checks the real wiring via source rather than a live .cget().)
assert hasattr(checker, "_progress_title")
src_build_canvas = inspect.getsource(app.KurtosisChecker._build_canvas)
assert 'text="Estimating gain"' in src_build_canvas
src_progress_start = inspect.getsource(app.KurtosisChecker._progress_start)
assert '_progress_title.pack' in src_progress_start and 'tab_var.get() == "gain"' in src_progress_start, \
    "expected the title to only be packed when the gain tab triggered the busy bar"
src_progress_stop = inspect.getsource(app.KurtosisChecker._progress_stop)
assert '_progress_title.pack_forget' in src_progress_stop
print("'Estimating gain' title is wired above the Gain Estimation busy animation only: OK")

checker.root.after = lambda delay, fn: fn()  # run marshaled calls immediately
checker._switch_tab("gain")
checker._progress_start()
assert checker._active_bar is not None, "expected a busy bar to start for the gain tab"
checker._progress_stop()
print("gain tab _progress_start/_progress_stop run cleanly with the title wired in: OK")

print("ALL OK")
