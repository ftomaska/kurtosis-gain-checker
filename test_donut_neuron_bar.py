import _fake_tkinter  # noqa: F401 -- must be imported before kurtosis_checker
import kurtosis_checker as app
import random

# Regression test for Filip's most recent Neuropil Sweep ask:
# "it is still choppy the hand movement. also the donut eating neuron is
# not there. lets have a donut eating neuron animation running at the
# default. Let the crumbs accumulate. then when the loading is happening
# sweep the crumbs away. keep the aspect ratio of the drawings."
#
# Covers both layers: (1) DonutNeuronBar's own idle/busy state machine in
# isolation, and (2) KurtosisChecker's wiring of it into the Neuropil
# Sweep tab (_switch_tab / _toggle_whimsy / _progress_start / _progress_stop).


class RecordingCanvas:
    def __init__(self):
        self.calls = []
    def delete(self, tag):
        self.calls.append(("delete", tag))
    def create_image(self, *c, **kw):
        self.calls.append(("create_image", c, kw.get("tags")))
        return len(self.calls)
    def create_line(self, *c, **kw):
        return len(self.calls)
    def create_text(self, *c, **kw):
        return len(self.calls)
    def create_rectangle(self, *c, **kw):
        return len(self.calls)
    def create_oval(self, *c, **kw):
        self.calls.append(("create_oval", c, kw.get("tags")))
        return len(self.calls)
    def bbox(self, tag_or_id):
        return (0, 0, 50, 12)


def make_bar():
    b = app.DonutNeuronBar.__new__(app.DonutNeuronBar)
    b.__class__ = type("FakeDonutBar", (RecordingCanvas,), dict(app.DonutNeuronBar.__dict__))
    b.calls = []
    b._width = 400
    b._height = 340
    b._running = False
    b._job = None
    b._mode = "idle"
    b._idle_seq_idx = 0
    b._pose_t0 = 0.0
    b._crumbs = []
    b._sweep_phase = None
    b._sweep_t0 = 0.0
    b.after = lambda delay, fn: None
    return b


def run_ticks(bar, n, fake_t, dt=0.033):
    for _ in range(n):
        fake_t["v"] += dt
        bar._tick()


# ── the new sprite assets all decode ─────────────────────────────────────
for name in ["neuron_idle_standing", "neuron_donut_approach", "neuron_donut_bite",
             "neuron_donut_chew", "neuron_donut_crumbs", "broom"]:
    im = app._load_art_image(name)
    assert im.mode == "RGBA" and im.width > 10 and im.height > 10, name
print("all donut-eating-neuron sprite assets decode to real images: OK")

# ── start_idle() actually starts ticking ─────────────────────────────────
random.seed(0)
bar = make_bar()
fake_t = {"v": 0.0}
orig_monotonic = app.time.monotonic
app.time.monotonic = lambda: fake_t["v"]
try:
    assert not bar._running
    bar.start_idle()
    assert bar._running and bar._mode == "idle"
    print("start_idle() begins ticking in idle mode: OK")

    # calling it again while already idle is a no-op (doesn't reset state)
    bar._idle_seq_idx = 2
    bar.start_idle()
    assert bar._idle_seq_idx == 2, "start_idle() shouldn't reset an already-idle bar"
    print("start_idle() while already idle leaves state alone: OK")
    bar._idle_seq_idx = 0

    # ── idle cycle visits every pose in order and loops ──────────────────
    # (IDLE_SEQUENCE repeats "chew" back-to-back -- an extra chewing beat
    # before "crumbs" -- so de-dup consecutive repeats on both sides
    # before comparing.)
    seen_poses = []
    for _ in range(100):
        pose = app.DonutNeuronBar.IDLE_SEQUENCE[bar._idle_seq_idx][0]
        if not seen_poses or seen_poses[-1] != pose:
            seen_poses.append(pose)
        run_ticks(bar, 1, fake_t)
    raw_cycle = [p for p, _ in app.DonutNeuronBar.IDLE_SEQUENCE]
    expected_cycle = [p for i, p in enumerate(raw_cycle) if i == 0 or p != raw_cycle[i - 1]]
    n = len(expected_cycle)
    assert seen_poses[:n] == expected_cycle, f"expected {expected_cycle}, got {seen_poses[:n]}"
    print("idle eat-cycle visits standing -> approach -> bite -> chew -> crumbs in order: OK")

    # ── the extra chewing beat is really there, right before "crumbs" ────
    chew_indices = [i for i, (p, _) in enumerate(app.DonutNeuronBar.IDLE_SEQUENCE) if p == "chew"]
    crumbs_index = [i for i, (p, _) in enumerate(app.DonutNeuronBar.IDLE_SEQUENCE) if p == "crumbs"][0]
    assert len(chew_indices) >= 2, \
        "expected at least two 'chew' beats in IDLE_SEQUENCE (Filip: add a chewing beat before crumbs)"
    assert chew_indices[-1] == crumbs_index - 1, "the last 'chew' beat should sit immediately before 'crumbs'"
    print("an extra chewing beat sits immediately before the crumbs/hand-cleaning pose: OK")

    # ── the "chew" pose jitters a little (a low-effort chewing motion on
    # the single sprite, per Filip: "make the chew original figure jitter
    # a bit as a low effort chewing") -- the y-offset draw_donut_neuron is
    # called with should actually vary across ticks while chewing, not
    # sit frozen at one position ───────────────────────────────────────────
    bar_jitter = make_bar()
    bar_jitter.start_idle()
    # advance straight into a "chew" beat
    while app.DonutNeuronBar.IDLE_SEQUENCE[bar_jitter._idle_seq_idx][0] != "chew":
        run_ticks(bar_jitter, 1, fake_t)
    seen_cy = []
    orig_draw_donut_neuron = app.draw_donut_neuron
    def _spy_draw_donut_neuron(canvas, cx, cy, scale, pose="standing", tag="donut_neuron"):
        if pose == "chew":
            seen_cy.append(cy)
        return orig_draw_donut_neuron(canvas, cx, cy, scale, pose=pose, tag=tag)
    app.draw_donut_neuron = _spy_draw_donut_neuron
    try:
        run_ticks(bar_jitter, 12, fake_t)
    finally:
        app.draw_donut_neuron = orig_draw_donut_neuron
    assert len(seen_cy) >= 2, "expected multiple 'chew' frames to have been rendered"
    assert len(set(seen_cy)) > 1, \
        f"the chew pose's draw position should jitter across ticks, got the same cy repeatedly: {seen_cy}"
    print("the 'chew' pose jitters (varying draw y-position) instead of sitting frozen: OK")

    # ── crumbs accumulate across multiple cycles, capped at MAX_CRUMBS ───
    bar2 = make_bar()
    bar2.start_idle()
    run_ticks(bar2, 500, fake_t)  # ~16.5s -> several full cycles
    assert len(bar2._crumbs) > 0, "crumbs should have accumulated over multiple cycles"
    assert len(bar2._crumbs) <= app.DonutNeuronBar.MAX_CRUMBS
    n_after_one_pass = len(bar2._crumbs)
    run_ticks(bar2, 500, fake_t)  # more cycles -- shouldn't reset, only grow/cap
    assert len(bar2._crumbs) >= n_after_one_pass, \
        "crumbs must never reset just from looping -- only a sweep clears them"
    assert len(bar2._crumbs) <= app.DonutNeuronBar.MAX_CRUMBS
    print("crumbs accumulate across repeated eat-cycles and respect MAX_CRUMBS: OK")

    # ── set_busy(True) sweeps all crumbs away, then keeps sweeping ───────
    # (Filip: "the broom should keep sweeping until the computation there
    # is done" -- an initial pass clears the pile, then the broom loops
    # back and forth indefinitely rather than parking, for as long as
    # busy stays True)
    assert len(bar2._crumbs) > 0
    bar2.set_busy(True)
    assert bar2._mode == "busy"
    run_ticks(bar2, 60, fake_t)  # ~2s, more than SWEEP_DUR
    assert len(bar2._crumbs) == 0, "a completed sweep must clear every crumb"
    assert bar2._sweep_phase == "looping", \
        "once the pile is clear the broom should keep looping, not park"
    print("set_busy(True) sweeps the accumulated crumb pile to zero, then keeps sweeping: OK")

    # ── the broom's ping-pong loop position keeps moving indefinitely as
    # long as busy stays True -- it must never settle/freeze, since that
    # would read as "finished" while the real computation is still going ─
    run_ticks(bar2, 1, fake_t)
    bar2.calls.clear()
    bar2._render(fake_t["v"])
    first_broom = [c for c in bar2.calls if c[0] == "create_image" and c[2] == "broom"]
    run_ticks(bar2, 20, fake_t)  # ~0.66s later, still well within busy
    assert bar2._sweep_phase == "looping"
    bar2.calls.clear()
    bar2._render(fake_t["v"])
    second_broom = [c for c in bar2.calls if c[0] == "create_image" and c[2] == "broom"]
    assert first_broom and second_broom
    assert first_broom[0][1] != second_broom[0][1], \
        "the broom's draw position should keep changing while looping, not freeze in place"
    print("broom keeps moving in a continuous loop for as long as busy stays True: OK")

    # while busy, pose is always 'standing' -- no eating while real work runs
    for _ in range(10):
        run_ticks(bar2, 1, fake_t)
        pose = (app.DonutNeuronBar.IDLE_SEQUENCE[bar2._idle_seq_idx][0]
                if bar2._mode == "idle" else "standing")
        assert pose == "standing"
    print("neuron holds a plain standing pose (no eating) while busy: OK")

    # ── set_busy(False) (i.e. the computation actually finished) is what
    # stops the broom and resumes the idle cycle fresh, from an empty pile ─
    bar2.set_busy(False)
    assert bar2._mode == "idle" and bar2._idle_seq_idx == 0
    assert bar2._sweep_phase is None
    assert len(bar2._crumbs) == 0
    run_ticks(bar2, 1, fake_t)
    print("set_busy(False) (computation done) stops the broom, resumes idle eat-cycle: OK")

    # ── set_busy(True) with an already-empty pile skips straight to the
    # continuous loop (no need to visibly sweep nothing, but the broom
    # still needs to keep moving for the duration of the run) ────────────
    bar3 = make_bar()
    bar3.start_idle()
    bar3.set_busy(True)
    assert bar3._sweep_phase == "looping"
    print("set_busy(True) with no crumbs to sweep goes straight into the continuous loop: OK")

    # ── set_busy()/stop() are no-ops before start_idle() has ever run ────
    bar4 = make_bar()
    bar4.set_busy(True)
    assert bar4._mode == "idle" and not bar4._running
    print("set_busy() before start_idle() is a safe no-op: OK")

    # ── stop() halts ticking and clears the canvas ───────────────────────
    bar2.stop()
    assert not bar2._running
    calls_before = len(bar2.calls)
    run_ticks(bar2, 5, fake_t)
    assert len(bar2.calls) == calls_before, "stop() should actually halt the tick loop"
    print("stop() halts the tick loop: OK")

    # ── _render actually draws the neuron + accumulated crumbs, and the
    # broom only while busy ───────────────────────────────────────────────
    bar5 = make_bar()
    bar5.start_idle()
    run_ticks(bar5, 400, fake_t)
    assert len(bar5._crumbs) > 0
    bar5.calls.clear()
    bar5._render(fake_t["v"])
    oval_calls = [c for c in bar5.calls if c[0] == "create_oval"]
    assert len(oval_calls) == len(bar5._crumbs), "one oval per accumulated crumb"
    image_tags = [c[2] for c in bar5.calls if c[0] == "create_image"]
    assert "broom" not in image_tags, "no broom while idle"
    print("idle _render draws the neuron + one oval per crumb, no broom: OK")

    bar5.set_busy(True)
    bar5.calls.clear()
    bar5._render(fake_t["v"])
    image_tags = [c[2] for c in bar5.calls if c[0] == "create_image"]
    assert "broom" in image_tags, "broom should be drawn while busy"
    print("busy _render draws the broom: OK")
finally:
    app.time.monotonic = orig_monotonic

print("DonutNeuronBar isolated state-machine checks: ALL OK")

# ── KurtosisChecker wiring: _switch_tab / _toggle_whimsy / _progress_start
# / _progress_stop all correctly show/hide/drive the donut widget ────────
fake_root = app.tk.Tk()
checker = app.KurtosisChecker(fake_root)
checker.root.after = lambda delay, fn: fn()  # run "marshaled" calls immediately

checker._switch_tab("neuropil")
assert checker._donut_idle_active is True
assert checker.donut_bar._running is True
print("switching to Neuropil Sweep with no results shows the live donut widget: OK")

checker._progress_start()
assert checker._progress_mode == "donut"
assert checker.donut_bar._mode == "busy"
print("_progress_start flips the already-showing donut widget into busy/sweep mode: OK")

# successful run -- results now exist, donut widget should hand off to the
# real plot canvas
checker.np_results = {"fake": True}
checker._progress_stop()
assert checker._donut_idle_active is False
assert checker.donut_bar._running is False
print("_progress_stop (with results) tears down the donut widget, reveals the plot: OK")

# failed run -- np_results stays None, donut widget should just resume idle
checker.np_results = None
checker._switch_tab("gain")
checker._switch_tab("neuropil")
checker._progress_start()
assert checker.donut_bar._mode == "busy"
checker._progress_stop()
assert checker._donut_idle_active is True
assert checker.donut_bar._mode == "idle"
assert checker.donut_bar._running is True
print("_progress_stop (failed run, no results) resumes the idle donut widget: OK")

# leaving the tab stops/hides the donut widget
checker._switch_tab("gain")
assert checker.donut_bar._running is False
assert checker._donut_idle_active is False
print("switching away from the Neuropil Sweep tab stops/hides the donut widget: OK")

# whimsy off -> plain text-only splash, no donut widget at all
checker._switch_tab("neuropil")
checker.whimsy_var.set(False)
checker._show_neuropil_idle()
assert checker._donut_idle_active is False
assert checker.donut_bar._running is False
print("whimsy off falls back to the plain splash (no donut widget): OK")
checker.whimsy_var.set(True)

# _toggle_whimsy on the neuropil tab (no results) swaps the idle view
# immediately, same as the gain/kurtosis tabs already do
checker._switch_tab("gain")
checker._switch_tab("neuropil")
assert checker._donut_idle_active is True
checker._toggle_whimsy()
assert checker._donut_idle_active is False
checker._toggle_whimsy()
assert checker._donut_idle_active is True
print("_toggle_whimsy immediately swaps the neuropil tab's donut/plain idle view: OK")

print("ALL OK")
