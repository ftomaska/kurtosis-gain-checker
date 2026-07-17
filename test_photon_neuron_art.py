import _fake_tkinter  # noqa: F401 -- must be imported before kurtosis_checker
import kurtosis_checker as app
import numpy as np
from PIL import Image


class RecordingCanvas:
    """Minimal stand-in for a Tk canvas that actually records what gets
    drawn, so we can assert the art functions produce real output instead
    of silently no-oping (the fake tkinter _Dummy stub would hide that).
    Art is now embedded real images (create_image), not hand-drawn vector
    primitives, so this only needs to track image/text/rect draws."""

    def __init__(self):
        self.calls = []

    def delete(self, tag):
        self.calls.append(("delete", tag))

    def create_image(self, *coords, **kw):
        self.calls.append(("create_image", coords, kw.get("tags"), kw.get("image")))
        return len(self.calls)

    def create_line(self, *coords, **kw):
        self.calls.append(("create_line", len(coords), kw.get("tags")))
        return len(self.calls)

    def create_text(self, *coords, **kw):
        self.calls.append(("create_text", coords, kw.get("text")))
        return len(self.calls)

    def create_rectangle(self, *coords, **kw):
        self.calls.append(("create_rectangle", coords, kw.get("fill")))
        return len(self.calls)

    def create_oval(self, *coords, **kw):
        self.calls.append(("create_oval", coords, kw.get("tags")))
        return len(self.calls)

    def bbox(self, tag_or_id):
        return (0, 0, 50, 12)


# ---- case 1: the real art assets actually load, with real transparency ----
for name in ("neuron_neutral", "neuron_annoyed", "neuron_surprised", "neuron_sleepy",
             "chameleon", "photon_red", "photon_green"):
    im = app._load_art_image(name)
    assert im.mode == "RGBA", f"{name}: expected RGBA (real transparency), got {im.mode}"
    alpha = im.split()[-1]
    lo, hi = alpha.getextrema()
    assert lo == 0 and hi == 255, f"{name}: expected real transparent + opaque pixels, got alpha range {lo}-{hi}"
    assert im.width > 10 and im.height > 10
    print(f"_load_art_image({name}): {im.size}, alpha range {lo}-{hi} -- OK, real embedded artwork")

# same image object should be cached, not re-read from disk every call
im_a = app._load_art_image("photon_red")
im_b = app._load_art_image("photon_red")
assert im_a is im_b, "expected _load_art_image to cache, not re-decode the PNG every call"
print("_load_art_image caches the decoded image: OK")

# ---- case 1b: outlined variant leaves the artwork's own pixels (black ink,
# colored fill) completely untouched, and adds a white ring strictly
# outside the original silhouette -- per Filip's "keep them black but add a
# white outline over the outside lines" feedback (the earlier approach
# inverted the ink to white instead; this replaces that). ----
for name in ("neuron_neutral", "chameleon", "photon_red", "photon_green"):
    raw = app._load_art_image(name)
    out = app._load_art_image_outlined(name, thickness=3)
    assert out.mode == "RGBA"
    raw_arr = np.array(raw)
    out_arr = np.array(out)
    pad = 3 + 2  # thickness + 2px slack, matching _add_white_outline
    h, w = raw_arr.shape[:2]
    assert out.size == (w + 2 * pad, h + 2 * pad), \
        f"{name}: expected the canvas padded by {pad}px on each side, got {out.size} vs raw {raw.size}"

    # the original artwork, re-pasted at the padded offset, must be
    # byte-identical to the source -- outlining must not touch ink/fill
    cropped = out_arr[pad:pad + h, pad:pad + w]
    opaque = raw_arr[..., 3] > 16
    np.testing.assert_array_equal(
        cropped[opaque], raw_arr[opaque],
        err_msg=f"{name}: outlining must leave the artwork's own opaque pixels byte-identical")
    print(f"_load_art_image_outlined({name}): original pixels untouched -- OK")

    # a white ring must exist just outside the original silhouette, and
    # must never overlap a pixel that was part of the original artwork
    orig_mask_padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    orig_mask_padded[pad:pad + h, pad:pad + w] = opaque
    white_ring = (out_arr[..., 0] == 255) & (out_arr[..., 1] == 255) & \
                 (out_arr[..., 2] == 255) & (out_arr[..., 3] == 255) & ~orig_mask_padded
    assert white_ring.sum() > 0, f"{name}: expected a nonzero white outline ring"
    assert not (white_ring & orig_mask_padded).any(), \
        f"{name}: outline ring must never overlap the original artwork's own pixels"
    print(f"_load_art_image_outlined({name}): white ring present, never overlaps original art -- OK")

    # cached, not recomputed every call
    assert app._load_art_image_outlined(name, thickness=3) is out
print("_load_art_image_outlined: ink/fill untouched, white ring traced around silhouette, cached -- OK")


# ---- case 2: _apply_row_distort actually shifts pixels, and is a no-op at distort=0 ----
synthetic = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
px = synthetic.load()
for y in range(10):
    for x in range(20):
        px[x, y] = (x * 10 % 256, y * 20 % 256, 128, 255)

out0 = app._apply_row_distort(synthetic, distort=0.0, phase=0.0)
assert out0.tobytes() == synthetic.tobytes(), "distort=0 should leave pixels unchanged"

out1 = app._apply_row_distort(synthetic, distort=1.0, phase=0.3)
assert out1.tobytes() != synthetic.tobytes(), "distort=1.0 should visibly shift pixels"
assert out1.size == synthetic.size, "distortion should not change image dimensions"
print("_apply_row_distort: no-op at distort=0, visibly shifts pixels at distort=1.0, OK")


# ---- case 3: draw_neuron/draw_chameleon/draw_photon draw a real embedded
# image (not vector primitives) and report a sensible display size ----
for expr in ("neutral", "surprised", "annoyed", "sleepy"):
    c = RecordingCanvas()
    w, h = app.draw_neuron(c, cx=100, cy=100, scale=60, expr=expr, distort=0.0, phase=0.0, tag="nrn")
    kinds = [call[0] for call in c.calls]
    assert kinds == ["delete", "create_image"], f"{expr}: expected a single delete+create_image pair, got {kinds}"
    assert c.calls[-1][2] == "nrn", f"{expr}: create_image should carry the requested tag"
    assert c.calls[-1][3] is not None, f"{expr}: create_image should reference a real PhotoImage"
    assert h > 50 and w > 20, f"{expr}: expected a plausible display size, got {(w, h)}"
    print(f"draw_neuron({expr}): embeds real image at size {(w, h)}, tag OK")

c = RecordingCanvas()
cw, ch = app.draw_chameleon(c, cx=50, cy=50, scale=40, tag="cham")
kinds = [call[0] for call in c.calls]
assert kinds == ["delete", "create_image"]
assert ch > 20 and cw > 20
print(f"draw_chameleon: embeds real image at size {(cw, ch)}, OK")

c = RecordingCanvas()
app.draw_photon(c, cx=200, cy=80, length=30, amplitude=8, angle_deg=15, phase=1.2, tag="photon")
kinds = [call[0] for call in c.calls]
assert kinds == ["delete", "create_image"]
print(f"draw_photon: embeds real image, OK")

# rotating the photon shouldn't error across a full range of angles/phases
c = RecordingCanvas()
for angle in (0, 45, 90, 137, 270, -30):
    app.draw_photon(c, cx=0, cy=0, length=30, amplitude=8, angle_deg=angle, phase=2.0, tag="photon")
print("draw_photon: rotation across a range of angles doesn't error, OK")


# ---- case 4: PhotonNeuronBar renders a frame without crashing and its
# internal counter/expression state machine behaves sanely across ticks ----
bar = app.PhotonNeuronBar.__new__(app.PhotonNeuronBar)
bar.__class__ = type("FakeBar", (RecordingCanvas,), dict(app.PhotonNeuronBar.__dict__))
bar.calls = []
bar._width = 400
bar._running = True
bar._hits = 0
bar._expr = "neutral"
bar._expr_until = 0.0
bar._flight_frac = 0.3   # force a photon mid-flight so draw_photon gets exercised too
bar._flight_t0 = 0.0
bar._flight_dir = "out"
bar._flight2_frac = None  # only shot A in flight for this check
bar._hand_frac = None
bar._hand_phase = "hidden"
bar._next_launch = 999.0
bar._t0 = 0.0
bar._job = None
bar._render(0.2)
kinds = [call[0] for call in bar.calls]
assert "create_text" in kinds, "expected the photon counter label to be drawn"
assert kinds.count("create_image") == 3, \
    f"expected 3 embedded images mid-flight (neuron + chameleon + photon), got {kinds.count('create_image')}"
print(f"PhotonNeuronBar._render mid-flight: {len(bar.calls)} draw calls, {kinds.count('create_image')} images")

bar._flight_frac = None
bar.calls = []
bar._render(0.2)
kinds = [call[0] for call in bar.calls]
assert kinds.count("create_image") == 2, \
    f"expected 2 embedded images without an in-flight photon (neuron + chameleon), got {kinds.count('create_image')}"
print(f"PhotonNeuronBar._render idle: {kinds.count('create_image')} images (no photon), OK")

# both shots of the burst in flight together -- 4 images (neuron+chameleon+
# shot A + shot B), each drawn under its own tag so they don't clobber
# each other via draw_photon's internal delete(tag)
bar._flight_frac = 0.3
bar._flight_dir = "out"
bar._flight2_frac = 0.15
bar.calls = []
bar._render(0.2)
img_tags = [call[2] for call in bar.calls if call[0] == "create_image"]
assert img_tags.count("photon") == 1
assert img_tags.count("photon2") == 1
assert len(img_tags) == 4, \
    f"expected 4 embedded images with both burst shots in flight (neuron+chameleon+2 photons), got {len(img_tags)}"
print("PhotonNeuronBar._render with both burst shots in flight: 4 images, distinct tags -- OK")
bar._flight2_frac = None

# neuron should be positioned left of chameleon (per Filip's "neuron on the left" feedback)
w = max(bar._width, 60)
nrn_cx_expected = w * 0.16
cham_cx_expected = w * 0.86
assert nrn_cx_expected < cham_cx_expected, "sanity: neuron cx ratio should be left of chameleon cx ratio"

# ---- case 4b: the chameleon fires two red photons right after each other
# (staggered by VOLLEY_STAGGER, genuinely overlapping in flight), and once
# BOTH have landed the neuron fires exactly one green return shot back --
# per Filip's "chameleon spits two photons right after each other and the
# neuron spits one back" feedback (replaces the old "return every 2nd hit,
# one shot per cycle" cadence). ----
def make_fake_bar():
    b = app.PhotonNeuronBar.__new__(app.PhotonNeuronBar)
    b.__class__ = type("FakeBar2", (RecordingCanvas,), dict(app.PhotonNeuronBar.__dict__))
    b.calls = []
    b._width = 400
    b._running = True
    b._hits = 0
    b._expr = "neutral"
    b._expr_until = 0.0
    b._flight_frac = None
    b._flight_t0 = 0.0
    b._flight_dir = "out"
    b._flight2_frac = None
    b._flight2_t0 = 0.0
    b._next_second_launch = None
    b._out_landed = None
    b._return_pending = False
    b._hand_phase = "hidden"
    b._hand_frac = None
    b._hand_t0 = 0.0
    b._next_launch = 0.0
    b._t0 = 0.0
    b._job = lambda *a, **k: None
    b.after = lambda delay, fn: None  # avoid touching the fake-tkinter after() plumbing
    return b


fake_clock = {"t": 0.0}
orig_monotonic = app.time.monotonic
app.time.monotonic = lambda: fake_clock["t"]

bar2 = make_fake_bar()
try:
    # --- shot A launches at t=0 (next_launch=0) ---
    fake_clock["t"] = 0.0
    bar2._tick()
    assert bar2._flight_dir == "out" and bar2._flight_frac == 0.0
    assert bar2._flight2_frac is None, "shot B shouldn't launch in the same instant as shot A"
    assert bar2._hits == 0

    # --- shot B launches ~VOLLEY_STAGGER later, while shot A is still airborne ---
    fake_clock["t"] = app.PhotonNeuronBar.VOLLEY_STAGGER
    bar2._tick()
    assert bar2._flight2_frac == 0.0, "shot B should launch once VOLLEY_STAGGER has elapsed"
    assert 0 < bar2._flight_frac < 1.0, "shot A should still be mid-flight when shot B launches"
    print("shot B launches VOLLEY_STAGGER after shot A, while shot A is still airborne -- OK")

    # --- shot A lands first; shot B is still in flight (genuine overlap) ---
    fake_clock["t"] = 0.56
    bar2._tick()
    assert bar2._hits == 1, f"expected 1 hit after shot A lands, got {bar2._hits}"
    assert bar2._flight_frac is None, "shot A's slot should free up once it lands"
    assert bar2._flight2_frac is not None and 0 < bar2._flight2_frac < 1.0, \
        "shot B should still be mid-flight when shot A lands -- they must genuinely overlap"
    assert bar2._flight_dir != "back", "return shouldn't fire until BOTH outbound shots have landed"
    print("shot A lands while shot B is still airborne (genuine overlap): hits=1, no return yet -- OK")

    # --- shot B lands -> both outbound shots landed -> neuron goes into
    # (or refreshes) its smashed/hit pose and HOLDS there for 0.5s -- the
    # return shot does NOT fire yet. Per Filip's "I want the neuron to
    # spit out the green photon as it is reverting from the 'hit'
    # stance" -- it used to fire the instant it got hit, while still
    # mid-flinch; now it waits for the hit-pose hold to actually finish.
    shot_b_land_t = app.PhotonNeuronBar.VOLLEY_STAGGER + 0.55
    fake_clock["t"] = shot_b_land_t
    bar2._tick()
    assert bar2._hits == 2, f"expected 2 hits after shot B lands, got {bar2._hits}"
    assert bar2._flight2_frac is None
    assert bar2._flight_dir == "out" and bar2._flight_frac is None and not bar2._return_pending, \
        "the return shot should NOT fire the instant both outbound shots land -- it should " \
        "wait for the hit-pose hold to finish"
    assert bar2._expr == app.NEURON_SMASHED_EXPR, "neuron should be in its smashed/hit pose right after landing"
    expr_until = bar2._expr_until
    print("shot B lands -> both outbound shots landed -> neuron holds hit pose, no return shot yet -- OK")

    # --- still mid-hold (before expr_until) -- return still hasn't fired ---
    fake_clock["t"] = shot_b_land_t + 0.2
    bar2._tick()
    assert bar2._flight_dir == "out" and not bar2._return_pending, \
        "return shot should still be withheld while the hit-pose hold is in progress"
    assert bar2._expr == app.NEURON_SMASHED_EXPR, "neuron should still be showing the hit pose mid-hold"
    print("mid-hold (before the hit pose reverts): return shot still withheld -- OK")

    # --- hold expires -> neuron reverts to a random idle pose AND fires
    # the return shot in the same tick, so the green photon visibly
    # leaves as it's coming out of the hit stance ---
    fake_clock["t"] = expr_until
    bar2._tick()
    assert bar2._flight_dir == "back" and bar2._flight_frac == 0.0 and bar2._return_pending, \
        "the return shot should fire the instant the hit-pose hold expires"
    assert bar2._expr != app.NEURON_SMASHED_EXPR and bar2._expr in app.NEURON_IDLE_REACTION_EXPRS, \
        f"neuron should have reverted to a random idle pose, got {bar2._expr!r}"
    print("hit-pose hold expires -> neuron reverts to idle pose AND fires the return shot together -- OK")

    # --- return shot lands -> volley fully done, back to idle ---
    # (+0.001 slack: floating-point summation of the 0.55s flight can
    # land a hair under 1.0, e.g. 0.9999999999998, which wouldn't
    # register as "landed" yet)
    fake_clock["t"] = expr_until + 0.551
    bar2._tick()
    assert bar2._flight_frac is None and bar2._flight2_frac is None, \
        "expected the bar to go idle only after the return shot lands"
    assert bar2._hits == 2, "the return shot itself shouldn't increment the hit counter"
    print("return shot lands -> volley done, back to idle, hit count unchanged by the return -- OK")
finally:
    app.time.monotonic = orig_monotonic

# ---- case 4d: the slab-hand's redesigned catch animation -- Filip: "I
# would like to have this hand come from below open and than close with
# the photon in it and drag it off screen down. It can come and leave in
# a hole." So: hidden through the whole outbound volley: starts "rise"ing
# (open) the instant the return shot LAUNCHES (not when it lands -- it
# needs time to climb up first), holds risen/open if it finishes early,
# force-jumps to "close" the instant the return shot actually lands,
# holds closed, then "drag"s back down into the hole.
bar4 = make_fake_bar()
app.time.monotonic = lambda: fake_clock["t"]
try:
    # --- through the whole outbound volley, the hand stays hidden ---
    fake_clock["t"] = 0.0
    bar4._tick()
    assert bar4._hand_phase == "hidden" and bar4._hand_frac is None
    fake_clock["t"] = app.PhotonNeuronBar.VOLLEY_STAGGER
    bar4._tick()
    assert bar4._hand_phase == "hidden" and bar4._hand_frac is None, \
        "hand shouldn't appear during the outbound leg, only once the return shot launches"
    fake_clock["t"] = 0.56
    bar4._tick()  # shot A lands
    assert bar4._hand_phase == "hidden" and bar4._hand_frac is None
    print("hand stays hidden throughout the outbound volley -- OK")

    # --- shot B lands -> neuron holds its hit pose -> return shot hasn't
    # launched yet (it now waits for the hit-pose hold to finish -- see
    # case 4b above) -- hand still hidden either way ---
    shot_b_land_t = app.PhotonNeuronBar.VOLLEY_STAGGER + 0.55
    fake_clock["t"] = shot_b_land_t
    bar4._tick()
    assert bar4._flight_dir == "out" and not bar4._return_pending
    assert bar4._hand_phase == "hidden" and bar4._hand_frac is None, \
        "hand shouldn't rise before the return shot has even launched"
    expr_until = bar4._expr_until
    print("hand stays hidden while the neuron is mid-hold, before the return shot even launches -- OK")

    # --- hit-pose hold expires -> return shot launches -> hand starts
    # rising (open) right away, well before the photon actually arrives ---
    fake_clock["t"] = expr_until
    bar4._tick()
    assert bar4._flight_dir == "back" and bar4._flight_frac == 0.0
    assert bar4._hand_phase == "rise", \
        "hand should start rising the instant the return shot launches, not wait for it to land"
    assert bar4._hand_frac == 0.0
    rise_t0 = fake_clock["t"]
    print("hand starts rising (open) the instant the return shot launches -- OK")

    # --- mid-rise: frac ramps 0 -> 1 over HAND_RISE_DUR ---
    fake_clock["t"] = rise_t0 + app.PhotonNeuronBar.HAND_RISE_DUR * 0.5
    bar4._tick()
    assert bar4._hand_phase == "rise"
    assert abs(bar4._hand_frac - 0.5) < 1e-6, \
        f"expected hand rise ~50% partway through HAND_RISE_DUR, got {bar4._hand_frac}"
    print("hand rise fraction ramps up linearly -- OK")

    # --- rise finishes (HAND_RISE_DUR < the 0.55s flight, so this happens
    # before the photon lands) -> holds at frac=1.0, still "rise" phase,
    # open, waiting -- no self-driven transition out of "rise" ---
    fake_clock["t"] = rise_t0 + app.PhotonNeuronBar.HAND_RISE_DUR + 0.02
    bar4._tick()
    assert bar4._hand_phase == "rise" and bar4._hand_frac == 1.0, \
        "hand should hold fully risen (open), waiting, once HAND_RISE_DUR elapses but before the photon lands"
    print("hand holds fully risen and open, waiting for the photon to land -- OK")

    # --- return shot lands at the midpoint -> hand force-jumps to "close" ---
    fake_clock["t"] = expr_until + 0.551
    bar4._tick()
    assert bar4._flight_frac is None and bar4._flight2_frac is None, "volley should be fully done"
    assert bar4._hand_phase == "close", "hand should snap to 'close' the instant the return shot lands"
    assert bar4._hand_frac == 0.0, f"hand should start its close from frac 0, got {bar4._hand_frac}"
    catch_t = fake_clock["t"]
    print("hand force-jumps to 'close' the instant the return shot lands at the midpoint -- OK")

    # --- regression: a new volley must NOT be able to launch while the
    # hand is still animating (close/hold/drag), even if _next_launch is
    # already satisfied. This was a real bug: the instant the return shot
    # lands, _flight_frac/_flight2_frac/_return_pending/_out_landed all
    # clear to their "idle" values immediately (as just confirmed above),
    # but _next_launch is only ever *rescheduled* once the hand finishes
    # dragging away and goes "hidden" -- so right here, _next_launch is
    # still whatever value gated the volley that just landed, already in
    # the past. Without an explicit hand-phase guard, every launch
    # condition would already be true on this exact tick, and a new
    # volley could fire while the hand is still visibly closing/holding/
    # dragging -- exactly Filip's "the hand is still up when the
    # chameleon volleys more photons." ---
    assert bar4._flight_frac is None and bar4._flight2_frac is None
    assert bar4._out_landed is None and not bar4._return_pending
    assert bar4._next_launch <= catch_t, \
        "test setup: _next_launch should already be satisfied at this point (stale from the prior volley)"
    fake_clock["t"] = catch_t + 0.001
    bar4._tick()
    assert bar4._flight_frac is None and bar4._out_landed is None, \
        "a new volley must not launch while the hand is still 'close' -- every other guard is already clear"
    assert bar4._hand_phase in ("close", "hold"), "hand should still be mid-animation, not hidden"
    print("a new volley cannot launch while the hand is still close/hold/dragging, even with a stale _next_launch -- OK")

    # --- closing: frac ramps 0 -> 1 over HAND_CLOSE_DUR ---
    fake_clock["t"] = catch_t + app.PhotonNeuronBar.HAND_CLOSE_DUR * 0.5
    bar4._tick()
    assert bar4._hand_phase == "close"
    assert abs(bar4._hand_frac - 0.5) < 1e-6, \
        f"expected hand close ~50% halfway through HAND_CLOSE_DUR, got {bar4._hand_frac}"
    print("hand close fraction ramps up linearly -- OK")

    # --- close completes -> holds closed for HAND_HOLD_DUR ---
    fake_clock["t"] = catch_t + app.PhotonNeuronBar.HAND_CLOSE_DUR + 0.01
    bar4._tick()
    assert bar4._hand_phase == "hold" and bar4._hand_frac == 1.0
    hold_t0 = fake_clock["t"]
    fake_clock["t"] = hold_t0 + app.PhotonNeuronBar.HAND_HOLD_DUR * 0.5
    bar4._tick()
    assert bar4._hand_phase == "hold" and bar4._hand_frac == 1.0, \
        "hand should stay closed throughout the hold phase"
    print("hand holds closed during the hold phase -- OK")

    # --- hold ends -> drags back down into the hole, frac 0 -> 1 over HAND_DRAG_DUR ---
    fake_clock["t"] = hold_t0 + app.PhotonNeuronBar.HAND_HOLD_DUR + 0.01
    bar4._tick()
    assert bar4._hand_phase == "drag"
    drag_t0 = fake_clock["t"]
    fake_clock["t"] = drag_t0 + app.PhotonNeuronBar.HAND_DRAG_DUR * 0.5
    bar4._tick()
    assert bar4._hand_phase == "drag"
    assert abs(bar4._hand_frac - 0.5) < 1e-6, \
        f"expected hand drag ~50% halfway through HAND_DRAG_DUR, got {bar4._hand_frac}"
    print("hand drag fraction ramps up linearly -- OK")

    # --- drag completes -> hand goes fully hidden again, back in the hole ---
    fake_clock["t"] = drag_t0 + app.PhotonNeuronBar.HAND_DRAG_DUR + 0.001
    bar4._tick()
    assert bar4._hand_phase == "hidden" and bar4._hand_frac is None, \
        "hand should be fully hidden again once it's dragged all the way back into the hole"
    print("hand goes hidden again once the drag-into-the-hole finishes -- OK")

    # --- and once the hand really is hidden AND NEXT_VOLLEY_PAUSE_S has
    # actually elapsed, a new volley legitimately can launch ---
    fake_clock["t"] = bar4._next_launch + 0.001
    bar4._tick()
    assert bar4._flight_frac is not None and bar4._out_landed == 0, \
        "a new volley should launch once the hand is hidden and the pause has elapsed"
    print("a new volley launches once the hand is hidden and the pause has elapsed -- OK")
finally:
    app.time.monotonic = orig_monotonic

# Hand rendering is re-enabled (H1/H2 assets landed) -- _render should draw
# a "hand"-tagged image whenever _hand_phase != "hidden", picking the open
# (H1) pose during "rise" and early "close", the closed (H2) pose from the
# midpoint of "close" onward through "hold"/"drag", and draw nothing (plus
# delete any stale one) once phase is "hidden". A caught "hand_photon"
# image should appear alongside it (tucked into the fist) from the moment
# it closes onward -- Filip: "can you also put the photon in the catching
# hand when it grabs it" -- but not while it's still open/rising.
seen_poses = []
orig_draw_slab_hand = app.draw_slab_hand
def _spy_draw_slab_hand(canvas, cx, cy, scale, tag="slab_hand", pose="open"):
    seen_poses.append(pose)
    return orig_draw_slab_hand(canvas, cx, cy, scale, tag=tag, pose=pose)
app.draw_slab_hand = _spy_draw_slab_hand
try:
    bar4b = make_fake_bar()
    cases = [
        ("rise", 0.3, "open", False), ("rise", 1.0, "open", False),
        ("close", 0.2, "open", False), ("close", 0.8, "closed", True),
        ("hold", 1.0, "closed", True), ("drag", 0.5, "closed", True),
    ]
    for phase, frac, expected_pose, expect_photon in cases:
        bar4b._hand_phase = phase
        bar4b._hand_frac = frac
        bar4b.calls = []
        seen_poses.clear()
        bar4b._render(0.2)
        hand_img_calls = [c for c in bar4b.calls if c[0] == "create_image" and c[2] == "hand"]
        assert len(hand_img_calls) == 1, \
            f"expected exactly one 'hand' image with phase={phase!r} frac={frac!r}, got {len(hand_img_calls)}"
        assert seen_poses == [expected_pose], \
            f"phase={phase!r} frac={frac!r}: expected pose {expected_pose!r}, got {seen_poses}"
        photon_img_calls = [c for c in bar4b.calls if c[0] == "create_image" and c[2] == "hand_photon"]
        if expect_photon:
            assert len(photon_img_calls) == 1, \
                f"phase={phase!r} frac={frac!r}: expected the caught photon drawn in the closed fist"
        else:
            assert len(photon_img_calls) == 0, \
                f"phase={phase!r} frac={frac!r}: hand isn't closed yet, shouldn't show a caught photon"
    print("PhotonNeuronBar._render draws the hand with the right open/closed pose at every phase -- OK")
    print("caught photon appears in the fist once closed (close>=0.5/hold/drag), not before -- OK")

    bar4b._hand_phase = "hidden"
    bar4b._hand_frac = None
    bar4b.calls = []
    bar4b._render(0.2)
    hand_img_calls = [c for c in bar4b.calls if c[0] == "create_image" and c[2] == "hand"]
    assert len(hand_img_calls) == 0, "hidden phase should draw no hand image"
    assert ("delete", "hand") in bar4b.calls and ("delete", "hand_hole") in bar4b.calls \
        and ("delete", "hand_photon") in bar4b.calls, \
        "hidden phase should delete the hand, the hole marker, and any caught photon"
    print("PhotonNeuronBar._render draws nothing and cleans up when hand_phase is 'hidden' -- OK")
finally:
    app.draw_slab_hand = orig_draw_slab_hand

# The rise/drag motion eases in/out (smoothstep) rather than moving at a
# flat linear speed -- per Filip's "the hand movement is very not smooth".
# At frac=0.5 the eased fraction should be exactly 0.5 (smoothstep is
# symmetric), but at frac=0.25 it should be well under the linear value
# (slower start), and at frac=0.75 well over it (still accelerating out
# of the middle) -- confirming the curve is actually being applied, not
# just linear interpolation relabeled.
assert app._smoothstep(0.0) == 0.0
assert app._smoothstep(1.0) == 1.0
assert abs(app._smoothstep(0.5) - 0.5) < 1e-9
assert app._smoothstep(0.25) < 0.25, "smoothstep should ease in (slower than linear near the start)"
assert app._smoothstep(0.75) > 0.75, "smoothstep should ease out (faster than linear approaching the end)"
print("_smoothstep is a real ease-in-out curve, not a linear pass-through -- OK")

# ---- case 4e: every hit shows the specific "smashed" pose (alt5), not a
# random one of the 6 -- per Filip's "use the one that is getting smashed
# for whenever a photon hits" -- and once the hold period expires it
# regresses to a random pose from the *other* 5, not back to plain
# neutral. Drives through a full two-shot volley (mirroring case 4b's
# timeline exactly) so shot A/B launches and landings don't collide with
# the expression checks in unexpected ways. ----
bar5 = make_fake_bar()
app.time.monotonic = lambda: fake_clock["t"]
try:
    bar5._expr = "neutral"
    fake_clock["t"] = 0.0
    bar5._tick()  # shot A launches
    assert bar5._expr == "neutral"

    fake_clock["t"] = app.PhotonNeuronBar.VOLLEY_STAGGER
    bar5._tick()  # shot B launches, shot A still airborne
    assert bar5._expr == "neutral"

    fake_clock["t"] = 0.56
    bar5._tick()  # shot A lands -- first hit
    assert bar5._expr == app.NEURON_SMASHED_EXPR, \
        f"expected the smashed pose ({app.NEURON_SMASHED_EXPR}) on hit, got {bar5._expr}"
    print("every hit shows the specific smashed pose, not a random one -- OK")
    expr_until_after_first_hit = bar5._expr_until

    shot_b_land_t = app.PhotonNeuronBar.VOLLEY_STAGGER + 0.55
    fake_clock["t"] = shot_b_land_t
    bar5._tick()  # shot B lands -> 2nd hit -> hold refreshes, return NOT fired yet
    assert bar5._expr == app.NEURON_SMASHED_EXPR, "2nd hit should re-show the smashed pose too"
    assert bar5._expr_until > expr_until_after_first_hit, \
        "the 2nd hit should refresh the hold timer"
    assert bar5._flight_dir == "out" and not bar5._return_pending, \
        "return shot shouldn't fire until the (refreshed) hold expires"

    # hold period from the 2nd (most recent) hit expires -- neuron
    # regresses to a random pose from the OTHER 5 alts (never back to
    # "neutral", never staying on the smashed pose) AND fires the return
    # shot in that same tick, so the photon visibly leaves as the neuron
    # is coming out of the hit stance.
    fake_clock["t"] = bar5._expr_until
    bar5._tick()
    assert bar5._expr in app.NEURON_IDLE_REACTION_EXPRS, \
        f"expected a random idle pose from {app.NEURON_IDLE_REACTION_EXPRS}, got {bar5._expr}"
    assert bar5._expr != app.NEURON_SMASHED_EXPR
    assert bar5._flight_dir == "back", "return shot should launch the instant the hold expires"
    print(f"after the hold expires, regresses to a random idle pose ({bar5._expr}) "
          f"and fires the return shot together -- OK")

    # it shouldn't keep re-randomizing every tick once idle (would cause
    # flicker) -- same idle expr should persist on the next tick, as long
    # as we stay before the return shot's own landing (which would count
    # as a 3rd hit and re-trigger the smashed pose -- a separate, correct
    # behavior, not what's being checked here)
    idle_expr = bar5._expr
    return_launch_t = fake_clock["t"]
    assert return_launch_t + 0.01 < return_launch_t + 0.55, \
        "test timing sanity: must stay before the return shot lands"
    fake_clock["t"] += 0.01
    bar5._tick()
    assert bar5._expr == idle_expr, \
        "idle expression shouldn't keep re-randomizing every tick (flicker)"
    print("idle pose stays stable across ticks (no flicker) until the next hit -- OK")
finally:
    app.time.monotonic = orig_monotonic

# ---- case 4c: outbound photons (shot A and shot B) are red, return photon
# is green-tinted ----
bar3 = make_fake_bar()
bar3._flight_frac = 0.3
bar3._flight_dir = "out"
bar3._flight2_frac = 0.1
bar3.calls = []
bar3._render(0.2)
img_calls = [c for c in bar3.calls if c[0] == "create_image" and c[2] == "photon"]
assert len(img_calls) == 1
red_photo = img_calls[0][3]
img_calls_b = [c for c in bar3.calls if c[0] == "create_image" and c[2] == "photon2"]
assert len(img_calls_b) == 1
red_photo_b = img_calls_b[0][3]

bar3._flight_dir = "back"
bar3._flight2_frac = None
bar3.calls = []
bar3._render(0.2)
img_calls = [c for c in bar3.calls if c[0] == "create_image" and c[2] == "photon"]
assert len(img_calls) == 1
green_photo = img_calls[0][3]
assert red_photo is not green_photo, "outbound (red) and return (green) photons should be visually distinct images"
print("outbound flights use the red photon, return flight uses a distinct (green) photon -- OK")

# red/green are now independent real artwork (photon_red.png / photon_green.png,
# both supplied by Filip) rather than a programmatic channel swap -- check each
# resolves to its own distinctly-named, outlined asset.
red_im = app._load_photon_variant("red")
green_im = app._load_photon_variant("green")
assert red_im is app._load_art_image_outlined("photon_red")
assert green_im is app._load_art_image_outlined("photon_green")
red_arr = np.array(red_im)
green_arr = np.array(green_im)
# both should have real color (not just black/white/transparent) somewhere,
# confirming each is genuinely tinted art rather than one being a colorless
# fallback for the other
assert (np.abs(red_arr[..., 0].astype(int) - red_arr[..., 1].astype(int)) > 60).any(), \
    "photon_red should have real red-dominant pixels"
assert (np.abs(green_arr[..., 1].astype(int) - green_arr[..., 0].astype(int)) > 60).any(), \
    "photon_green should have real green-dominant pixels"
print("_load_photon_variant: 'red'/'green' resolve to real, independent artwork -- OK")

w = max(bar2._width, 60)
nrn_cx_expected = w * 0.16
cham_cx_expected = w * 0.86
print(f"layout: neuron cx={nrn_cx_expected:.0f} (left), chameleon cx={cham_cx_expected:.0f} (right)")


# ---- case 5: MotionCorrectionOverlay progress climbs toward 0.95 and caps
# there until close() snaps it to 1.0 ----
class FakeRoot:
    def after(self, delay, fn=None):
        if fn is not None:
            fn()
    def update_idletasks(self): pass
    def winfo_rootx(self): return 0
    def winfo_rooty(self): return 0
    def winfo_width(self): return 800
    def winfo_height(self): return 900

ov = app.MotionCorrectionOverlay(FakeRoot())
for i in range(50):
    ov.tick(f"chunk {i}")
assert ov._progress < 0.951, f"progress should stay capped below/at 0.95 pre-close, got {ov._progress}"
assert ov._progress > 0.8, f"progress should have climbed substantially after 50 ticks, got {ov._progress}"
print(f"overlay progress after 50 ticks (capped): {ov._progress:.4f}")
ov.close()
assert ov._progress == 1.0
assert ov._closing is True
print("overlay.close() snaps progress to 1.0 and marks closing")

# tick() after close() should be a no-op (closing guard)
ov.tick("should be ignored")
assert ov._progress == 1.0
print("tick() after close() is correctly ignored")

# ---- case 6: on_normcorre_start/on_normcorre_done fire exactly once each,
# and on_normcorre_done fires even when run_normcorre raises ----
events = []


def fake_run_normcorre_ok(tiff_paths, pw_rigid, progress_cb=None, scratch_dir=None):
    events.append("normcorre_ran")
    return np.zeros((3, 4, 4), dtype="float32")


def fake_run_normcorre_fail(tiff_paths, pw_rigid, progress_cb=None, scratch_dir=None):
    events.append("normcorre_ran")
    raise RuntimeError("boom")


orig_run_normcorre = app.run_normcorre

app.run_normcorre = fake_run_normcorre_ok
events.clear()
arr, note = app._normcorre_then_maybe_save(
    ["/tmp/fake.tif"], max_frames=100, pw_rigid=False, save_mc=False, save_dir="/tmp",
    on_normcorre_start=lambda: events.append("start"),
    on_normcorre_done=lambda: events.append("done"))
assert events == ["start", "normcorre_ran", "done"], events
print("success path: on_normcorre_start/done fired in order:", events)

app.run_normcorre = fake_run_normcorre_fail
events.clear()
try:
    app._normcorre_then_maybe_save(
        ["/tmp/fake.tif"], max_frames=100, pw_rigid=False, save_mc=False, save_dir="/tmp",
        on_normcorre_start=lambda: events.append("start"),
        on_normcorre_done=lambda: events.append("done"))
    raise AssertionError("expected RuntimeError to propagate")
except RuntimeError:
    pass
assert events == ["start", "normcorre_ran", "done"], \
    f"on_normcorre_done must fire even when run_normcorre raises (overlay must not get stuck open), got {events}"
print("failure path: on_normcorre_done still fired despite exception:", events)

app.run_normcorre = orig_run_normcorre

print("ALL OK")
