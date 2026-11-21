"""Signal processing: filter responses, physical sanity, and chunk invariance (live batches must give
the same numbers as one pass over the whole trip)."""

import json
import math

import numpy as np
import pytest

from roadride import dsp

FS = dsp.FS
T0 = 1_795_000_000.0  # epoch seconds


def _phone_frame():
    """A fixed, tilted phone: rotation from vehicle (x fwd, y left, z up) to phone axes."""
    a, b = math.radians(25), math.radians(-40)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
    return rx @ rz


def synth(seconds, *, vert=lambda t: 0 * t, speed=lambda t: 0 * t + 12.0, lon=None, seed=1,
          jitter=True, rate=FS):
    """deviceMotion-like samples (epoch s, ua g, gravity g) and 1 Hz fixes, for a vehicle signal.
    ua has CoreMotion's sign: minus the phone's acceleration (dsp.UA_SIGN)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    if jitter:
        t = t + rng.uniform(-0.002, 0.002, n)
        t.sort()
    r = _phone_frame()
    a_vehicle = np.zeros((n, 3))
    a_vehicle[:, 2] = vert(t)
    if lon is not None:
        a_vehicle[:, 0] = lon(t)
    ua = dsp.UA_SIGN * (a_vehicle @ r.T) / dsp.G
    grav = np.tile(np.array([0, 0, -1.0]) @ r.T, (n, 1))
    fixes = [(T0 + s + 0.3, float(speed(np.array(s + 0.3)))) for s in range(int(seconds))]
    return T0 + t, ua, grav, fixes


def run_whole(t, ua, grav, fixes):
    p = dsp.Pipeline()
    return p.feed(t, ua, grav, fixes), p


def run_chunked(t, ua, grav, fixes, seed=7):
    rng = np.random.default_rng(seed)
    series, events, state, out = [], [], None, None
    start = t[0]
    while start <= t[-1]:
        end = start + rng.uniform(1.0, 3.0)
        m = (t >= start) & (t < end)
        f = [x for x in fixes if start <= x[0] < end]
        p = dsp.Pipeline(json.loads(json.dumps(state)) if state else None)  # DB round trip
        out = p.feed(t[m], ua[m], grav[m], f)
        series += out.series
        events += out.events
        state = p.to_state()
        start = end
    return series, events, out


# --- Wf filter ----------------------------------------------------------------------------

def test_wf_digital_matches_analog_in_the_sickness_band():
    f = np.geomspace(0.08, 0.63, 25)
    analog, digital = dsp.wf_analog_gain(f), dsp.wf_digital_gain(f)
    assert np.max(np.abs(digital / analog - 1)) < 0.015  # bilinear warping at 20 Hz: ≈ 1 % at 0.63 Hz


def test_wf_peaks_near_017_hz_and_rejects_road_frequencies():
    f = np.geomspace(0.02, 5, 400)
    g = dsp.wf_analog_gain(f)
    assert 0.14 < f[np.argmax(g)] < 0.20  # ISO 2631 Wf peaks at ≈ 0.17 Hz
    assert dsp.wf_analog_gain(np.array([4.0]))[0] < 0.01 * g.max()
    assert dsp.wf_analog_gain(np.array([0.02]))[0] < 0.05 * g.max()


# --- Physical sanity ----------------------------------------------------------------------

def test_vertical_sign_screen_up():
    p = dsp.Pipeline()
    grav = np.array([[0.0, 0.0, -1.0]])  # screen up: gravity along -z
    # CoreMotion in free fall: the accelerometer reads 0, so userAcceleration = 0 - gravity = +1 g on z.
    free_fall = np.array([[0.0, 0.0, 1.0]])
    assert p._vert(np.hstack([free_fall * p.ua_sign, grav]))[0] == pytest.approx(-dsp.G)  # falling


def test_a_trip_saved_before_the_sign_fix_keeps_its_sign():
    assert dsp.Pipeline().ua_sign == dsp.UA_SIGN
    old = dsp.Pipeline().to_state()
    del old["ua_sign"]
    assert dsp.Pipeline(old).ua_sign == dsp.LEGACY_UA_SIGN
    assert dsp.Pipeline(dsp.Pipeline().to_state()).ua_sign == dsp.UA_SIGN


def test_roughness_of_a_10hz_sine_is_its_rms():
    amp = 3.0
    t, ua, grav, fixes = synth(30, vert=lambda t: amp * np.sin(2 * np.pi * 10 * t))
    out, _ = run_whole(t, ua, grav, fixes)
    assert out.roughness_now == pytest.approx(amp / math.sqrt(2), rel=0.05)


def test_slow_sway_is_not_roughness():
    t, ua, grav, fixes = synth(60, vert=lambda t: 1.0 * np.sin(2 * np.pi * 0.2 * t))
    out, _ = run_whole(t, ua, grav, fixes)
    assert out.roughness_now < 0.02


def test_msdv_of_a_sine_matches_the_weighting():
    amp, f0, secs = 1.0, 0.16, 600
    t, ua, grav, fixes = synth(secs, vert=lambda t: amp * np.sin(2 * np.pi * f0 * t), jitter=False)
    out, _ = run_whole(t, ua, grav, fixes)
    expected = dsp.wf_analog_gain(np.array([f0]))[0] * amp / math.sqrt(2) * math.sqrt(secs)
    assert out.msdv == pytest.approx(expected, rel=0.05)


def _spike(at):
    return lambda t: 15.0 * np.exp(-((t - at) / 0.01) ** 2)


def test_pothole_detected_above_15_kmh():
    t, ua, grav, fixes = synth(10, vert=_spike(5.0), speed=lambda t: 0 * t + 10.0)
    out, _ = run_whole(t, ua, grav, fixes)
    (ev,) = out.events
    assert ev.type == "pothole" and abs(ev.t - (T0 + 5.0)) < 0.05 and ev.speed == pytest.approx(10.0)
    assert ev.magnitude > dsp.POTHOLE_THRESHOLD_MS2 and 0 < ev.confidence <= 1


def test_no_pothole_when_slow():
    t, ua, grav, fixes = synth(10, vert=_spike(5.0), speed=lambda t: 0 * t + 3.0)
    out, _ = run_whole(t, ua, grav, fixes)
    assert out.events == []


def test_gap_resets_filters_but_keeps_the_dose():
    a = synth(20, vert=lambda t: np.sin(2 * np.pi * 0.2 * t))
    p = dsp.Pipeline()
    p.feed(*a)
    dose = p.dose
    t, ua, grav, fixes = a
    out = p.feed(t + 25, ua, grav, [(x + 25, v) for x, v in fixes])
    assert out.reset and p.dose > dose


def _with_hole(seconds, start, length, **kw):
    """synth() with the samples of [start, start + length) s removed (a CoreMotion stall)."""
    t, ua, grav, fixes = synth(seconds, **kw)
    m = ~((t >= T0 + start) & (t < T0 + start + length))
    return t[m], ua[m], grav[m], fixes


def test_hole_inside_a_batch_resets_and_is_not_bridged():
    hole = (3.0, 0.6)  # a 0.6 s stall in the middle of one feed
    t, ua, grav, fixes = _with_hole(6, *hole, vert=lambda t: np.sin(2 * np.pi * 5 * t))
    out, p = run_whole(t, ua, grav, fixes)
    assert out.reset
    lo, hi = T0 + hole[0], T0 + hole[0] + hole[1]
    # The 5 Hz series lands on grid points: none of them may lie inside the hole (interpolated).
    times = [s.t for s in out.series]
    assert times and not any(lo < x < hi for x in times)
    assert all(b > a for a, b in zip(times, times[1:]))


def test_hole_inside_a_batch_equals_a_hole_between_batches():
    t, ua, grav, fixes = _with_hole(8, 3.0, 0.6, vert=lambda t: np.sin(2 * np.pi * 5 * t))
    whole, _ = run_whole(t, ua, grav, fixes)

    cut = int(np.searchsorted(t, T0 + 3.0))  # the first sample after the hole starts a new batch
    p = dsp.Pipeline()
    a = p.feed(t[:cut], ua[:cut], grav[:cut], [f for f in fixes if f[0] < t[cut]])
    p = dsp.Pipeline(json.loads(json.dumps(p.to_state())))
    b = p.feed(t[cut:], ua[cut:], grav[cut:], [f for f in fixes if f[0] >= t[cut]])
    assert b.reset
    split = a.series + b.series
    assert [round(s.t, 9) for s in split] == [round(s.t, 9) for s in whole.series]
    assert [s.a_vert for s in split] == pytest.approx([s.a_vert for s in whole.series], abs=1e-9)
    assert b.msdv == pytest.approx(whole.msdv, rel=1e-9)


def test_jitter_below_the_gap_threshold_does_not_reset():
    t, ua, grav, fixes = _with_hole(4, 2.0, dsp.MAX_GAP_S / 2)
    out, _ = run_whole(t, ua, grav, fixes)
    assert not out.reset


def test_forward_axis_learned_from_speed_changes():
    # Accelerate and brake every 10 s: dv/dt and the longitudinal acceleration agree.
    lon = lambda t: 1.5 * np.sign(np.sin(2 * np.pi * t / 20))  # noqa: E731

    def speed_from_lon(s):
        # Integrate the square wave: triangle between 10 and 25 m/s.
        phase = s % 20
        return 10 + (1.5 * phase if phase < 10 else 1.5 * (20 - phase))

    t, ua, grav, _ = synth(240, lon=lon)
    fixes = [(T0 + s + 0.3, speed_from_lon(s + 0.3)) for s in range(240)]
    out, p = run_whole(t, ua, grav, fixes)
    assert p.forward_known
    # The axis itself, not only a_lon: a backwards axis gives the same a_lon (both signs flip) but
    # inverts the pitch rate, and then no speed bump is found.
    fwd = np.array(p.forward_sum) / np.linalg.norm(p.forward_sum)
    assert fwd @ (_phone_frame() @ np.array([1.0, 0.0, 0.0])) > 0.99
    late = [pt for pt in out.series if pt.t > T0 + 120]
    corr = np.corrcoef([pt.a_lon for pt in late], [lon(np.array(pt.t - T0)) for pt in late])[0, 1]
    assert corr > 0.9


# --- One pipeline: chunked == whole ---------------------------------------------------------

def test_chunked_feed_equals_one_pass():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.4, 600 * 100 + 10)

    def vert(t):
        idx = np.clip((t * 100).astype(int), 0, len(noise) - 1)
        return 0.8 * np.sin(2 * np.pi * 0.18 * t) + noise[idx] + _spike(123.4)(t) + _spike(400.0)(t)

    t, ua, grav, fixes = synth(600, vert=vert, speed=lambda t: 12 + 5 * np.sin(t / 30))
    whole, _ = run_whole(t, ua, grav, fixes)
    series, events, last = run_chunked(t, ua, grav, fixes)

    assert len(series) == len(whole.series) > 2900
    for a, b in zip(series, whole.series):
        assert a.t == pytest.approx(b.t, abs=1e-9)
        assert a.a_vert == pytest.approx(b.a_vert, abs=1e-6)
        assert a.msdv == pytest.approx(b.msdv, rel=1e-6, abs=1e-9)
        assert a.rough_rms == pytest.approx(b.rough_rms, rel=1e-6, abs=1e-9)
        assert a.dose_inc == pytest.approx(b.dose_inc, rel=1e-6, abs=1e-12)
        assert (a.speed is None) == (b.speed is None)
    assert [(round(e.t, 6), e.type) for e in events] == [(round(e.t, 6), e.type) for e in whole.events]
    assert len(events) == 2
    assert last.msdv == pytest.approx(whole.msdv, rel=1e-6)
    assert last.roughness_now == pytest.approx(whole.roughness_now, rel=1e-6)


def test_overlapping_batches_are_not_interleaved():
    """A batch that starts 25 ms before the previous one ended (clock re-anchoring) must not feed
    out-of-order times to the interpolation: no false events, same roughness as aligned batches."""
    # 3 Hz: well inside the band, and slow enough that the half-sample shift below costs no amplitude
    # in linear interpolation (at 10 Hz it would lose ≈ 4 %, which is interpolation, not the overlap).
    amp = 2.0
    t, ua, grav, fixes = synth(20, vert=lambda t: amp * np.sin(2 * np.pi * 3 * t), jitter=False)
    half = len(t) // 2
    aligned = dsp.Pipeline()
    aligned.feed(t[:half], ua[:half], grav[:half], [f for f in fixes if f[0] < t[half]])
    ref = aligned.feed(t[half:], ua[half:], grav[half:], [f for f in fixes if f[0] >= t[half]])

    p = dsp.Pipeline()
    first = p.feed(t[:half], ua[:half], grav[:half], [f for f in fixes if f[0] < t[half]])
    back = half - 3  # the second batch repeats the last 3 samples with times shifted 5 ms earlier
    out = p.feed(t[back:] - 0.005, ua[back:], grav[back:], [f for f in fixes if f[0] >= t[half]])

    assert not out.reset and out.events == [] and first.events == []
    times = [s.t for s in first.series + out.series]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert out.roughness_now == pytest.approx(ref.roughness_now, rel=0.02)


def test_clock_jumping_back_resets():
    t, ua, grav, fixes = synth(4)
    p = dsp.Pipeline()
    p.feed(t, ua, grav, fixes)
    out = p.feed(t - 2.0, ua, grav, [])
    assert out.reset


def test_window_sums_add_up_to_the_trip():
    """The per-window outputs the finalize job builds segment passes from must agree with the trip
    totals: Σ dose_inc = MSDV², and a 10 Hz sine's window RMS is its RMS."""
    amp = 3.0
    t, ua, grav, fixes = synth(60, vert=lambda t: amp * np.sin(2 * np.pi * 10 * t) + np.sin(0.9 * t))
    out, _ = run_whole(t, ua, grav, fixes)
    assert sum(p.dose_inc for p in out.series) == pytest.approx(out.msdv ** 2, rel=0.01)
    steady = [p.rough_rms for p in out.series if p.t > T0 + 5]
    assert np.median(steady) == pytest.approx(amp / math.sqrt(2), rel=0.05)


def test_state_is_json():
    t, ua, grav, fixes = synth(5)
    p = dsp.Pipeline()
    p.feed(t, ua, grav, fixes)
    assert json.loads(json.dumps(p.to_state())) == p.to_state()


# --- Speed bumps ----------------------------------------------------------------------------

def _pulse(at, amp, width=0.08):
    return lambda t: amp * np.exp(-((t - at) / width) ** 2)


def synth_bumps(seconds, bumps, *, speed=15 / 3.6 + 0.2, wheelbase=2.7, rate_amp=0.4, vert_amp=15.0,
                extra_vert=None, seed=2):
    """synth() plus a rotation rate: each bump at time b pitches the nose up (front axle climbing) and,
    one wheelbase later, down (rear axle), with a vertical jolt at both axles."""
    t, ua, grav, fixes = synth(seconds, speed=lambda t: 0 * t + speed, seed=seed)
    rel = t - T0
    pitch_up = np.zeros_like(rel)
    vert = np.zeros_like(rel) if extra_vert is None else extra_vert(rel)
    for b in bumps:
        rear = b + wheelbase / speed
        pitch_up += _pulse(b, rate_amp)(rel) - _pulse(rear, rate_amp)(rel)
        vert += _pulse(b, vert_amp, 0.01)(rel) + _pulse(rear, vert_amp, 0.01)(rel)
    r = _phone_frame()
    ua = ua + dsp.UA_SIGN * (np.column_stack([0 * rel, 0 * rel, vert]) @ r.T) / dsp.G
    # Nose up is a negative rotation about the vehicle's left (+y) axis.
    rr = np.column_stack([0 * rel, -pitch_up, 0 * rel]) @ r.T
    return t, ua, grav, fixes, rr


def _forward_known_state():
    fwd = _phone_frame() @ np.array([1.0, 0.0, 0.0])
    return {"forward_sum": (fwd * dsp.FWD_MIN_EVIDENCE * 2).tolist(), "forward_n": dsp.FWD_MIN_EVIDENCE * 2}


def _feed_bumps(t, ua, grav, fixes, rr, state=None):
    p = dsp.Pipeline(state if state is not None else _forward_known_state())
    out = p.feed(t, ua, grav, fixes, rr=rr)
    return out, p


def test_pitch_rate_sign_and_axis():
    p = dsp.Pipeline(_forward_known_state())
    r = _phone_frame()
    down = np.tile(np.array([0, 0, -1.0]) @ r.T, (3, 1))
    rr = np.array([[0, -0.3, 0], [0.3, 0, 0], [0, 0, 0.3]]) @ r.T  # nose up, roll, yaw
    assert p._pitch_up(down, rr) == pytest.approx([0.3, 0.0, 0.0], abs=1e-9)
    assert dsp.Pipeline()._pitch_up(down, rr) == pytest.approx([0, 0, 0])  # forward axis unknown


def test_speed_bump_detected_and_its_jolts_are_not_potholes():
    t, ua, grav, fixes, rr = synth_bumps(12, [5.0])
    out, _ = _feed_bumps(t, ua, grav, fixes, rr)
    (ev,) = out.events
    assert ev.type == "speed_bump" and abs(ev.t - (T0 + 5.0)) < 0.1
    assert ev.magnitude > dsp.POTHOLE_THRESHOLD_MS2 and 0 < ev.confidence <= 1
    assert ev.speed == pytest.approx(15 / 3.6 + 0.2)


def test_speed_bumps_found_with_a_learned_forward_axis():
    """As on a real trip: no preset axis. 3 min of speed changes teach it, then 4 bumps at 15 km/h.
    With the raw CoreMotion sign the axis came out backwards and all 4 were reported as potholes."""
    secs, learn_s, v = 230, 180, 15 / 3.6 + 0.2
    lon = lambda t: np.where(t < learn_s, 4 * 2 * np.pi / 20 * np.cos(2 * np.pi * t / 20), 0.0)  # noqa: E731
    speed = lambda t: np.where(t < learn_s, 8 + 4 * np.sin(2 * np.pi * t / 20), v)  # noqa: E731
    t, ua, grav, fixes = synth(secs, lon=lon, speed=speed, seed=5)
    rel = t - T0
    bumps = [190.0, 200.0, 210.0, 220.0]
    pitch_up, vert = np.zeros_like(rel), np.zeros_like(rel)
    for b in bumps:
        rear = b + 2.7 / v
        pitch_up += _pulse(b, 0.4)(rel) - _pulse(rear, 0.4)(rel)
        vert += _pulse(b, 15.0, 0.01)(rel) + _pulse(rear, 15.0, 0.01)(rel)
    r = _phone_frame()
    ua = ua + dsp.UA_SIGN * (np.column_stack([0 * rel, 0 * rel, vert]) @ r.T) / dsp.G
    rr = np.column_stack([0 * rel, -pitch_up, 0 * rel]) @ r.T
    p = dsp.Pipeline()
    out = p.feed(t, ua, grav, fixes, rr=rr)
    events = out.events + p.flush()
    assert [round(e.t - T0) for e in events if e.type == "speed_bump"] == bumps
    assert not [e for e in events if e.type == "pothole"]
    # a_vert is positive upwards: the jolt of the front axle climbing the bump.
    jolt = max((s for s in out.series if abs(s.t - T0 - bumps[0]) < 0.3), key=lambda s: abs(s.a_vert))
    assert jolt.a_vert > dsp.POTHOLE_THRESHOLD_MS2


@pytest.mark.parametrize("wheelbase", [2.2, 3.0, 6.0])
def test_speed_bump_for_car_and_bus_wheelbases(wheelbase):
    t, ua, grav, fixes, rr = synth_bumps(14, [5.0], speed=6.0, wheelbase=wheelbase)
    out, _ = _feed_bumps(t, ua, grav, fixes, rr)
    assert [e.type for e in out.events] == ["speed_bump"]


def test_no_speed_bump_without_the_forward_axis():
    t, ua, grav, fixes, rr = synth_bumps(12, [5.0])
    out, _ = _feed_bumps(t, ua, grav, fixes, rr, state={})
    # As before speed bumps existed: the jolts are potholes (one, since the second is within 1 s).
    assert [e.type for e in out.events] == ["pothole"]


def test_peaks_too_far_apart_for_an_axle_pair_are_not_a_bump():
    t, ua, grav, fixes, rr = synth_bumps(14, [5.0], wheelbase=15.0)
    out, _ = _feed_bumps(t, ua, grav, fixes, rr)
    assert [e.type for e in out.events] == ["pothole", "pothole"]


def test_no_speed_bump_when_fast_or_from_one_pitch_peak():
    t, ua, grav, fixes, rr = synth_bumps(12, [5.0], speed=60 / 3.6, vert_amp=0.0)
    assert _feed_bumps(t, ua, grav, fixes, rr)[0].events == []
    t, ua, grav, fixes, _ = synth_bumps(12, [], vert_amp=0.0)
    rr = np.column_stack([0 * t, -_pulse(5.0, 0.4)(t - T0), 0 * t]) @ _phone_frame().T  # a crest
    assert _feed_bumps(t, ua, grav, fixes, rr)[0].events == []


def test_potholes_are_held_back_and_flushed_at_the_end():
    t, ua, grav, fixes = synth(10, vert=_spike(8.5), speed=lambda t: 0 * t + 10.0)
    out, p = run_whole(t, ua, grav, fixes)
    assert out.events == []  # still within POTHOLE_HOLD_S of the end
    (ev,) = p.flush()
    assert ev.type == "pothole" and abs(ev.t - (T0 + 8.5)) < 0.05
    assert p.flush() == [] and p.to_state()["held"] == []


def test_speed_bumps_chunked_equal_one_pass():
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.3, 200 * 100 + 10)
    extra = lambda rel: noise[np.clip((rel * 100).astype(int), 0, len(noise) - 1)] + _spike(150.0)(rel)  # noqa: E731
    t, ua, grav, fixes, rr = synth_bumps(200, [20.0, 61.3, 120.7, 181.0], speed=5.5, extra_vert=extra)
    whole, p = _feed_bumps(t, ua, grav, fixes, rr)
    whole_events = whole.events + p.flush()

    rng = np.random.default_rng(9)
    events, state, start = [], _forward_known_state(), t[0]
    while start <= t[-1]:
        end = start + rng.uniform(1.0, 3.0)
        m = (t >= start) & (t < end)
        q = dsp.Pipeline(json.loads(json.dumps(state)))
        events += q.feed(t[m], ua[m], grav[m], [x for x in fixes if start <= x[0] < end], rr=rr[m]).events
        state = q.to_state()
        start = end
    events += dsp.Pipeline(state).flush()

    key = lambda es: [(e.type, round(e.t, 6), e.magnitude) for e in es]  # noqa: E731
    assert key(events) == key(whole_events)
    assert [e.type for e in whole_events].count("speed_bump") == 4
    assert [e.type for e in whole_events].count("pothole") == 1


def test_a_trip_saved_before_rotation_rate_continues():
    """State written by the previous version (no rotation channel, pitch filter, held potholes)."""
    t, ua, grav, fixes = synth(8, vert=_spike(6.0), speed=lambda t: 0 * t + 10.0)
    half = len(t) // 2
    p = dsp.Pipeline()
    p.feed(t[:half], ua[:half], grav[:half], [f for f in fixes if f[0] < t[half]])
    old = json.loads(json.dumps(p.to_state()))
    for k in ("held", "last_bump_t"):
        old.pop(k)
    for k in ("zi_pitch", "bump"):
        old["grid"].pop(k)
    old["grid"]["prev"] = old["grid"]["prev"][:7]
    q = dsp.Pipeline(old)
    out = q.feed(t[half:], ua[half:], grav[half:], [f for f in fixes if f[0] >= t[half]])
    assert not out.reset and [e.type for e in out.events + q.flush()] == ["pothole"]
