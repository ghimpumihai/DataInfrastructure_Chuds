"""Signal processing for one trip (plan §9 steps 2, 4, 5, 6): vertical acceleration, road roughness,
potholes, speed bumps and the ISO 2631-1 motion-sickness dose.

One pure pipeline, fed in pieces. The stream processor feeds it one 2-second batch at a time and saves
`to_state()` in the database with the rows it wrote; the finalize job can feed a whole trip at once.
Both must give the same numbers, so every piece of memory (filter states, the resampling grid, partial
windows, the dose, the forward-axis estimate, an open pothole) lives in the JSON-serializable state,
and tests feed the same signal whole and in random chunks.

Inputs are the app's deviceMotion samples: user acceleration and gravity in g and the rotation rate
in rad/s, in the phone's frame, at a nominal 100 Hz with jitter. CoreMotion's userAcceleration is the
negative of the phone's acceleration (see UA_SIGN), so it is flipped once on the way in. The samples
are resampled onto a uniform 100 Hz grid (IIR filters assume a fixed rate); a time gap (between two
batches or inside one) or a missing batch resets the filters and starts a new grid, so nothing is
interpolated across a hole.

ISO 2631-1 Wf (motion sickness): H = Hh·Hl·Ht·Hs with f1=0.08, f2=0.63, f3=∞, f4=0.25, Q4=0.86,
f5=0.0625, Q5=0.80, f6=0.1, Q6=0.80 Hz (parameters and transfer functions as reproduced in the Chalmers
thesis odr.chalmers.se/bitstreams/7acd3e0a-09bd-4914-a01e-cfe3832b3ab3; MSDV = (∫a_w² dt)^½ as in
arXiv:2501.17664 eq. 3). Its poles sit far below 100 Hz, so the signal is low-passed and decimated to
20 Hz before Wf is applied.

Speed bumps (plan §9 step 5): a double peak in the vehicle's pitch rate, nose up as the front axle
climbs, then nose down as the rear axle climbs, one wheelbase apart (2–7 m, car to bus or tram) at the
GPS speed. Pitch is the rotation about the lateral axis, so it needs the learned forward axis: until
that is known, no speed bump is detected. A speed bump also shakes the vertical channel, so potholes
are held back for POTHOLE_HOLD_S and dropped if a speed bump turns out to cover them; `flush()` releases
the ones still held at the end of a trip.

Thresholds for potholes, speed bumps and the forward-axis estimate are PROVISIONAL: they haven't seen
a real drive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

G = 9.80665  # m/s² per g
# CoreMotion reports a phone lying screen up at rest as -1 g on z and a falling one as 0, and
# userAcceleration = total - gravity: in free fall that is +1 g "up" while the phone falls. So the
# phone's acceleration is -userAcceleration. Taken as is, the learned forward axis pointed backwards,
# which flipped the pitch rate and hid every speed bump (and a_vert was positive downwards).
UA_SIGN = -1.0
LEGACY_UA_SIGN = 1.0  # trips whose saved state predates UA_SIGN

FS = 100.0  # resampling grid, Hz
DT = 1.0 / FS
DECIMATE = 5  # → 20 Hz for Wf: low enough for well-conditioned poles at 0.06 Hz, high enough that
#                 the bilinear transform's frequency warping stays < 1 % up to 0.63 Hz (10 Hz gave 4 %)
FS_WF = FS / DECIMATE
SERIES_EVERY = 20  # → 5 Hz session_series
MAX_GAP_S = 0.25  # a hole longer than this in the motion samples resets the filters

ROUGH_BAND_HZ = (1.0, 30.0)  # plan §9 step 4
ROUGHNESS_TAU_S = 5.0  # roughnessNow: RMS over roughly the last 5 s (exponential average)

POTHOLE_THRESHOLD_MS2 = 5.0  # PROVISIONAL: |band-passed vertical| above this starts a pothole
POTHOLE_MIN_SPEED_MS = 15 / 3.6  # plan §9 step 5: only above 15 km/h
POTHOLE_PEAK_WINDOW_S = 0.3  # the event's peak is searched this long after the crossing
POTHOLE_REFRACTORY_S = 1.0  # one pothole per second at most

BUMP_BAND_HZ = (0.5, 5.0)  # body pitch from an axle hitting a bump; drops gyro noise and slow hill pitch
BUMP_PITCH_RATE = 0.15  # PROVISIONAL: rad/s of nose-up / nose-down pitch rate that starts a peak
BUMP_MIN_SPEED_MS = 1.5  # slower than this the wheelbase spacing is too long to tell apart
BUMP_MAX_SPEED_MS = 40 / 3.6  # nobody takes a speed bump faster (the dashboard's mock uses the same)
BUMP_WHEELBASE_M = (2.0, 7.0)  # front to rear axle: small car to bus / tram
BUMP_PEAK_MAX_S = 0.5  # a longer stretch of pitch rate (a hill crest, a slow turn) isn't an axle
BUMP_REFRACTORY_S = 2.0
BUMP_SPAN_PAD_S = 0.5  # potholes this close around a speed bump's two peaks belong to it
# A pothole is reported once no speed bump can cover it any more: at 15 km/h (the pothole minimum) the
# rear axle follows ≤ 7 m / 4.2 m/s ≈ 1.7 s later, plus the pad and the end of that peak.
POTHOLE_HOLD_S = 2.5

FIX_MAX_HOLD_S = 3.0  # use the last GPS speed for at most this long
FWD_MIN_DVDT = 0.5  # m/s², only clear speed changes say where "forward" is
FWD_MIN_DT, FWD_MAX_DT = 0.5, 3.0  # s between the two fixes of a speed change
FWD_MIN_EVIDENCE = 10  # speed changes before the forward axis is trusted

WF = dict(f1=0.08, f2=0.63, f4=0.25, q4=0.86, f5=0.0625, q5=0.80, f6=0.1, q6=0.80)


# --- Filters ------------------------------------------------------------------------------

def _w(f: float) -> float:
    return 2 * math.pi * f


def wf_analog_factors() -> list[tuple[list[float], list[float]]]:
    """Wf as four analog (b, a) factors in s (f3 = ∞, so Ht has no zero)."""
    w1, w2, w4, w5, w6 = (_w(WF[k]) for k in ("f1", "f2", "f4", "f5", "f6"))
    hh = ([1.0, 0.0, 0.0], [1.0, math.sqrt(2) * w1, w1 ** 2])  # high-pass, band limiting
    hl = ([w2 ** 2], [1.0, math.sqrt(2) * w2, w2 ** 2])  # low-pass, band limiting
    ht = ([w4 ** 2], [1.0, w4 / WF["q4"], w4 ** 2])  # acceleration-velocity transition
    # Upward step: (1 + p/(Q5ω5) + (p/ω5)²)/(1 + p/(Q6ω6) + (p/ω6)²) · (ω5/ω6)² = N5(s)/N6(s)
    hs = ([1.0, w5 / WF["q5"], w5 ** 2], [1.0, w6 / WF["q6"], w6 ** 2])
    return [hh, hl, ht, hs]


def wf_analog_gain(freqs_hz: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * np.asarray(freqs_hz, dtype=float)
    gain = np.ones_like(w)
    for b, a in wf_analog_factors():
        gain *= np.abs(signal.freqs(b, a, worN=w)[1])
    return gain


def _wf_sos(fs: float) -> np.ndarray:
    zs, ps, k = [], [], 1.0
    for b, a in wf_analog_factors():
        z, p, kk = signal.tf2zpk(b, a)
        zd, pd, kd = signal.bilinear_zpk(z, p, kk, fs)
        zs.extend(zd)
        ps.extend(pd)
        k *= kd
    return signal.zpk2sos(np.array(zs), np.array(ps), k)


SOS_BAND = signal.butter(4, ROUGH_BAND_HZ, btype="bandpass", fs=FS, output="sos")
SOS_LOW = signal.butter(4, 2.0, btype="lowpass", fs=FS, output="sos")  # anti-alias before 10 Hz
SOS_WF = _wf_sos(FS_WF)
SOS_PITCH = signal.butter(2, BUMP_BAND_HZ, btype="bandpass", fs=FS, output="sos")


def wf_digital_gain(freqs_hz: np.ndarray) -> np.ndarray:
    return np.abs(signal.sosfreqz(SOS_WF, worN=np.asarray(freqs_hz, dtype=float), fs=FS_WF)[1])


# --- Results ------------------------------------------------------------------------------

@dataclass
class SeriesPoint:
    t: float  # epoch seconds
    a_vert: float  # m/s², band-passed vertical, the largest |value| of the 0.2 s window (signed)
    a_lat: float  # m/s², low-passed; 0 until the forward axis is known
    a_lon: float
    speed: float | None  # m/s
    msdv: float  # m/s^1.5 so far
    # For the finalize job's segment passes (not stored in session_series):
    rough_rms: float = 0.0  # m/s², RMS of the band-passed vertical over the 0.2 s window
    dose_inc: float = 0.0  # ∫ a_w² dt over the window (m²/s³), so a pass's a_w RMS = √(Σ dose_inc / Σ dt)


@dataclass
class Event:
    t: float
    type: str
    magnitude: float  # m/s²
    speed: float | None
    confidence: float


@dataclass
class FeedResult:
    series: list[SeriesPoint] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    roughness_now: float | None = None
    msdv: float = 0.0
    reset: bool = False  # filters restarted (gap) during this feed


# --- Pipeline -----------------------------------------------------------------------------

def _zeros(sos: np.ndarray, channels: int = 1) -> list:
    return np.zeros((channels, sos.shape[0], 2)).tolist()


class Pipeline:
    def __init__(self, state: dict | None = None):
        s = state or {}
        self.dose = float(s.get("dose", 0.0))  # ∫ a_w² dt over the whole trip, never reset
        # A trip in progress during the deploy (saved with a dose, without ua_sign) keeps the old sign so
        # its filters and forward axis carry on consistently; the finalize job redoes it with UA_SIGN.
        self.ua_sign = float(s.get("ua_sign", LEGACY_UA_SIGN if "dose" in s else UA_SIGN))
        self.forward_sum = list(s.get("forward_sum", [0.0, 0.0, 0.0]))
        self.forward_n = int(s.get("forward_n", 0))
        self.prev_fix = s.get("prev_fix")  # [t, speed or None]
        self.h_acc = list(s.get("h_acc", [0.0, 0.0, 0.0]))
        self.h_n = int(s.get("h_n", 0))
        self.last_event_t = s.get("last_event_t")
        self.open_event = s.get("open_event")  # [t, peak, speed] while searching the peak
        self.rough_ema = s.get("rough_ema")
        self.grid = s.get("grid")  # None until the first sample, then the segment's resampling state
        # Fixes newer than the last processed sample wait here, so they're taken at their own time
        # whatever the batch boundaries are.
        self.pending_fixes = [tuple(f) for f in s.get("pending_fixes", [])]
        self.held = [list(h) for h in s.get("held", [])]  # potholes [t, peak, speed] waiting for POTHOLE_HOLD_S
        self.last_bump_t = s.get("last_bump_t")
        if self.grid is not None:
            self._upgrade_grid(self.grid)

    def to_state(self) -> dict:
        return {
            "dose": self.dose, "ua_sign": self.ua_sign,
            "forward_sum": self.forward_sum, "forward_n": self.forward_n,
            "prev_fix": self.prev_fix, "h_acc": self.h_acc, "h_n": self.h_n,
            "last_event_t": self.last_event_t, "open_event": self.open_event,
            "rough_ema": self.rough_ema, "grid": self.grid,
            "pending_fixes": [list(f) for f in self.pending_fixes],
            "held": self.held, "last_bump_t": self.last_bump_t,
        }

    @staticmethod
    def _upgrade_grid(gr: dict) -> None:
        """A trip that was in progress when the rotation rate was added: its saved grid has no
        rotation-rate channel or pitch filter yet. Zero rotation for the interpolation's first point
        only delays the first speed bump by a sample."""
        if len(gr["prev"]) == 7:
            gr["prev"] = gr["prev"] + [0.0, 0.0, 0.0]
        gr.setdefault("zi_pitch", _zeros(SOS_PITCH)[0])
        gr.setdefault("bump", {"peak": None, "p1": None})

    def flush(self) -> list[Event]:
        """End of the trip: the potholes still held back (and one whose peak is still being searched)
        are final now. The finalize job calls this after the last batch; the live path never does,
        so the last POTHOLE_HOLD_S of a trip only appear once it is finalized."""
        if self.open_event is not None:
            self.held.append(self.open_event)
            self.open_event = None
        out = [self._pothole_event(h) for h in self.held]
        self.held = []
        return out

    # -- helpers --

    @property
    def forward_known(self) -> bool:
        return self.forward_n >= FWD_MIN_EVIDENCE and np.linalg.norm(self.forward_sum) > 0

    def _new_grid(self, t: float, sample: list[float]) -> dict:
        ua_g = np.array(sample[:6])
        x0 = self._vert(ua_g[None, :])[0]
        return {
            "origin": t, "k": 0, "prev": [t] + list(sample),
            "zi_band": _zeros(SOS_BAND)[0],
            "zi_low": (signal.sosfilt_zi(SOS_LOW) * x0).tolist(),
            "zi_h": _zeros(SOS_LOW, 3),
            "zi_wf": _zeros(SOS_WF)[0],
            "zi_pitch": _zeros(SOS_PITCH)[0],
            "win": {"n": 0, "peak": 0.0, "lat": 0.0, "lon": 0.0, "b2": 0.0, "aw2": 0.0},
            # Speed-bump search: the open pitch peak, and the nose-up peak waiting for its rear axle.
            # Kept in the grid so a gap (new grid) drops a half-seen bump.
            "bump": {"peak": None, "p1": None},
        }

    @staticmethod
    def _down(g: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(g, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return g / n

    def _vert(self, ua_g: np.ndarray) -> np.ndarray:
        """Upward acceleration, m/s²: the phone's acceleration (userAcceleration · ua_sign) projected on
        minus the gravity direction."""
        down = self._down(ua_g[:, 3:6])
        return -np.einsum("ij,ij->i", ua_g[:, 0:3], down) * G

    def _speed_at(self, t: float) -> float | None:
        """Causal: the last fix at or before t (fixes are taken in time order before each sample).
        Never looks ahead, so a batch that doesn't contain the next fix yet gives the same answer."""
        before = self.prev_fix
        if before is None or before[1] is None or before[0] > t:
            return None
        return before[1] if t - before[0] <= FIX_MAX_HOLD_S else None

    def _take_fix(self, t: float, speed: float | None) -> None:
        if self.prev_fix and self.prev_fix[1] is not None and speed is not None and self.h_n > 0:
            dt_fix = t - self.prev_fix[0]
            if FWD_MIN_DT <= dt_fix <= FWD_MAX_DT:
                dvdt = (speed - self.prev_fix[1]) / dt_fix
                if abs(dvdt) >= FWD_MIN_DVDT:
                    mean_h = np.array(self.h_acc) / self.h_n
                    self.forward_sum = (np.array(self.forward_sum) + dvdt * mean_h).tolist()
                    self.forward_n += 1
        self.h_acc, self.h_n = [0.0, 0.0, 0.0], 0
        self.prev_fix = [t, speed]

    # -- main entry --

    def feed(self, t: np.ndarray, ua: np.ndarray, grav: np.ndarray,
             fixes: list[tuple[float, float | None]], *, reset: bool = False,
             rr: np.ndarray | None = None) -> FeedResult:
        """t: epoch seconds (n,), ua (CoreMotion's userAcceleration) and grav: (n, 3) in g, rr: (n, 3)
        rad/s (None = no rotation, so no speed bumps), all in the phone frame. fixes: (epoch s, speed m/s)."""
        out = FeedResult()
        fixes = sorted(set(self.pending_fixes) | {(float(ft), fv) for ft, fv in fixes},
                       key=lambda f: f[0])
        self.pending_fixes = []
        order = np.argsort(t, kind="stable")
        t = np.asarray(t, dtype=float)[order]
        rr = np.zeros((len(t), 3)) if rr is None else np.asarray(rr, float).reshape(-1, 3)
        samples = np.hstack([np.asarray(ua, float).reshape(-1, 3) * self.ua_sign,
                             np.asarray(grav, float).reshape(-1, 3),
                             rr])[order]
        if reset:
            self.grid = None
            out.reset = True
        if len(t) and self.grid is not None and abs(t[0] - self.grid["prev"][0]) > MAX_GAP_S:
            # A hole, or the clock jumped backwards: either way the filters can't continue.
            self.grid = None
            out.reset = True
        if len(t) == 0:
            self.pending_fixes = fixes  # no samples to place them against yet
            out.msdv = math.sqrt(self.dose)
            out.roughness_now = self._rough()
            return out
        # A stall inside the batch (CoreMotion pausing for a moment) is treated exactly like a hole
        # between two batches: the batch is split there and each piece starts a new grid, so the
        # filters restart and nothing is interpolated across the hole.
        cuts = (np.flatnonzero(np.diff(t) > MAX_GAP_S) + 1).tolist()
        for i, (a, b) in enumerate(zip([0] + cuts, cuts + [len(t)])):
            if i > 0:
                self.grid = None
                out.reset = True
            self._feed_segment(t[a:b], samples[a:b], fixes, out)
            fixes, self.pending_fixes = self.pending_fixes, []
        self.pending_fixes = fixes
        out.msdv = math.sqrt(self.dose)
        out.roughness_now = self._rough()
        return out

    def _feed_segment(self, t: np.ndarray, samples: np.ndarray,
                      fixes: list[tuple[float, float | None]], out: FeedResult) -> None:
        """Samples without a hole longer than MAX_GAP_S. Leaves the fixes not taken yet in
        self.pending_fixes."""
        if self.grid is None:
            self.grid = self._new_grid(float(t[0]), samples[0].tolist())

        gr = self.grid
        # Resample onto the uniform grid, interpolating from the last sample of the previous feed.
        src_t = np.concatenate([[gr["prev"][0]], t])
        src = np.vstack([np.array(gr["prev"][1:])[None, :], samples])
        # np.interp needs strictly increasing x and doesn't check it. Keep only samples later than
        # everything before them (an overlap with the previous batch is dropped, not interleaved).
        running_max = np.maximum.accumulate(src_t)
        keep = np.concatenate([[True], src_t[1:] > running_max[:-1]])
        src_t, src = src_t[keep], src[keep]
        k0 = gr["k"]
        k1 = int(math.floor((src_t[-1] - gr["origin"]) / DT + 1e-9))
        if k1 >= k0:
            ks = np.arange(k0, k1 + 1)
            gt = gr["origin"] + ks * DT
            grid = np.column_stack([np.interp(gt, src_t, src[:, c]) for c in range(src.shape[1])])
            self._process(ks, gt, grid, fixes, out)
            gr["k"] = int(k1 + 1)
        else:
            self.pending_fixes = fixes
        gr["prev"] = [float(src_t[-1])] + src[-1].tolist()

    def _rough(self) -> float | None:
        return None if self.rough_ema is None else math.sqrt(self.rough_ema)

    def _process(self, ks: np.ndarray, gt: np.ndarray, grid: np.ndarray,
                 fixes: list[tuple[float, float | None]], out: FeedResult) -> None:
        gr = self.grid
        down = self._down(grid[:, 3:6])
        ua = grid[:, 0:3]
        vert = -np.einsum("ij,ij->i", ua, down) * G
        horiz = (ua + np.einsum("ij,ij->i", ua, down)[:, None] * down * -1.0) * G  # ua minus its vertical part

        band, zb = signal.sosfilt(SOS_BAND, vert, zi=np.array(gr["zi_band"]))
        low, zl = signal.sosfilt(SOS_LOW, vert, zi=np.array(gr["zi_low"]))
        zh = np.array(gr["zi_h"])
        h_low = np.empty_like(horiz)
        for c in range(3):
            h_low[:, c], zh[c] = signal.sosfilt(SOS_LOW, horiz[:, c], zi=zh[c])
        gr["zi_band"], gr["zi_low"], gr["zi_h"] = zb.tolist(), zl.tolist(), zh.tolist()

        # Motion-sickness dose: Wf on the 20 Hz decimation of the low-passed vertical.
        dec = ks % DECIMATE == 0
        if dec.any():
            aw, zw = signal.sosfilt(SOS_WF, low[dec], zi=np.array(gr["zi_wf"]))
            gr["zi_wf"] = zw.tolist()
        else:
            aw = np.empty(0)
        aw_iter = iter(aw)

        alpha = DT / ROUGHNESS_TAU_S
        fix_i = 0
        win = gr["win"]
        zp = np.array(gr["zi_pitch"])
        n = len(ks)
        i = 0
        while i < n:
            while fix_i < len(fixes) and fixes[fix_i][0] <= gt[i]:
                self._take_fix(*fixes[fix_i])
                fix_i += 1
            # The forward axis only changes when a fix is taken, so the pitch rate is computed and
            # filtered per stretch up to the next fix: the same whatever the batch boundaries are.
            j = int(np.searchsorted(gt, fixes[fix_i][0], side="left")) if fix_i < len(fixes) else n
            pitch, zp = signal.sosfilt(SOS_PITCH, self._pitch_up(down[i:j], grid[i:j, 6:9]), zi=zp)
            for k in range(i, j):
                tt = float(gt[k])
                b = float(band[k])
                self.rough_ema = b * b if self.rough_ema is None else self.rough_ema + alpha * (b * b - self.rough_ema)
                win["b2"] = win.get("b2", 0.0) + b * b
                if dec[k]:
                    a = float(next(aw_iter))
                    self.dose += a * a / FS_WF
                    win["aw2"] = win.get("aw2", 0.0) + a * a / FS_WF
                self.h_acc = (np.array(self.h_acc) + h_low[k]).tolist() if self.h_n else h_low[k].tolist()
                self.h_n += 1

                self._pothole(tt, b, out)
                self._bump(tt, float(pitch[k - i]), b, out)
                while self.held and tt - self.held[0][0] > POTHOLE_HOLD_S:
                    out.events.append(self._pothole_event(self.held.pop(0)))

                # 5 Hz series window
                if abs(b) > abs(win["peak"]) or win["n"] == 0:
                    win["peak"] = b
                if self.forward_known:
                    fwd, lat = self._axes(down[k])
                    win["lon"] += float(h_low[k] @ fwd)
                    win["lat"] += float(h_low[k] @ lat)
                win["n"] += 1
                if (int(ks[k]) + 1) % SERIES_EVERY == 0:
                    out.series.append(SeriesPoint(
                        t=tt, a_vert=round(win["peak"], 4), a_lat=round(win["lat"] / win["n"], 4),
                        a_lon=round(win["lon"] / win["n"], 4), speed=self._speed_at(tt),
                        msdv=round(math.sqrt(self.dose), 5),
                        rough_rms=math.sqrt(win.get("b2", 0.0) / win["n"]), dose_inc=win.get("aw2", 0.0)))
                    win.update(n=0, peak=0.0, lat=0.0, lon=0.0, b2=0.0, aw2=0.0)
            i = j
        gr["zi_pitch"] = zp.tolist()
        self.pending_fixes = list(fixes[fix_i:])  # later than the last sample: taken next time

    def _axes(self, down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = np.array(self.forward_sum)
        f = f - (f @ down) * down  # keep it horizontal for the current attitude
        n = np.linalg.norm(f)
        f = f / n if n > 0 else f
        left = np.cross(-down, f)  # up × forward = left; aLat is positive to the left
        return f, left

    def _pitch_up(self, down: np.ndarray, rr: np.ndarray) -> np.ndarray:
        """Nose-up pitch rate, rad/s: minus the rotation about the lateral (left) axis, since turning
        about +left takes forward towards down. Zero until the forward axis is known."""
        if not self.forward_known or len(down) == 0:
            return np.zeros(len(down))
        f = np.array(self.forward_sum)
        fh = f[None, :] - (down @ f)[:, None] * down
        norm = np.linalg.norm(fh, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        left = np.cross(-down, fh / norm)
        return -np.einsum("ij,ij->i", rr, left)

    def _bump(self, t: float, q: float, b: float, out: FeedResult) -> None:
        """One sample of the speed-bump search; q = band-passed nose-up pitch rate, b = band-passed
        vertical acceleration (for the event's magnitude, in m/s² like a pothole's)."""
        st = self.grid["bump"]
        p1 = st["p1"]
        if p1 is not None:
            p1[2] = max(p1[2], abs(b))
            if t - p1[0] > BUMP_WHEELBASE_M[1] / p1[3] + BUMP_PEAK_MAX_S:
                st["p1"] = p1 = None  # no rear axle in time
        pk = st["peak"]  # [t start, t of the extreme, signed extreme, max |b|, too long]
        if pk is not None:
            if q * pk[2] > 0 and abs(q) >= BUMP_PITCH_RATE / 2:  # still the same peak (with hysteresis)
                if abs(q) > abs(pk[2]):
                    pk[1], pk[2] = t, q
                pk[3] = max(pk[3], abs(b))
                pk[4] = pk[4] or t - pk[0] > BUMP_PEAK_MAX_S
                return
            st["peak"] = None
            if not pk[4]:  # a longer one isn't an axle
                self._bump_peak(pk, out)
        if abs(q) >= BUMP_PITCH_RATE:
            st["peak"] = [t, t, q, abs(b), False]

    def _bump_peak(self, pk: list, out: FeedResult) -> None:
        st = self.grid["bump"]
        t, q = pk[1], pk[2]
        if q > 0:  # nose up: a front axle climbing, the possible start of a bump
            v = self._speed_at(t)
            st["p1"] = [t, q, pk[3], v] if v is not None and BUMP_MIN_SPEED_MS <= v <= BUMP_MAX_SPEED_MS else None
            return
        p1 = st["p1"]
        if p1 is None:
            return
        dt_axles = t - p1[0]
        if dt_axles < BUMP_WHEELBASE_M[0] / p1[3]:
            return  # the front axle coming down the bump; the rear one is still to come
        if not 1 / 3 <= -q / p1[1] <= 3:
            return
        st["p1"] = None
        if self.last_bump_t is not None and p1[0] - self.last_bump_t < BUMP_REFRACTORY_S:
            return
        self.last_bump_t = p1[0]
        lo, hi = p1[0] - BUMP_SPAN_PAD_S, t + BUMP_SPAN_PAD_S
        self.held = [h for h in self.held if not lo <= h[0] <= hi]
        if self.open_event is not None and lo <= self.open_event[0] <= hi:
            self.open_event = None
        rate = (p1[1] - q) / 2
        out.events.append(Event(t=p1[0], type="speed_bump", magnitude=round(max(p1[2], pk[3]), 3),
                                speed=p1[3], confidence=round(min(1.0, rate / (2 * BUMP_PITCH_RATE)), 3)))

    @staticmethod
    def _pothole_event(h: list) -> Event:
        et, peak, spd = h
        return Event(t=et, type="pothole", magnitude=round(peak, 3), speed=spd,
                     confidence=round(min(1.0, peak / (2 * POTHOLE_THRESHOLD_MS2)), 3))

    def _pothole(self, t: float, b: float, out: FeedResult) -> None:
        if self.open_event is not None:
            et, peak, spd = self.open_event
            if t - et <= POTHOLE_PEAK_WINDOW_S:
                if abs(b) > peak:
                    self.open_event = [et, abs(b), spd]
                return
            self.held.append([et, peak, spd])  # reported after POTHOLE_HOLD_S unless a speed bump covers it
            self.open_event = None
        if abs(b) < POTHOLE_THRESHOLD_MS2:
            return
        if self.last_event_t is not None and t - self.last_event_t < POTHOLE_REFRACTORY_S:
            return
        speed = self._speed_at(t)
        if speed is None or speed < POTHOLE_MIN_SPEED_MS:
            return
        self.open_event = [t, abs(b), speed]
        self.last_event_t = t
