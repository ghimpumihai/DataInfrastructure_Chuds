"""SQL and row → JSON for every route of dashboard/API.md.

Aggregates (segments, segment_agg, segment_sickness, aggregation_fit) are written by the finalize and
aggregate jobs; until a trip is finalized those routes answer with valid, empty shapes. Field names,
defaults and rules follow the dashboard's types (src/types/) and its mock backend (src/mock/), which is
the working spec.

Every list route runs a fixed number of queries, whatever the number of sessions: the per-session
numbers (distance, GPS accuracy, ratings, public event counts, stream rates) are computed for all
sessions at once with GROUP BY / LATERAL joins. Positions are returned untrimmed (D40); only fixes
less accurate than MAX_HACC_M are left out.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING

from .fmt import epoch_ms, iso, rnd

if TYPE_CHECKING:  # only for type hints; keeps the pure helpers importable without a driver
    from psycopg import Connection

MAX_HACC_M = 50.0  # fixes less accurate than this (or without an accuracy) are never returned
TRIP_SILENCE_CLOSE_S = 15 * 60  # API.md: a trip has ended after 15 min without batches
LIVE_WINDOW_S = 60  # devices heard from in the last minute
LIVE_TRACK_S = 120  # last ~2 minutes of track on the live view
REF_SPEED_KMH = 36.0

SOURCES = ("upload", "live", "replay")
VEHICLE_TYPES = ("car", "bus", "tram", "bike", "boat", "unknown")
VEHICLE_FILTERS = ("all", "car", "bus", "tram")
EVENT_TYPES = ("pothole", "speed_bump", "rail_joint")

MOUNT_DEFAULTS = {"location": "", "phoneOrientation": "", "rigid": False, "handheld": False}
PEOPLE_DEFAULTS = {"operatorSeat": "", "raterSeat": "", "raterPosture": "seated", "passengerCount": 0}
SICKNESS_LABEL = "sickness"

# ingest_batches.sample_counts key → (dashboard stream name, expected rate in Hz, as in the mock).
STREAMS = {
    "deviceMotion": ("device_motion", 100.0),
    "accelRaw": ("accel_raw", 100.0),
    "gyroRaw": ("gyro_raw", 100.0),
    "altimeter": ("altimeter", 1.0),
    "audioLevel": ("audio_level", 10.0),
    "location": ("location", 1.0),
}
MOTION_STREAMS = ("deviceMotion", "accelRaw", "gyroRaw")  # always listed once a trip has batches


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


def enum_param(value: str | None, name: str, allowed: tuple[str, ...]) -> str | None:
    if value is None or value == "":
        return None
    if value not in allowed:
        raise BadRequest(f"{name} must be one of {', '.join(allowed)}")
    return value


def _params(**extra) -> dict:
    """Named parameters shared by the fragments below, plus the query's own."""
    return {"max_hacc": MAX_HACC_M, "silence_s": TRIP_SILENCE_CLOSE_S,
            "sickness": SICKNESS_LABEL, **extra}


# A trip has ended when any sign says so, even if its status is still 'live': status not 'live', an
# end time, a Stop message (expected_batch_count), or 15 min without batches.
ENDED_SQL = """(s.status <> 'live' OR s.ended_at IS NOT NULL OR s.expected_batch_count IS NOT NULL
        OR s.last_received_at < now() - make_interval(secs => %(silence_s)s))"""


# Positions are returned whole: there is no start/end privacy trim (D40). Only inaccurate fixes are left out.
def _accurate(p: str) -> str:
    return f"{p}.h_acc_m <= %(max_hacc)s"


# --- Sessions ----------------------------------------------------------------------------

SESSION_FILTERS = {
    "one": "s.session_id = %(sid)s",
    "list": "(%(source)s::text IS NULL OR s.source = %(source)s)"
            " AND (%(vehicle)s::text IS NULL OR s.vehicle_type = %(vehicle)s)",
    "all": "true",
}


def _summary_sql(where: str) -> str:
    """The sessions' own columns plus everything cheap: ratings, public event counts, last sample time."""
    return f"""
WITH s AS MATERIALIZED (
  SELECT s.session_id, s.source, s.replay_of, s.device_id, s.status, s.incomplete, s.quality_status,
         s.include_in_agg, s.exclusion_reasons, s.context_known, s.vehicle_type, s.vehicle_make_model,
         s.vehicle_line, s.mount, s.people, s.audio_enabled, s.synthetic_sensors,
         COALESCE(s.started_at, s.created_at) AS started_at, s.ended_at,
         s.late_batches, s.duplicate_batches, s.msdv_z, s.updated_at AS version_at,
         s.valid_fraction, s.quality AS stored_quality, s.finalized_at, s.distance_m AS stored_distance_m,
         {ENDED_SQL} AS ended
  FROM sessions s WHERE {where}
),
rt AS (
  SELECT l.session_id, count(*) AS ratings FROM labels l
  WHERE l.type = %(sickness)s AND l.session_id IN (SELECT session_id FROM s)
  GROUP BY l.session_id
),
ev AS (
  SELECT e.session_id,
         count(*) FILTER (WHERE e.type = 'pothole') AS pothole,
         count(*) FILTER (WHERE e.type = 'speed_bump') AS speed_bump,
         count(*) FILTER (WHERE e.type = 'rail_joint') AS rail_joint
  FROM events e WHERE e.session_id IN (SELECT session_id FROM s) AND e.geom IS NOT NULL
  GROUP BY e.session_id
)
SELECT s.*,
       (SELECT b.end_utc FROM ingest_batches b WHERE b.session_id = s.session_id
        ORDER BY b.batch_seq DESC LIMIT 1) AS last_sample_at,
       COALESCE(rt.ratings, 0) AS ratings,
       COALESCE(ev.pothole, 0) AS ev_pothole, COALESCE(ev.speed_bump, 0) AS ev_speed_bump,
       COALESCE(ev.rail_joint, 0) AS ev_rail_joint
FROM s LEFT JOIN rt USING (session_id) LEFT JOIN ev USING (session_id)
ORDER BY s.started_at DESC, s.session_id
"""


def _rows(conn: Connection, sql: str, args: dict) -> list[dict]:
    cur = conn.execute(sql, args)
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# Whole-track numbers: the only part of a session summary that reads every fix of a trip.
TRACK_STATS_SQL = """
SELECT tp.session_id,
       ST_Length(ST_MakeLine(tp.geom::geometry ORDER BY tp.t)::geography),
       count(*),
       percentile_cont(0.5) WITHIN GROUP (ORDER BY tp.h_acc_m),
       percentile_cont(0.95) WITHIN GROUP (ORDER BY tp.h_acc_m)
FROM track_points tp WHERE tp.session_id = ANY(%(ids)s)
GROUP BY tp.session_id
"""


def _streams_sql() -> str:
    """One pass over the trips' batches in batch order (the primary key), one set of columns per stream.

    For each stream: its sample count per batch, and the pause before each batch that carries it,
    i.e. since the end of the latest earlier batch that carried it (a window max with FILTER).
    """
    per_batch, pauses, out = [], [], []
    for i, key in enumerate(STREAMS):
        # The processor writes these counts as len() of each block, so they are always integers.
        per_batch.append(f"(b.sample_counts ->> '{key}')::int AS n{i}")
        pauses.append(f"extract(epoch FROM b.t0_utc - max(b.end_utc) FILTER (WHERE b.n{i} > 0) OVER w)::float8"
                      f" AS p{i}")
        out.append(f"sum(n{i}) FILTER (WHERE n{i} > 0), count(*) FILTER (WHERE n{i} > 0),"
                   f" max(p{i}) FILTER (WHERE n{i} > 0), count(*) FILTER (WHERE n{i} > 0 AND p{i} > 1)")
    return f"""
WITH b AS (
  SELECT b.session_id, b.batch_seq, b.t0_utc, b.end_utc, {", ".join(per_batch)}
  FROM ingest_batches b WHERE b.session_id = ANY(%(ids)s)
),
p AS (
  SELECT b.*, {", ".join(pauses)}
  FROM b WINDOW w AS (PARTITION BY b.session_id ORDER BY b.batch_seq
                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT session_id, extract(epoch FROM max(end_utc) - min(t0_utc))::float8, {", ".join(out)}
FROM p GROUP BY session_id
"""


STREAMS_SQL = _streams_sql()


def _track_stats(conn: Connection, ids: list[str]) -> dict[str, dict]:
    out = {sid: {"distance_m": 0.0, "fixes": 0, "hacc_median": None, "hacc_p95": None} for sid in ids}
    for sid, distance, fixes, median, p95 in conn.execute(TRACK_STATS_SQL, {"ids": ids}).fetchall():
        out[sid] = {"distance_m": float(distance or 0), "fixes": int(fixes), "hacc_median": median,
                    "hacc_p95": p95}
    return out


def _stream_quality(conn: Connection, ids: list[str]) -> dict[str, dict]:
    """Per-stream rows, rate and gaps of each trip, from its live batches (ingest_batches.sample_counts).

    The live path only knows each batch's sample count and time span, so gaps are seen at batch
    resolution: maxGapS is the longest pause between two batches that carry the stream (never less
    than one sample period), and a pause inside a batch isn't visible. Uploaded trips have no live
    batches, so their streams stay empty until the finalize job computes them from the files.
    """
    out: dict[str, dict] = {sid: {} for sid in ids}
    for sid, span_s, *cols in conn.execute(STREAMS_SQL, {"ids": ids}).fetchall():
        streams = out[sid]
        for i, (name, hz) in enumerate(STREAMS.values()):
            n, batches, max_pause, pauses = cols[4 * i: 4 * i + 4]
            if not batches:
                continue
            streams[name] = {
                "rows": int(n),
                "rateHz": None if not span_s or span_s <= 0 else round(float(n) / (span_s + 1 / hz), 2),
                "expectedHz": hz,
                "maxGapS": None if batches < 2 else round(max(max_pause, 1 / hz), 3),
                "gapsOver1s": int(pauses),
            }
        for key in MOTION_STREAMS:  # batches without motion samples are a problem, not "no data"
            name, hz = STREAMS[key]
            streams.setdefault(name, {"rows": 0, "rateHz": 0.0, "expectedHz": hz, "maxGapS": None,
                                      "gapsOver1s": 0})
    return out


class TripStatsCache:
    """Remembers each trip's whole-track and per-stream numbers until the trip changes.

    /api/sessions is polled every few seconds while any trip is unfinalized, and its track length reads
    every fix of every trip. A trip's data only changes together with its `sessions` row: the processor
    sets updated_at = now() in the same transaction as every batch and session message, and finalizing
    changes the status. So (updated_at, status) is the version, and only trips that changed since the
    last request are recomputed. One cache per app (i.e. per database); thread-safe enough for the
    endpoint thread pool, since a race only recomputes a value twice.
    """

    def __init__(self, max_entries: int = 20_000) -> None:
        self._entries: dict[tuple[str, str], tuple[tuple, dict]] = {}
        self._max_entries = max_entries

    def get_many(self, kind: str, rows: list[dict], compute) -> dict[str, dict]:
        out, missing = {}, []
        for r in rows:
            hit = self._entries.get((kind, r["session_id"]))
            if hit is not None and hit[0] == _version(r):
                out[r["session_id"]] = hit[1]
            else:
                missing.append(r)
        if missing:
            fresh = compute([r["session_id"] for r in missing])
            if len(self._entries) + len(missing) > self._max_entries:
                self._entries.clear()
            for r in missing:
                self._entries[(kind, r["session_id"])] = (_version(r), fresh[r["session_id"]])
                out[r["session_id"]] = fresh[r["session_id"]]
        return out


def _version(r: dict) -> tuple:
    return r["version_at"], r["status"]


def _stats(conn: Connection, cache: TripStatsCache | None, kind: str, rows: list[dict]) -> dict[str, dict]:
    compute = {"track": lambda ids: _track_stats(conn, ids),
               "streams": lambda ids: _stream_quality(conn, ids)}[kind]
    if not rows:
        return {}
    if cache is None:
        return compute([r["session_id"] for r in rows])
    return cache.get_many(kind, rows, compute)


def _summaries(conn: Connection, cache: TripStatsCache | None, scope: str, with_gps: bool,
               **args) -> list[dict]:
    """Session rows with their track numbers merged in: 1 query, plus 1 for trips that need them.

    A finalized trip has its track length in sessions.distance_m; only the others (and the GPS
    accuracy of the quality panel) are computed from track_points, and cached.
    """
    rows = _rows(conn, _summary_sql(SESSION_FILTERS[scope]), _params(**args))
    need = rows if with_gps else [r for r in rows if r["stored_distance_m"] is None]
    track = _stats(conn, cache, "track", need)
    out = []
    for r in rows:
        stats = track.get(r["session_id"], {})
        distance = r["stored_distance_m"] if r["stored_distance_m"] is not None else stats["distance_m"]
        out.append({**r, **stats, "distance_m": distance})
    return out


def _stored_quality(r: dict) -> dict:
    return r["stored_quality"] if isinstance(r["stored_quality"], dict) else {}  # written by the finalize job


def _session_streams(conn: Connection, cache: TripStatsCache | None, rows: list[dict]) -> dict[str, dict]:
    """The finalize job's per-stream quality (from the whole trip) when there is one, else the live
    batches'. The finalize job names streams by their batch block (deviceMotion); the dashboard by
    its file (device_motion), so stored names are mapped."""
    out = {}
    for r in rows:
        stored = _stored_quality(r).get("streams")
        if isinstance(stored, dict) and stored:
            out[r["session_id"]] = {STREAMS.get(k, (k,))[0]: v for k, v in stored.items()}
    live = _stats(conn, cache, "streams", [r for r in rows if r["session_id"] not in out])
    return {**live, **out}


def _merge(defaults: dict, stored) -> dict:
    out = dict(defaults)
    if isinstance(stored, dict):
        out.update({k: stored[k] for k in defaults if k in stored and stored[k] is not None})
    return out


def _summary(r: dict) -> dict:
    mount = _merge(MOUNT_DEFAULTS, r["mount"])
    msdv = rnd(r["msdv_z"], 4)
    roughness_eligible = bool(r["context_known"]) and mount["rigid"] is True and mount["handheld"] is not True
    end = r["ended_at"] or r["last_sample_at"] or r["started_at"]
    return {
        "sessionId": r["session_id"],
        "source": r["source"],
        "isMock": bool(r["synthetic_sensors"]),
        "replayOf": r["replay_of"],
        "deviceId": r["device_id"],
        "vehicle": {"type": r["vehicle_type"] or "unknown", "makeModel": r["vehicle_make_model"] or "",
                    "line": r["vehicle_line"]},
        "mount": mount,
        "people": _merge(PEOPLE_DEFAULTS, r["people"]),
        "audioEnabled": bool(r["audio_enabled"]),
        "startedAt": iso(r["started_at"]),
        "endedAt": iso(r["ended_at"]),
        "durationS": max(0.0, round((end - r["started_at"]).total_seconds(), 3)),
        "distanceM": round(float(r["distance_m"]), 1),  # the whole track: a length reveals no position
        "status": r["status"],
        "incomplete": bool(r["incomplete"]),
        "qualityStatus": r["quality_status"],
        "validFraction": rnd(r["valid_fraction"], 4),  # set by the finalize job's quality checks
        "includeInAgg": bool(r["include_in_agg"]),
        "exclusionReasons": list(r["exclusion_reasons"] or []),  # set by the processor (plan §10)
        "roughnessEligible": roughness_eligible,
        "sicknessExperimental": not roughness_eligible,
        "msdvZ": msdv,  # ISO 2631-1 Wf, vertical, so far (live) or whole trip
        "msiPct": None if msdv is None else round(msdv / 3, 4),  # API.md: MSI % ≈ MSDV / 3
        "eventCounts": {t: int(r[f"ev_{t}"]) for t in EVENT_TYPES},
        "ratingsCount": int(r["ratings"]),
    }


def _quality(r: dict, streams: dict) -> dict:
    invalid = {"phone_moved": 0, "gravity_drift": 0, "gyro_gps_mismatch": 0, "gps_poor": 0}
    invalid.update({k: v for k, v in (_stored_quality(r).get("invalidSeconds") or {}).items() if k in invalid})
    return {
        "streams": streams,
        "gps": {"fixes": int(r["fixes"]), "medianHAccM": rnd(r["hacc_median"], 2),
                "p95HAccM": rnd(r["hacc_p95"], 2)},
        "validFraction": rnd(r["valid_fraction"], 4),
        "invalidSeconds": invalid,
        "rejectedBatches": 0,  # rejected batches never reach the database (the ingest function answers 4xx)
        "lateBatches": int(r["late_batches"]),
        "duplicateBatches": int(r["duplicate_batches"]),
    }


DETECTION_TYPES = ("pothole", "speed_bump")
DETECTION_COUNTS = ("truePositives", "falsePositives", "falseNegatives")


def _detection(r: dict) -> dict | None:
    """API.md DetectionMetrics, scored by the finalize job against the manual taps
    (`quality.detection`). None until a trip is finalized, or if what's stored isn't that shape."""
    stored = _stored_quality(r).get("detection")
    if not isinstance(stored, dict):
        return None
    out = {}
    for typ in DETECTION_TYPES:
        d = stored.get(typ)
        if not isinstance(d, dict) or not all(type(d.get(k)) is int for k in DETECTION_COUNTS):
            return None
        out[typ] = {**{k: d[k] for k in DETECTION_COUNTS},
                    **{k: rnd(v, 4) if isinstance(v := d.get(k), (int, float)) and not isinstance(v, bool) else None
                       for k in ("precision", "recall")}}
    return out


def rating_value(value) -> int | None:
    """A MISC rating's number, or None for a label without one (a skipped prompt)."""
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _ratings(conn: Connection, session_id: str) -> list[dict]:
    """Each MISC rating with the dose the passenger had had by then (plan §9 step 6): the last
    session_series point at or before it, None before the first one."""
    out = []
    for t, value, msdv in conn.execute(
            """SELECT l.t, l.value, s.msdv FROM labels l
               LEFT JOIN LATERAL (SELECT msdv FROM session_series
                                  WHERE session_id = l.session_id AND t <= l.t
                                  ORDER BY t DESC LIMIT 1) s ON true
               WHERE l.session_id = %s AND l.type = %s ORDER BY l.t""",
            (session_id, SICKNESS_LABEL)).fetchall():
        number = rating_value(value)
        if number is not None:
            out.append({"tUtc": iso(t), "value": number, "msdvSoFar": rnd(msdv, 5)})
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation with average ranks for ties, the mock's rule (src/mock/dataset.ts)."""
    if len(xs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda k: values[k])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    return num / (dx * dy) ** 0.5 if dx > 0 and dy > 0 else None


def _sickness(r: dict, ratings: list[dict]) -> dict:
    paired = [x for x in ratings if x["msdvSoFar"] is not None]
    rho = spearman([x["msdvSoFar"] for x in paired], [x["value"] for x in paired])
    return {"msdvZ": rnd(r["msdv_z"], 4),
            "msiPct": None if r["msdv_z"] is None else round(r["msdv_z"] / 3, 4),
            "ratings": ratings, "spearman": rnd(rho, 4)}


def _require_session(conn: Connection, session_id: str) -> None:
    if conn.execute("SELECT 1 FROM sessions WHERE session_id = %s", (session_id,)).fetchone() is None:
        raise NotFound(f"session {session_id} not found")


def session_list(conn: Connection, source: str | None, vehicle: str | None,
                 cache: TripStatsCache | None = None) -> dict:
    rows = _summaries(conn, cache, "list", False, source=source, vehicle=vehicle)
    return {"sessions": [_summary(r) for r in rows]}


def session_detail(conn: Connection, session_id: str, cache: TripStatsCache | None = None) -> dict:
    rows = _summaries(conn, cache, "one", True, sid=session_id)
    if not rows:
        raise NotFound(f"session {session_id} not found")
    r = rows[0]
    streams = _session_streams(conn, cache, rows)[session_id]
    return {
        **_summary(r),
        "quality": _quality(r, streams),
        "detection": _detection(r),
        "sickness": _sickness(r, _ratings(conn, session_id)),
        "processedAt": iso(r["finalized_at"]),
        "processorVersion": None,
    }


def quality_list(conn: Connection, cache: TripStatsCache | None = None) -> dict:
    rows = _summaries(conn, cache, "all", True)
    streams = _session_streams(conn, cache, rows)
    return {"sessions": [{**_summary(r), "quality": _quality(r, streams[r["session_id"]])} for r in rows]}


TRACK_SQL = f"""
SELECT ST_X(p.geom::geometry), ST_Y(p.geom::geometry), p.t, p.speed_ms
FROM track_points p WHERE p.session_id = %(sid)s AND {_accurate("p")}
ORDER BY p.t
"""


def session_track(conn: Connection, session_id: str) -> dict:
    _require_session(conn, session_id)
    pts = conn.execute(TRACK_SQL, _params(sid=session_id)).fetchall()
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[rnd(lon), rnd(lat)] for lon, lat, _, _ in pts]},
        "properties": {
            "sessionId": session_id,
            "times": [epoch_ms(t) for _, _, t, _ in pts],
            "speeds": [rnd(speed, 2) for *_, speed in pts],
            "trimmedM": 0,
        },
    }


def session_labels(conn: Connection, session_id: str) -> dict:
    _require_session(conn, session_id)
    rows = conn.execute("SELECT t, type, value, note FROM labels WHERE session_id = %s ORDER BY t",
                        (session_id,)).fetchall()
    return {"labels": [{"tUtc": iso(t), "type": ty, "value": v, "note": n} for t, ty, v, n in rows]}


def session_series(conn: Connection, session_id: str) -> dict:
    _require_session(conn, session_id)
    rows = conn.execute(
        "SELECT t, a_vert, a_lat, a_lon, speed_ms, msdv FROM session_series WHERE session_id = %s ORDER BY t",
        (session_id,)).fetchall()
    return {
        "t": [epoch_ms(t) for t, *_ in rows],
        "aVert": [rnd(r[1], 4) for r in rows],
        "aLat": [rnd(r[2], 4) for r in rows],
        "aLon": [rnd(r[3], 4) for r in rows],
        "speed": [rnd(r[4], 2) for r in rows],
        "msdv": [rnd(r[5], 5) for r in rows],
    }


EVENTS_SQL = """
SELECT e.session_id, e.type, e.t, ST_X(e.geom::geometry), ST_Y(e.geom::geometry), e.speed_ms,
       e.magnitude, e.confidence, e.matched_label, s.vehicle_type
FROM events e JOIN sessions s USING (session_id)
WHERE {where} AND e.geom IS NOT NULL AND (%(etype)s::text IS NULL OR e.type = %(etype)s)
ORDER BY e.t, e.session_id, e.type
"""


def _event_features(conn: Connection, where: str, **args) -> list[dict]:
    """Events of the matching sessions (those with a position) as GeoJSON features."""
    sql = EVENTS_SQL.format(where=where)
    features = []
    for sid, etype, t, lon, lat, speed, magnitude, confidence, matched, vtype in conn.execute(
            sql, _params(**{"etype": None, **args})).fetchall():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [rnd(lon), rnd(lat)]},
            "properties": {
                "id": f"{sid}:{etype}:{epoch_ms(t)}",
                "sessionId": sid,
                "type": etype,
                "tUtc": iso(t),
                "speedKmh": None if speed is None else round(speed * 3.6, 1),
                "magnitude": rnd(magnitude, 3),
                "confidence": rnd(confidence, 3),
                "matchedLabel": bool(matched),
                "vehicleType": vtype or "unknown",
            },
        })
    return features


def session_events(conn: Connection, session_id: str) -> dict:
    _require_session(conn, session_id)
    return {"type": "FeatureCollection",
            "features": _event_features(conn, "s.session_id = %(sid)s", sid=session_id)}


def all_events(conn: Connection, vehicle: str | None, etype: str | None) -> dict:
    """Events of aggregated trips only (API.md), optionally one vehicle type and one event type."""
    vt = None if vehicle in (None, "all") else vehicle
    where = "s.include_in_agg AND (%(vt)s::text IS NULL OR s.vehicle_type = %(vt)s)"
    return {"type": "FeatureCollection", "features": _event_features(conn, where, vt=vt, etype=etype)}


# --- Live ---------------------------------------------------------------------------------

LIVE_SQL = f"""
SELECT ls.device_id, ls.session_id, s.source, s.status, s.vehicle_type, ls.updated_at, ls.received_at,
       ls.lat, ls.lon, ls.speed_ms, ls.course_deg, ls.h_acc_m, ls.roughness_now, ls.msdv_so_far,
       ls.battery, ls.thermal, tr.lons, tr.lats, rt.ts, rt.vals
FROM live_state ls JOIN sessions s USING (session_id)
LEFT JOIN LATERAL (
  SELECT array_agg(ST_X(p.geom::geometry) ORDER BY p.t) AS lons,
         array_agg(ST_Y(p.geom::geometry) ORDER BY p.t) AS lats
  FROM track_points p
  WHERE p.session_id = ls.session_id AND p.t >= ls.updated_at - make_interval(secs => %(track_s)s)
    AND {_accurate("p")}) tr ON true
LEFT JOIN LATERAL (
  SELECT array_agg(l.t ORDER BY l.t DESC) AS ts, array_agg(l.value ORDER BY l.t DESC) AS vals
  FROM (SELECT t, value FROM labels WHERE session_id = ls.session_id AND type = %(sickness)s
        ORDER BY t DESC LIMIT 5) l) rt ON true
WHERE ls.received_at > %(since)s AND ls.lat IS NOT NULL AND ls.lon IS NOT NULL
ORDER BY ls.device_id
"""


def live(conn: Connection, now: dt.datetime) -> dict:
    """Devices heard from in the last minute, with the last ~2 min of their track, including a trip
    that has just ended (D40: no start/end privacy rule)."""
    rows = _rows(conn, LIVE_SQL, _params(since=now - dt.timedelta(seconds=LIVE_WINDOW_S), track_s=LIVE_TRACK_S))
    devices = []
    for r in rows:
        latest = None
        for t, value in zip(r["ts"] or [], r["vals"] or []):
            number = rating_value(value)
            if number is not None:
                latest = {"value": number, "tUtc": iso(t)}
                break
        devices.append({
            "deviceId": r["device_id"],
            "sessionId": r["session_id"],
            "source": r["source"],
            "sessionStatus": r["status"],
            "vehicleType": r["vehicle_type"] or "unknown",
            "updatedAt": iso(r["updated_at"]),
            "receivedAt": iso(r["received_at"]),
            "lat": rnd(r["lat"]),
            "lon": rnd(r["lon"]),
            "speedMs": rnd(r["speed_ms"], 2),
            "courseDeg": rnd(r["course_deg"], 1),
            "hAccM": rnd(r["h_acc_m"], 1),
            "roughnessNow": rnd(r["roughness_now"], 3),
            "msdvSoFar": rnd(r["msdv_so_far"], 3),
            "latestRating": latest,
            "battery": rnd(r["battery"], 2),
            "thermal": r["thermal"],
            "track": [[rnd(lon), rnd(lat)] for lon, lat in zip(r["lons"] or [], r["lats"] or [])],
        })
    return {"serverTime": iso(now), "devices": devices}


# --- Aggregates (written by the finalize and aggregate jobs) ------------------------------

def _aggregation_info(conn: Connection) -> dict:
    row = conn.execute("SELECT method, beta, vehicle_effects, fitted_at FROM aggregation_fit").fetchone()
    if row is None:
        return {"method": "none", "beta": None, "refSpeedKmh": REF_SPEED_KMH, "vehicleEffects": {},
                "fittedAt": None}
    method, beta, effects, fitted_at = row
    return {"method": method, "beta": rnd(beta, 4), "refSpeedKmh": REF_SPEED_KMH,
            "vehicleEffects": {k: rnd(v, 4) for k, v in (effects or {}).items()}, "fittedAt": iso(fitted_at)}


def overview(conn: Connection) -> dict:
    total, upload, live_n, replay, included, mock, distance_m, last_processed = conn.execute("""
        SELECT count(*), count(*) FILTER (WHERE source = 'upload'), count(*) FILTER (WHERE source = 'live'),
               count(*) FILTER (WHERE source = 'replay'), count(*) FILTER (WHERE include_in_agg),
               count(*) FILTER (WHERE synthetic_sensors),
               COALESCE(sum(distance_m) FILTER (WHERE include_in_agg), 0), max(finalized_at)
        FROM sessions
        """).fetchone()
    segments, confirmed = conn.execute(
        "SELECT count(*), count(*) FILTER (WHERE pothole_confirmed) FROM segment_agg WHERE vehicle_type = 'all'"
    ).fetchone()
    speed_bumps = conn.execute(
        "SELECT count(*) FROM events e JOIN sessions s USING (session_id) "
        "WHERE s.include_in_agg AND e.type = 'speed_bump'").fetchone()[0]
    return {
        "sessions": {"total": total, "upload": upload, "live": live_n, "replay": replay,
                     "includedInAgg": included, "mock": mock},
        "distanceKm": round(float(distance_m) / 1000, 1),  # finalized trips in the aggregates
        "segments": segments,
        "confirmedPotholes": confirmed,
        "speedBumps": speed_bumps,
        "lastProcessedAt": iso(last_processed),
        "aggregation": _aggregation_info(conn),
    }


AGG_COLUMNS = """a.h3, a.vehicle_type, a.n_sessions, a.n_passes, a.mean_speed_kmh, a.roughness, a.roughness_pctl,
       a.roughness_method, a.pothole_hits, a.pothole_passes, a.pothole_ratio, a.pothole_confirmed,
       a.sickness_aw_rms, a.msi_1h_pct, a.sickness_sessions"""


def _aggregate(r: dict) -> dict:
    """One segment_agg row as API.md SegmentAggregate."""
    return {
        "h3": r["h3"],
        "vehicleType": r["vehicle_type"],
        "nSessions": r["n_sessions"],
        "nPasses": r["n_passes"],
        "meanSpeedKmh": rnd(r["mean_speed_kmh"], 1),
        "roughness": rnd(r["roughness"], 4),
        "roughnessPctl": rnd(r["roughness_pctl"], 4),
        "roughnessMethod": r["roughness_method"],
        "potholeHits": r["pothole_hits"],
        "potholePasses": r["pothole_passes"],
        "potholeRatio": rnd(r["pothole_ratio"], 4),
        "potholeConfirmed": bool(r["pothole_confirmed"]),
        "sicknessAwRms": rnd(r["sickness_aw_rms"], 4),
        "msi1hPct": rnd(r["msi_1h_pct"], 4),
        "sicknessSessions": r["sickness_sessions"],
    }


def segments(conn: Connection, vehicle: str | None) -> dict:
    rows = _rows(conn, f"""
        SELECT {AGG_COLUMNS}, ST_AsGeoJSON(g.boundary::geometry, 6) AS boundary
        FROM segment_agg a JOIN segments g USING (h3)
        WHERE a.vehicle_type = %s ORDER BY a.h3
        """, (vehicle or "all",))
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": json.loads(r["boundary"]), "properties": _aggregate(r)} for r in rows]}


# Every pass is returned, including ones finalized before D40 with the old `in_privacy_zone` flag.
OBSERVATIONS_SQL = """
SELECT o.session_id, s.vehicle_type, o.t_start, o.mean_speed_ms, o.roughness_rms, o.aw_rms,
       o.pothole_count, s.include_in_agg
FROM segment_observations o JOIN sessions s USING (session_id)
WHERE o.h3 = %(h3)s
ORDER BY o.t_start
"""


def segment_detail(conn: Connection, h3: str) -> dict:
    if not is_h3_cell(h3):
        raise BadRequest(f"{h3} is not an H3 cell")
    center = conn.execute("SELECT ST_X(center::geometry), ST_Y(center::geometry) FROM segments WHERE h3 = %s",
                          (h3,)).fetchone()
    if center is None:
        raise NotFound(f"segment {h3} has no observations")
    order = "CASE a.vehicle_type WHEN 'all' THEN 0 WHEN 'car' THEN 1 WHEN 'bus' THEN 2 WHEN 'tram' THEN 3 ELSE 4 END"
    by_vehicle = [_aggregate(r) for r in _rows(
        conn, f"SELECT {AGG_COLUMNS} FROM segment_agg a WHERE a.h3 = %s ORDER BY {order}", (h3,))]
    sickness = [{
        "vehicleType": vt, "raterSeat": seat, "raterPosture": posture, "sicknessAwRms": rnd(aw, 4),
        "msi1hPct": rnd(msi, 4), "nSessions": n,
    } for vt, seat, posture, aw, msi, n in conn.execute(
        "SELECT vehicle_type, rater_seat, rater_posture, sickness_aw_rms, msi_1h_pct, n_sessions "
        "FROM segment_sickness WHERE h3 = %s ORDER BY vehicle_type, rater_seat, rater_posture", (h3,)).fetchall()]
    # Every pass outside the private zones; trips not (yet) in the aggregates are flagged.
    observations = [{
        "sessionId": sid, "vehicleType": vt or "unknown", "tStart": iso(t),
        "speedKmh": None if v is None else round(v * 3.6, 1),
        "roughnessRms": rnd(rough, 4), "awRms": rnd(aw, 4), "potholeCount": potholes, "includedInAgg": bool(inc),
    } for sid, vt, t, v, rough, aw, potholes, inc in conn.execute(OBSERVATIONS_SQL, _params(h3=h3)).fetchall()]
    fitted = conn.execute("SELECT fitted_at FROM aggregation_fit").fetchone()
    return {
        "h3": h3,
        "center": {"lon": rnd(center[0]), "lat": rnd(center[1])},
        "byVehicle": by_vehicle,
        "sicknessBySeat": sickness,
        "observations": observations,
        "updatedAt": iso(fitted[0]) if fitted else None,
    }


def is_h3_cell(value: str) -> bool:
    """H3 cell index: 15 hex digits, mode 1 (cell), so the first digit is 8."""
    if len(value) != 15 or value[0] != "8":
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
