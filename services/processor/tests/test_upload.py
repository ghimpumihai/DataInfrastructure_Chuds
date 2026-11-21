"""Finalizing a laptop upload (plan §7 fallback): a session folder in the app's on-disk format goes
through tools/upload (dry run = a local `sessions` container), then the finalize job reads it.

The folder must give the same numbers as the same trip streamed: tools/replay rebuilds the live
batches from the folder, and both go through finalize.run_pipeline."""

import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import sys

import numpy as np
import pytest

from conftest import ROOT, requires_db

from roadride import finalize as fz
from roadride import upload as up
from roadride.live import parse_batch, parse_session

HEADERS = json.loads((ROOT / "src/mobile-ios/tools/fixtures/csv_headers.json").read_text())
SID = "4f1c2b9e-1111-4222-8333-444455556666"
START_UP = 5000.0
UTC0 = 1_795_000_000_000  # ms at START_UP
SECONDS = 64
POTHOLE_S = 21.0


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"tool_{name}", ROOT / "tools" / name / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"tool_{name}"] = mod  # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


uploader, replay = _tool("upload"), _tool("replay")


def _row(name, t_up, values):
    cols = HEADERS[name].split(",")
    # Every stream's UTC comes from the same clock anchors (spec §5), which drift a little.
    skew_ms = 0.4 * math.sin((t_up - START_UP) / 5)
    vals = {"t_up": f"{t_up:.6f}", "t_utc_ms": str(round(UTC0 + (t_up - START_UP) * 1000 + skew_ms)), **values}
    return ",".join(str(vals.get(c, "")) for c in cols)


def make_folder(root, *, sid=SID, seconds=SECONDS, ended=True):
    """A bus trip north at 12 m/s, phone screen up: 0.2 Hz sway, noise, a pothole at 21 s, a tap for
    it, a MISC rating with a quoted note, and a torn last line as after a crash."""
    d = root / "2025-11-21T09-15-32Z_4f1c"
    d.mkdir(parents=True)
    rng = np.random.default_rng(11)
    n = seconds * 100
    t = START_UP + np.arange(n) / 100 + rng.uniform(-0.002, 0.002, n)
    rel = t - START_UP
    vert = 0.8 * np.sin(2 * np.pi * 0.2 * rel) + rng.normal(0, 0.3, n) + 15 * np.exp(-((rel - POTHOLE_S) / 0.01) ** 2)
    rows = {k: [] for k in HEADERS}
    for i in range(n):
        rows["device_motion"].append(_row("device_motion", t[i], {
            "ua_x": "0.000000", "ua_y": "0.000000", "ua_z": f"{-vert[i] / 9.80665:.6f}",
            "g_x": "0.000000", "g_y": "0.000000", "g_z": "-1.000000",
            "rr_x": "0.000000", "rr_y": "0.000000", "rr_z": "" if i == 77 else "0.000000",
            "q_w": "1.000000", "q_x": "0.000000", "q_y": "0.000000", "q_z": "0.000000",
            "roll": "0", "pitch": "0", "yaw": "0"}))
        rows["accel_raw"].append(_row("accel_raw", t[i], {"x": "0", "y": "0", "z": "-1"}))
        rows["gyro_raw"].append(_row("gyro_raw", t[i], {"x": "0", "y": "0", "z": "0"}))
    for s in range(seconds):
        tu = START_UP + s + 0.3
        rows["location"].append(_row("location", tu, {
            "lat": f"{46.77 + 12 * (s + 0.3) / 111_320:.8f}", "lon": "23.62000000", "alt": "350", "ell_alt": "390",
            "h_acc": "4.7", "v_acc": "3", "speed": "12.000000", "speed_acc": "0.5", "course": "0.0",
            "course_acc": "5", "is_simulated": "0"}))
        rows["altimeter"].append(_row("altimeter", tu, {"rel_alt_m": "0.1", "pressure_kpa": "97.1"}))
    rows["audio_level"] = [_row("audio_level", START_UP + i / 10, {"avg_dbfs": "-40", "peak_dbfs": "-30"})
                           for i in range(seconds * 10)]
    rows["labels"] = [_row("labels", START_UP + POTHOLE_S + 0.8, {"type": "pothole"}),
                      _row("labels", START_UP + 30.0, {"type": "sickness", "value": "3", "note": '"a, b"'}),
                      _row("labels", START_UP + 50.0, {"type": "phone_moved"})]
    for name, lines in rows.items():
        (d / f"{name}.csv").write_text("\n".join([HEADERS[name], *lines]) + "\n")
    with open(d / "device_motion.csv", "a") as fh:
        fh.write(f"{START_UP + seconds:.6f},123,0.1")  # torn
    fixture = ROOT / "src/mobile-ios/tools/fixtures/session_meta_stop_v1.json.gz"
    meta = json.loads(gzip.decompress(fixture.read_bytes()))
    meta.pop("batchCount", None)
    meta.update(sessionId=sid, deviceId="PHONE-1", startUptimeSec=START_UP - 0.4, recovered=False,
                startedAtUtc="2026-11-18T07:59:59.600Z",
                endedAtUtc="2026-11-18T08:01:04.000Z" if ended else None,
                vehicle={"type": "bus", "makeModel": "Solaris", "line": "24", "notes": ""},
                mount={"location": "dash", "phoneOrientation": "flat", "rigid": True, "handheld": False},
                people={"operatorSeat": "front", "raterSeat": "rear left", "raterPosture": "seated",
                        "passengerCount": 2, "raterName": "Someone"})
    (d / "session.json").write_text(json.dumps(meta))
    (d / "audio_000.m4a").write_bytes(b"\0" * 64)
    return d


@pytest.fixture
def folder(tmp_path):
    return make_folder(tmp_path / "phone")


@pytest.fixture
def uploaded(folder, tmp_path):
    """The `sessions` container after tools/upload."""
    root = tmp_path / "sessions"
    uploader.run(folder, client=None, dry_run=root, log=lambda m: None)
    return up.DirUploads(root)


def replay_batches(folder, *, source="live"):
    """The live batches of the same trip (what the phone sent), rebuilt by tools/replay with the
    original identity and times."""
    plan = replay.make_plan(folder, session_id=SID, device_id="PHONE-1", now_ms=0)
    plan.shift_ms = 0.0
    b = replay.BatchBuilder(folder, plan)
    try:
        out = list(b.batches())
    finally:
        b.close()
    for x in out:
        x["source"] = source
    return out


def event_grid(sid=SID):
    return json.dumps({"id": "1", "eventType": "Microsoft.Storage.BlobCreated",
                       "subject": f"/blobServices/default/containers/sessions/blobs/{sid}/_complete.json",
                       "data": {"api": "PutBlob"}}).encode()


# --- Reading the folder ---------------------------------------------------------------------

def test_the_folder_gives_the_same_signal_as_the_streamed_trip(folder, uploaded):
    trip = up.load(uploaded, SID)
    live = [parse_batch(b) for b in replay_batches(folder)]
    assert [b.batch_seq for b in trip.batches] == [b.batch_seq for b in live]
    assert [b.sample_counts["deviceMotion"] for b in trip.batches] == [b.sample_counts["deviceMotion"] for b in live]

    a, b = fz.run_pipeline(trip.batches), fz.run_pipeline(live)
    assert len(a.series) == len(b.series) > 300
    assert [p.t for p in a.series] == pytest.approx([p.t for p in b.series], abs=1e-6)
    assert [p.a_vert for p in a.series] == pytest.approx([p.a_vert for p in b.series], abs=1e-3)
    assert a.msdv == pytest.approx(b.msdv, rel=1e-5) and a.resets == b.resets == 0
    assert [(e.type, round(e.t, 2)) for e in a.events] == [(e.type, round(e.t, 2)) for e in b.events]
    assert len(a.events) == 1 and abs(a.events[0].t - (UTC0 / 1000 + POTHOLE_S)) < 0.05

    fixes = [x for f in trip.batches for x in f.fixes]
    assert len(fixes) == SECONDS and fixes[0].speed_ms == 12.0 and fixes[0].h_acc_m == 4.7
    labels = [x for f in trip.batches for x in f.labels]
    assert [(lab.type, lab.value, lab.note) for lab in labels] == \
        [("pothole", None, None), ("sickness", "3", "a, b"), ("phone_moved", None, None)]
    assert "raterName" not in trip.meta["people"] and not trip.incomplete
    assert trip.streams["accelRaw"]["rows"] == SECONDS * 100 and trip.streams["audioLevel"]["expectedHz"] == 10.0


def test_unparseable_value_falls_back_to_the_forgiving_reader(folder):
    data = (folder / "device_motion.csv").read_bytes()
    fast = up.read_columns(data, up.MOTION_COLS)
    broken = data.replace(b",-1.000000,", b",abc,", 1)
    slow = up.read_columns(broken, up.MOTION_COLS)
    assert len(fast["t_up"]) == len(slow["t_up"]) == SECONDS * 100  # the torn line is dropped
    assert math.isnan(slow["g_z"][0]) and np.array_equal(fast["g_z"][1:], slow["g_z"][1:])
    assert np.isnan(fast["rr_z"][77])  # an empty field


def _tamper(uploaded, name, data):
    (uploaded.root / SID / name).write_bytes(data)


@pytest.mark.parametrize("change,reason", [
    (lambda u: (u.root / SID / "location.csv").unlink(), "location.csv is missing"),
    (lambda u: _tamper(u, "labels.csv", b"t_up\n"), "labels.csv has"),
    (lambda u: _tamper(u, "labels.csv", bytes(reversed((u.root / SID / "labels.csv").read_bytes()))), "changed"),
    (lambda u: (u.root / SID / "_complete.json").unlink(), "no _complete.json"),
    (lambda u: _tamper(u, "_complete.json", b"{"), "not JSON"),
])
def test_a_folder_that_doesnt_match_its_manifest_is_refused(uploaded, change, reason):
    change(uploaded)
    with pytest.raises(up.UploadInvalid, match=reason):
        up.load(uploaded, SID)


def test_session_json_of_another_trip_is_refused(uploaded):
    path = uploaded.root / SID / "session.json"
    meta = json.loads(path.read_text())
    meta["sessionId"] = "someone-else"
    data = json.dumps(meta).encode()
    path.write_bytes(data)
    m = json.loads((uploaded.root / SID / up.MANIFEST).read_text())
    m["files"]["session.json"] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (uploaded.root / SID / up.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(up.UploadInvalid, match="another session"):
        up.load(uploaded, SID)


def test_crashed_recording_is_incomplete(tmp_path):
    folder = make_folder(tmp_path / "phone", ended=False)
    uploader.run(folder, client=None, dry_run=tmp_path / "s", log=lambda m: None)
    assert up.load(up.DirUploads(tmp_path / "s"), SID).incomplete


# --- Finalize (a real PostGIS) --------------------------------------------------------------

def _one(db, sql, *args):
    return db.execute(sql, args).fetchone()


def _write_archive(root, batches, metas=()):
    folder = root / "raw" / f"session={SID}"
    folder.mkdir(parents=True, exist_ok=True)
    for b in batches:
        (folder / f"{b['batchSeq']:08d}.json.gz").write_bytes(gzip.compress(json.dumps(b).encode()))
    for m in metas:
        phase = "stop" if "batchCount" in m else "start"
        (folder / f"session-{phase}.json.gz").write_bytes(gzip.compress(json.dumps(m).encode()))


@requires_db
def test_upload_only_trip_is_finalized_and_aggregated(db, uploaded, tmp_path):
    import pyarrow.parquet as pq

    lake = tmp_path / "lake"
    assert fz.handle(db, fz.DirArchive(lake), event_grid(), uploaded) == ("complete", SID)

    row = _one(db, "SELECT source, status, incomplete, context_known, vehicle_type, include_in_agg, "
                   "exclusion_reasons, quality_status, people FROM sessions")
    assert row[:8] == ("upload", "complete", False, True, "bus", True, [], "degraded")  # 10 s phone_moved
    assert "raterName" not in row[8]
    q = _one(db, "SELECT quality FROM sessions")[0]
    assert q["finalizedFrom"] == "upload" and q["missingBatches"] is None
    assert {"deviceMotion", "accelRaw", "gyroRaw", "altimeter", "audioLevel", "location"} <= set(q["streams"])
    assert q["detection"]["pothole"]["truePositives"] == 1
    assert q["invalidSeconds"]["phone_moved"] == pytest.approx(10, abs=0.4)
    assert _one(db, "SELECT count(*) FROM track_points")[0] == SECONDS
    assert _one(db, "SELECT count(*) FROM labels")[0] == 3
    assert db.execute("SELECT type, matched_label FROM events").fetchall() == [("pothole", True)]
    assert _one(db, "SELECT count(*) FROM segment_observations")[0] > 10
    assert _one(db, "SELECT count(*) FROM segment_agg WHERE vehicle_type = 'bus'")[0] > 0

    motion = pq.read_table(lake / f"processed/session={SID}/motion.parquet")
    assert motion.num_rows == SECONDS * 100 and "rr_z" in motion.column_names
    assert str(motion.schema.field("t_utc").type) == "timestamp[us]"
    assert pq.read_table(lake / f"processed/session={SID}/location.parquet").num_rows == SECONDS
    series = pq.read_table(lake / f"processed/session={SID}/series.parquet")
    assert series.num_rows == _one(db, "SELECT count(*) FROM session_series")[0]
    assert q["parquet"] == [f"processed/session={SID}/{n}.parquet" for n in ("motion", "location", "series")]

    # A repeated message (or a second upload) rewrites the same rows.
    counts = _one(db, "SELECT (SELECT count(*) FROM session_series), (SELECT count(*) FROM labels), "
                      "(SELECT count(*) FROM track_points), (SELECT count(*) FROM segment_observations)")
    assert fz.handle(db, fz.DirArchive(lake), event_grid(), uploaded)[0] == "complete"
    assert _one(db, "SELECT (SELECT count(*) FROM session_series), (SELECT count(*) FROM labels), "
                    "(SELECT count(*) FROM track_points), (SELECT count(*) FROM segment_observations)") == counts


@requires_db
def test_uploaded_folder_wins_over_a_partly_streamed_trip(db, folder, uploaded, tmp_path):
    from roadride.store import Store

    batches = replay_batches(folder)
    meta = json.loads((folder / "session.json").read_text())
    start = dict(meta, source="live")
    store = Store(db)
    got = dt.datetime.fromtimestamp(UTC0 / 1000, dt.timezone.utc)
    with db.transaction():  # the phone lost its connection after 10 batches, then died
        store.apply_session(parse_session(start))
        for b in batches[:10]:
            store.apply_batch(parse_batch(b), got)
    live_labels = _one(db, "SELECT count(*) FROM labels")[0]
    assert live_labels == 0 and _one(db, "SELECT status FROM sessions")[0] == "live"
    lake = tmp_path / "lake"
    _write_archive(lake, batches[:10], [start])

    assert fz.handle(db, fz.DirArchive(lake), event_grid(), uploaded) == ("complete", SID)
    full = _one(db, "SELECT count(*) FROM session_series")[0]
    assert full > 300
    source, status, q = _one(db, "SELECT source, status, quality FROM sessions")
    assert (source, status, q["finalizedFrom"]) == ("live", "complete", "upload")
    assert _one(db, "SELECT count(*) FROM labels")[0] == 3

    # The phone comes back and sends its backlog: the trip reopens, and its live message is
    # finalized from the upload again, not from the shorter live archive.
    with db.transaction():
        assert store.apply_batch(parse_batch(batches[10]), got) == "new"
    assert _one(db, "SELECT status FROM sessions")[0] == "processing"
    _write_archive(lake, batches[10:11])
    assert fz.handle(db, fz.DirArchive(lake), json.dumps({"sessionId": SID}).encode(), uploaded) == ("complete", SID)
    assert _one(db, "SELECT count(*) FROM session_series")[0] == full
    assert _one(db, "SELECT quality->>'finalizedFrom' FROM sessions")[0] == "upload"


@requires_db
def test_broken_upload_is_dead_lettered_and_the_live_archive_still_works(db, folder, uploaded, tmp_path):
    batches = replay_batches(folder)
    lake = tmp_path / "lake"
    meta = json.loads((folder / "session.json").read_text())
    _write_archive(lake, batches, [dict(meta, source="live"), dict(meta, source="live", batchCount=len(batches))])
    (uploaded.root / SID / "location.csv").write_bytes(b"t_up,t_utc_ms\n")

    outcome, reason = fz.handle(db, fz.DirArchive(lake), event_grid(), uploaded)
    assert outcome == "dead_letter" and "location.csv has" in reason
    assert _one(db, "SELECT count(*) FROM sessions")[0] == 0  # nothing written

    assert fz.handle(db, fz.DirArchive(lake), json.dumps({"sessionId": SID}).encode(), uploaded) == ("complete", SID)
    q = _one(db, "SELECT quality FROM sessions")[0]
    assert q["finalizedFrom"] == "archive" and q["missingBatches"] == 0


@requires_db
def test_upload_message_without_a_folder_or_container(db, tmp_path):
    lake = fz.DirArchive(tmp_path / "lake")
    assert fz.handle(db, lake, event_grid(), None)[0] == "dead_letter"
    outcome, reason = fz.handle(db, lake, event_grid("nobody"), up.DirUploads(tmp_path / "empty"))
    assert outcome == "dead_letter" and "no _complete.json" in reason


@requires_db
def test_a_failed_parquet_write_does_not_block_the_trip(db, uploaded, tmp_path):
    class ReadOnlyLake(fz.DirArchive):
        def write(self, name, data):
            raise PermissionError("This request is not authorized to perform this operation")

    assert fz.handle(db, ReadOnlyLake(tmp_path / "lake"), event_grid(), uploaded) == ("complete", SID)
    status, q = _one(db, "SELECT status, quality FROM sessions")
    assert status == "complete" and q["parquet"] == [] and "not authorized" in q["parquetError"]
