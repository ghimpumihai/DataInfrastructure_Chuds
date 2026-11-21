import copy
import gzip
import json
import os
import pathlib
import sys
import uuid

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FIXTURE = ROOT / "src" / "mobile-ios" / "tools" / "fixtures" / "ingest_batch_v1.json.gz"
MIGRATIONS = ROOT / "db" / "migrations"


@pytest.fixture(scope="session")
def _fixture_batch():
    return json.loads(gzip.decompress(FIXTURE.read_bytes()))


@pytest.fixture
def batch(_fixture_batch):
    return copy.deepcopy(_fixture_batch)


@pytest.fixture
def session_meta(batch):
    return {
        "sessionId": batch["sessionId"],
        "deviceId": batch["deviceId"],
        "startedAtUtc": "2025-11-21T09:15:30.000Z",
        "startUptimeSec": 98.0,
        "endedAtUtc": None,
        "recovered": False,
        "syntheticSensors": False,
        "audioEnabled": False,
        "vehicle": {"type": "bus", "makeModel": "Solaris Urbino 12", "line": "24", "notes": ""},
        "mount": {"location": "dash", "phoneOrientation": "flat", "rigid": True, "handheld": False},
        "people": {"operatorSeat": "front", "raterSeat": "rear left", "raterPosture": "seated",
                   "passengerCount": 2},
        "conditions": {"weather": "dry", "notes": ""},
    }


def motion_batch(template: dict, session_id: str, seq: int, t0_epoch: float, vert, *, speed=12.0,
                 lat0=46.77, step_deg=0.0002) -> dict:
    """A 2-second batch of 100 Hz deviceMotion (phone screen up) with vertical acceleration vert(t)
    in m/s² (t = epoch seconds) and one GPS fix moving north."""
    import numpy as np

    b = copy.deepcopy(template)
    rel = np.arange(200) / 100.0
    up_g = vert(t0_epoch + rel) / 9.80665
    zeros = [0.0] * 200
    dm = {"t": rel.tolist(), "ua_x": zeros, "ua_y": zeros, "ua_z": (-up_g).tolist(),  # CoreMotion's sign
          "g_x": zeros, "g_y": zeros, "g_z": [-1.0] * 200,
          "rr_x": zeros, "rr_y": zeros, "rr_z": zeros,
          "q_w": [1.0] * 200, "q_x": zeros, "q_y": zeros, "q_z": zeros}
    t0_ms = int(round(t0_epoch * 1000))
    # Uptime advances with the batches, as on the phone (2 s windows since an arbitrary boot time).
    b.update(sessionId=session_id, batchSeq=seq, t0UtcMs=t0_ms, t0Up=5000.0 + 2 * seq, labels=[],
             deviceMotion=dm)
    b["location"] = [dict(template["location"][0], tUtcMs=t0_ms + 300, lat=lat0 + seq * step_deg,
                          speed=speed)]
    return b


# --- Integration tests: a real PostGIS (CI service container or docker-compose) ------------

def _have_db() -> bool:
    if not os.environ.get("PGHOST") or not os.environ.get("PGPASSWORD"):
        return False
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


requires_db = pytest.mark.skipif(not _have_db(), reason="needs PGHOST/PGPASSWORD and psycopg")


@pytest.fixture
def db():
    """A fresh, migrated database per test; dropped afterwards."""
    import psycopg

    from roadride.migrate import apply_migrations

    name = f"rr_test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(dbname="postgres", autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    try:
        with psycopg.connect(dbname=name, autocommit=True) as conn:
            apply_migrations(conn, MIGRATIONS)
        conn = psycopg.connect(dbname=name)
        yield conn
        conn.close()
    finally:
        with psycopg.connect(dbname="postgres", autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
