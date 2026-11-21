# Read API contract (dashboard ⇄ read API)

The dashboard reads everything from the **read API**, a small web service running as an Azure Container App
([02-infrastructure-plan.md](../decisions/02-infrastructure-plan.md) §5 "Read API (Container App)"). This file is
the contract between the two. The TypeScript source of truth is [`src/types/`](src/types); the mock
backend in [`src/mock/`](src/mock) implements every route below, so the dashboard can be developed and
demoed without Azure.

General rules:

- `GET` only, JSON responses (`application/json`). Map data is GeoJSON (RFC 7946, `[lon, lat]`). The
  dashboard rejects any other content type, and checks each response's envelope (e.g. `sessions` is an
  array) before using it, so a wrong API URL shows a clear error instead of breaking a page.
- Field names are camelCase. Timestamps are ISO-8601 UTC strings (`2026-11-21T09:12:03.512Z`) unless
  the field is documented as epoch milliseconds.
- Units: acceleration m/s² (gravity removed), speed m/s unless the name ends in `Kmh`, MSDV m/s^1.5.
- Missing values are `null`, never omitted.
- Errors: `4xx`/`5xx` with body `{"error": "human readable reason"}`. Unknown session/segment → `404`;
  invalid query parameter → `400`.
- CORS: only the dashboard's Static Web Apps origin (plan §5).
- **No privacy trim** (D40, replaces plan §11): tracks, events, segment observations and live positions are
  served whole, start and end included. `trimmedM` is always `0`. Only fixes less accurate than 50 m are left out.
- **Aggregation rules** (plan §10): only finalized trips count: replays never, and a live trip only once it is finalized
  (until then its exclusion reason is `live_only`); sessions with
  `quality_status = 'bad'` are excluded; hand-held / non-rigid mounts are excluded from roughness and
  potholes, and their sickness numbers are flagged experimental.

## Routes

| Route | Response type (`src/types/…`) | Notes |
|---|---|---|
| `GET /api/overview` | `Overview` (overview) | Headline counts + the aggregation fit (`method`, `beta`, `vehicleEffects`) |
| `GET /api/sessions?source=&vehicle=` | `SessionListResponse` (sessions) | Newest first. Optional filters `source` = `upload\|live\|replay`, `vehicle` = vehicle type |
| `GET /api/sessions/{id}` | `SessionDetail` (sessions) | Summary + data-quality panel + detection scores + sickness ratings |
| `GET /api/sessions/{id}/series` | `SessionSeries` (sessions) | Columnar `session_series`, about 5 Hz, `t` in epoch ms |
| `GET /api/sessions/{id}/track` | `SessionTrack` (sessions) | GeoJSON `Feature<LineString>`, 1 Hz, `properties.times` (epoch ms) and `speeds` (m/s) parallel to the coordinates |
| `GET /api/sessions/{id}/events` | `RoadEventCollection` (sessions) | GeoJSON points: potholes, speed bumps, rail joints |
| `GET /api/sessions/{id}/labels` | `SessionLabelsResponse` (sessions) | Rows of the app's `labels.csv` |
| `GET /api/segments?vehicle=all\|car\|bus\|tram` | `SegmentCollection` (segments) | One H3 hexagon polygon per segment with its `segment_agg` row for that vehicle filter |
| `GET /api/segments/{h3}` | `SegmentDetail` (segments) | Per-vehicle aggregates, sickness split by seat/posture, every pass |
| `GET /api/events?vehicle=all\|car\|bus\|tram&type=` | `RoadEventCollection` (sessions) | Detected events of aggregated sessions only; `vehicle` takes the same values as `/api/segments` (the dashboard always sends it), `type` is optional (`pothole\|speed_bump\|rail_joint`) |
| `GET /api/live` | `LiveResponse` (live) | `live_state` of devices the ingest function heard from in the last 60 s, with the last ~2 min of track, `receivedAt` (arrival of the newest processed batch), `sessionStatus` (`live`, or `processing`/`complete` for a trip that ended within the last 60 s) and the latest MISC rating. Since D40 a device is listed from its first position, and stays listed for up to 60 s after its trip ends. Polled every 2 s; `serverTime` lets the dashboard measure ages on the server's clock |
| `GET /api/quality` | `QualityListResponse` (sessions) | The data-quality panel for every session |

The ingest routes (`POST /api/ingest`, and `PUT /api/ingest/sessions/{id}` from spec §6.1) belong to the ingest
function, on a different host, and are not used by the dashboard. There is no health route.

## Session status and exclusion reasons (plan §8, §10)

| `status` | Meaning |
|---|---|
| `live` | Recording: batches are still arriving through the live path |
| `processing` | The trip ended (Stop received, or 15 min without batches); the finalize job hasn't finished |
| `complete` | Finalized: rebuilt from the raw archive (or the uploaded folder) and aggregated |

`incomplete: true` marks a trip closed without the app's Stop message; it is still finalized, with its
gaps marked. `source` is how the trip reached the cloud: `live` (streamed, the main path), `upload`
(laptop fallback) or `replay`.

`exclusionReasons` explain why a trip is (partly) kept out of the aggregated map: `replay`, `live_only`
(streamed, not finalized yet), `processing` (an upload not finalized yet), `bad_quality`,
`unknown_context` (session.json never arrived, spec §6.1), `not_rigid`, `handheld` (the last two exclude
it from roughness only).

## Live freshness (plan §5 "Dashboard")

For each live device the dashboard derives, on the server's clock:
- data age = now − `updatedAt` (sample time): the end-to-end delay to the screen (target < 10 s);
- silence = now − `receivedAt`: amber after 10 s, red after 30 s (a dead zone). A trip that ends leaves
  the view at once (above), so silence on a listed device always means the phone stopped sending;
- transport delay = `receivedAt` − `updatedAt`: above 10 s while batches are arriving, the phone is
  sending a backlog after a dead zone, shown as **catching up** rather than live.

If the dashboard itself can't reach the read API (a failed poll, or no answer for 3 poll intervals), it
says so and shows each card's last data time, instead of reporting the phones as silent.

## Where each field comes from (plan §8)

| Dashboard field | Table / computation |
|---|---|
| `SessionSummary.*` context | `sessions` (+ `vehicles`, `devices`), from `session.json` |
| `isMock` | `session.json` → `syntheticSensors` (mock sessions generated for the pipeline) |
| `includeInAgg`, `exclusionReasons` | `sessions.include_in_agg` and the rule that set it |
| `msdvZ`, `msiPct` | MSDV of the whole session (ISO 2631-1 Wf, vertical); `msiPct = msdvZ / 3` |
| `sickness.ratings[].msdvSoFar`, `sickness.spearman` | MSDV of the last `session_series` point at or before each MISC rating (null before the first one); Spearman rank correlation of those with the ratings (average ranks for ties; null below 3 ratings) |
| `quality.*` | per-stream rows/rates/gaps, GPS accuracy, invalid windows, `rejected/late/duplicate` live batches |
| `detection` | detections vs. manual taps within ±1.5 s (precision, recall) |
| `SessionSeries` | `session_series` (≈ 5 Hz vertical/lateral/longitudinal acceleration, speed, MSDV so far) |
| `SegmentAggregate` | `segment_agg` table, rebuilt by the aggregation step (segment × vehicle type, plus an `all` row) |
| `roughness` | `exp(mean(log r − vehicle effect − β·log(v / 10 m/s)))`: what a car would feel at 36 km/h |
| `roughnessPctl` | mean per-session percentile of raw roughness (fallback when the model can't be fitted) |
| `potholeHits / potholePasses / potholeConfirmed` | passes > 15 km/h with a detection; confirmed at ≥ k of n passes (per vehicle type) |
| `sicknessAwRms`, `msi1hPct` | `sqrt(Σ a_w²·dt / Σ dt)` over rigid-mount sessions; `msi1hPct = a_w·√3600 / 3` |
| `LiveDevice` | `live_state` + the last positions of the live session; `receivedAt` from `ingest_batches` (arrival time); `latestRating` from `labels` |

Implementation note learned while building the mock: measure a segment pass's **speed over the same
samples as its roughness RMS** (i.e. moving samples only). Mixing stationary samples into the speed
biases the fitted β low (0.54 instead of 0.60 on the mock data).

## Hosting (Azure Static Web Apps)

`public/staticwebapp.config.json` makes client-side routes work on refresh (fallback to `index.html`),
serves `config.json` with `Cache-Control: no-store` and outside the fallback, caches `assets/*` for a
year, and sends the security headers (CSP: scripts only from the site, API and tiles over HTTPS only,
no framing). `vite preview` serves the same headers, so the e2e tests run under that CSP.

## Runtime configuration

The deployed site reads `/config.json` at startup: `{"apiBase": "https://<read-api>.<env>.northeurope.azurecontainerapps.io"}`.
The `deploy-dashboard.yml` workflow writes this file from a Terraform output just before uploading;
the committed `public/config.json` stays empty. An empty `apiBase` (the
default in `public/config.json`) makes the dashboard use its built-in mock data. A missing, unreachable
(5 s timeout) or invalid `config.json` stops the dashboard with an error instead: a broken deploy must not
quietly show mock data.
