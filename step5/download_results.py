# ─────────────────────────────────────────────────────────────
# STEP 5 – Download public matched-filter pipeline results
# ─────────────────────────────────────────────────────────────
"""
Download publicly available matched-filter pipeline results from GWOSC.

What is downloaded
------------------
* **GWTC event catalog** (JSON): metadata from all publicly released
  gravitational-wave events, including per-pipeline FAR, combined SNR,
  source masses, and luminosity distances.  Covers GWTC-2 (O3a) and
  GWTC-3 (O3b) by default.

* **Per-event parameter summary** (JSON): individual event pages on GWOSC
  provide the best-fit parameters and matched-filter statistics from each
  pipeline (gstlal, PyCBC, MBTA).  These are saved alongside the catalog.

These downloaded results can be used to:
    - Calibrate the SNR → FAR threshold for the Step 5 sensitivity estimate.
    - Compare the SparseBank pipeline sensitivity to the official search.
    - Validate the reference distance / SNR calibration.

GWOSC API endpoints used
------------------------
Catalog listing:
    https://gwosc.org/eventapi/json/query/?releases=<CATALOG>

Individual event:
    https://gwosc.org/eventapi/json/<EVENT_NAME>/

GWTC data-release Zenodo DOIs (for full trigger files)
------------------------------------------------------
GWTC-2 (O3a):  https://zenodo.org/records/5546662
GWTC-3 (O3b):  https://zenodo.org/records/8177023

Full trigger databases are large (tens of GB) and therefore not downloaded
automatically; see ``print_zenodo_instructions()`` for guidance.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger("step5.download")

GWOSC_EVENT_API = "https://gwosc.org/eventapi/json/query/"
GWOSC_EVENT_PAGE = "https://gwosc.org/eventapi/json/{event_name}/"

# Catalogs that contain matched-filter BNS/NSBH results
DEFAULT_CATALOGS = ["GWTC-2", "GWTC-3"]

# Known BNS / NSBH events (for targeted downloading)
BNS_EVENTS = [
    "GW170817",
    "GW190425",
    "GW200105_162426",
    "GW200115_042309",
]


# ──────────────────────────────────────────────────────────────────────────────
# Network helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_json(url: str, timeout: int = 30) -> dict | list:
    """Fetch JSON from *url* with basic retry logic (up to 3 attempts)."""
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            wait = 2 ** attempt
            log.warning("  Attempt %d failed (%s); retrying in %ds …", attempt + 1, exc, wait)
            if attempt < 2:
                time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts")


# ──────────────────────────────────────────────────────────────────────────────
# Catalog download
# ──────────────────────────────────────────────────────────────────────────────

def download_gwosc_catalog(
    catalog: str,
    output_dir: Path,
) -> Path:
    """
    Download the full event listing for *catalog* from GWOSC and save it
    as ``{output_dir}/{catalog}.json``.

    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{catalog}.json"

    if out_path.exists():
        log.info("[Step 5] Catalog already present: %s (skipping download)", out_path)
        return out_path

    url = f"{GWOSC_EVENT_API}?releases={catalog}"
    log.info("[Step 5] Downloading %s catalog from %s …", catalog, url)

    data = _get_json(url)
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2)

    n_events = len(data.get("events", data) if isinstance(data, dict) else data)
    log.info("[Step 5]  → saved %d events to %s", n_events, out_path)
    return out_path


def download_event_parameters(
    event_name: str,
    output_dir: Path,
) -> Path | None:
    """
    Download the per-pipeline parameter summary for a single GWOSC event
    and save it as ``{output_dir}/{event_name}.json``.

    Returns the path to the saved file, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{event_name}.json"

    if out_path.exists():
        log.info("[Step 5]  Event %s already present (skipping)", event_name)
        return out_path

    url = GWOSC_EVENT_PAGE.format(event_name=event_name)
    log.info("[Step 5]  Downloading event parameters: %s …", event_name)

    try:
        data = _get_json(url)
    except RuntimeError as exc:
        log.warning("[Step 5]  Could not fetch %s: %s", event_name, exc)
        return None

    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2)

    log.debug("[Step 5]  → saved to %s", out_path)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Summary extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_pipeline_summary(event_data: dict) -> list[dict]:
    """
    Extract per-pipeline matched-filter summary rows from a GWOSC event JSON.

    Returns a list of dicts, one per pipeline, with keys:
        pipeline, far_hz, far_per_year, network_snr, mass1, mass2,
        chirp_mass, distance_mpc
    """
    rows = []
    events = event_data.get("events", {})
    if isinstance(events, dict):
        events = list(events.values())

    for ev in events:
        pipeline = ev.get("pipeline", "unknown")
        parameters = ev.get("parameters", {})
        far_hz  = ev.get("far", None)
        snr     = ev.get("network_matched_filter_snr", None)

        # masses / distance are typically under 'parameters'
        m1  = parameters.get("mass_1_source", {}).get("best", None)
        m2  = parameters.get("mass_2_source", {}).get("best", None)
        mc  = parameters.get("chirp_mass_source", {}).get("best", None)
        dl  = parameters.get("luminosity_distance", {}).get("best", None)

        far_per_year = (far_hz * 365.25 * 24 * 3600) if far_hz is not None else None

        rows.append(
            {
                "pipeline": pipeline,
                "far_hz": far_hz,
                "far_per_year": far_per_year,
                "network_snr": snr,
                "mass1_msun": m1,
                "mass2_msun": m2,
                "chirp_mass_msun": mc,
                "distance_mpc": dl,
            }
        )
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def download_pipeline_results(cfg: dict) -> dict:
    """
    Download available matched-filter pipeline results from GWOSC.

    Behaviour:
    ----------
    1. Download the full event catalog JSON for each catalog in
       ``sensitive_volume.gwosc_catalogs`` (default: GWTC-2, GWTC-3).

    2. Download individual event pages for all BNS / NSBH events listed in
       ``sensitive_volume.bns_events`` plus any events matching the pipeline's
       mass range found in the catalogs.

    3. Parse each event page and build a consolidated summary table saved to
       ``{output_dir}/pipeline_summary.json``.

    Parameters
    ----------
    cfg:
        Top-level SparseBank config dict.

    Returns
    -------
    dict with keys:
        output_dir          – directory where results are saved
        catalogs_downloaded – list of catalog names fetched
        events_downloaded   – list of event names fetched
        pipeline_summary    – consolidated per-event, per-pipeline rows
        summary_path        – path to the saved summary JSON
    """
    sv_cfg     = cfg.get("sensitive_volume", {})
    output_dir = Path(sv_cfg.get("output_dir", "sensitive_volume_output")) / "gwosc"
    output_dir.mkdir(parents=True, exist_ok=True)

    catalogs   = sv_cfg.get("gwosc_catalogs", DEFAULT_CATALOGS)
    bns_events = sv_cfg.get("bns_events",    BNS_EVENTS)

    # chirp-mass range from the pipeline config (Step 1 BNS range)
    m1_min = float(cfg.get("m1_min", 1.0))
    m1_max = float(cfg.get("m1_max", 2.5))
    mc_min = m1_min * (0.25 ** 0.6)   # conservative: equal-mass lower bound
    mc_max = m1_max * (0.25 ** 0.6)

    downloaded_catalogs: list[str]  = []
    downloaded_events:   list[str]  = []
    pipeline_summary:    list[dict] = []

    # ── 1. Catalog download ───────────────────────────────────────────────────
    for catalog in catalogs:
        try:
            cat_path = download_gwosc_catalog(catalog, output_dir / "catalogs")
            downloaded_catalogs.append(catalog)
        except Exception as exc:
            log.warning("[Step 5] Could not download catalog %s: %s", catalog, exc)
            continue

        # Collect events in the pipeline mass range from this catalog
        with open(cat_path) as fh:
            cat_data = json.load(fh)

        events_in_catalog = cat_data if isinstance(cat_data, list) else \
                            cat_data.get("events", list(cat_data.values()))

        for ev in events_in_catalog:
            name = ev.get("commonName") or ev.get("name") or ev.get("id", "")
            if not name:
                continue
            # Include BNS / NSBH events and any event whose chirp mass falls
            # in the pipeline's BNS mass range
            ev_mc = ev.get("chirp_mass") or ev.get("chirpMass")
            in_mass_range = (
                ev_mc is not None and mc_min <= float(ev_mc) <= mc_max
            )
            if name in bns_events or in_mass_range:
                if name not in bns_events:
                    bns_events = list(bns_events) + [name]

    # ── 2. Per-event download ─────────────────────────────────────────────────
    log.info(
        "[Step 5] Downloading per-pipeline parameters for %d events …",
        len(bns_events),
    )
    events_dir = output_dir / "events"
    for name in bns_events:
        ev_path = download_event_parameters(name, events_dir)
        if ev_path is None:
            continue
        downloaded_events.append(name)
        try:
            with open(ev_path) as fh:
                ev_data = json.load(fh)
            rows = _extract_pipeline_summary(ev_data)
            for row in rows:
                row["event_name"] = name
            pipeline_summary.extend(rows)
        except Exception as exc:
            log.warning("[Step 5]  Could not parse %s: %s", ev_path, exc)

    # ── 3. Save consolidated summary ─────────────────────────────────────────
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(pipeline_summary, fh, indent=2)

    log.info(
        "[Step 5] Downloaded %d catalog(s), %d event(s).  Summary → %s",
        len(downloaded_catalogs), len(downloaded_events), summary_path,
    )

    # ── 4. Print Zenodo instructions for full trigger files ───────────────────
    _print_zenodo_instructions(output_dir)

    return {
        "output_dir":          str(output_dir),
        "catalogs_downloaded": downloaded_catalogs,
        "events_downloaded":   downloaded_events,
        "pipeline_summary":    pipeline_summary,
        "summary_path":        str(summary_path),
    }


def _print_zenodo_instructions(output_dir: Path) -> None:
    """Write a README with instructions for downloading full trigger files."""
    readme = output_dir / "FULL_TRIGGER_FILES.md"
    content = """\
# Downloading Full Matched-Filter Trigger Files

The GWOSC catalog JSON and per-event parameter summaries are downloaded
automatically by SparseBank Step 5.  Full trigger databases (gstlal SQLite /
PyCBC HDF5) are large (10–100 GB) and must be downloaded separately.

## GWTC-2 (O3a) Data Release

    Zenodo DOI: 10.5281/zenodo.5546662
    URL:        https://zenodo.org/records/5546662

Files of interest:
    gstlal/        – gstlal_inspiral trigger databases (.sqlite)
    pycbc/         – PyCBC trigger HDF5 files
    mbta/          – MBTA trigger XML files

## GWTC-3 (O3b) Data Release

    Zenodo DOI: 10.5281/zenodo.8177023
    URL:        https://zenodo.org/records/8177023

## Usage with SparseBank

Place the downloaded trigger files under:

    sensitive_volume_output/gwosc/triggers/<pipeline>/

Then set  `sensitive_volume.external_triggers_dir`  in config.yaml and
re-run Step 5 to include the external pipeline results in the V_T comparison.
"""
    with open(readme, "w") as fh:
        fh.write(content)
    log.info("[Step 5] Zenodo download instructions written to %s", readme)
