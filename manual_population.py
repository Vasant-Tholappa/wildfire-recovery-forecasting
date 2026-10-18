"""Bounded production waves/sessions; dry-run is local-only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import re
import numpy as np
import shapely

import geopandas as gpd
import pandas as pd

from wildfire_recovery_forecasting import config, persistence, pipeline, production, work_items, reductions, daymet
from wildfire_recovery_forecasting.grid import build_fire_grid, assign_spatial_batches

DATA = Path(__file__).resolve().parent / "data/processed"
VERSION = "phase3_ca175_tile1_v1"
MANIFEST = DATA / (VERSION + ".parquet")
REPORT = DATA / (VERSION + ".json")
OCCUPANCY = DATA / (VERSION + "_batches.parquet")
# Verified COMPLETE outputs are referenced in place; never copied or overwritten.
REUSED_BATCHES = {("CA4118512343320200728", "r215_c-210"): "phase3_ca175_v1"}
INVENTORY = {"landsat": ("nbrpre", "season1", "nbr5", "season5_count"),
             "landfire": ("evc", "height", "evt_support", "evt_categories"),
             "terrain": (None,), "daymet": ("combined",), "lcms": (None,),
             "mtbs_severity": (None,)}
ACTIVE = {"READY", "RUNNING", "CANCEL_REQUESTED"}
SNAPSHOT = DATA / (VERSION + "_progress.json")


def key(item):
    return {**{name: item[name] for name in ("source", "product", "fire_id", "batch_id")},
            "production_version": VERSION}


def load_population():
    manifest = gpd.read_parquet(MANIFEST).sort_values("fire_id")
    report = json.loads(REPORT.read_text())
    ids_hash = hashlib.sha256("\n".join(manifest.fire_id).encode()).hexdigest()
    if ids_hash != report["selected_ids_sha256"]:
        raise ValueError("Selected manifest IDs changed")
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != report["manifest_sha256"]:
        raise ValueError("Frozen manifest contents changed")
    if report["selected_ids_sha256"] != "3fc06786066229a08fbd63016826f6fd49beac1c6b43fbf09989b92e4bc5659f":
        raise ValueError("Wrong approved California selection")
    if not manifest.fire_id.str.startswith("CA").all():
        raise ValueError("Non-California fire")
    batches = pd.read_parquet(OCCUPANCY)
    if hashlib.sha256(OCCUPANCY.read_bytes()).hexdigest() != report["batches_sha256"]:
        raise ValueError("Frozen tile selection changed")
    if report["production_version"] != VERSION or {
        (r["fire_id"], r["batch_id"]): r["production_version"] for r in report["reuse"]
    } != REUSED_BATCHES:
        raise ValueError("Frozen version/reuse identity changed")
    batches = batches[batches.fire_id.isin(manifest.fire_id)].sort_values(["fire_id", "batch_id"])
    if (len(manifest) != 175 or not manifest.fire_id.is_unique or manifest.crs is None
            or len(batches) != 175 or batches.patch_count.sum() != 308048
            or not batches.fire_id.is_unique or set(batches.fire_id) != set(manifest.fire_id)
            or batches.duplicated(["fire_id", "batch_id"]).any()):
        raise ValueError("Frozen population/occupancy mismatch")
    items = [{"fire_id": row.fire_id, "batch_id": row.batch_id,
              "patch_count": int(row.patch_count), "source": source, "product": product}
             for row in batches.itertuples() for source, products in INVENTORY.items()
             for product in products]
    assert len(items) == 2100
    paths = [production.status_path(**key(item)) for item in items]
    assert len(set(paths)) == len(items)
    return manifest, items


def proposed_wave(items, records, active_count, max_new, max_active):
    """Never reselect saved work; ambiguous submission intents block new work."""
    if any(r["status"] == "RUNNING" and not r.get("ee_task_id") for r in records.values()):
        return []
    capacity = max(0, min(max_new, max_active - active_count))
    return [item for item in items if production.status_path(**key(item)) not in records
            and (item["fire_id"], item["batch_id"]) not in REUSED_BATCHES][:capacity]


def read_cloud_state(items, client, tasks, completed_cache=None):
    prefix = f"production/{VERSION}/status/"
    expected = {production.status_path(**key(item)): item for item in items}
    records = {}
    for blob in client.list_blobs(config.GCS_BUCKET, prefix=prefix):
        if blob.name not in expected:
            raise ValueError(f"Unexpected work-item status in production namespace: {blob.name}")
        record = (completed_cache or {}).get(blob.name)
        if record is None:
            record = persistence.read_status(**key(expected[blob.name]), client=client)
        if record and record["status"] == "COMPLETE" and completed_cache is not None:
            completed_cache[blob.name] = record
        if record is None:
            raise ValueError("Status disappeared during scan; stop and inspect")
        records[blob.name] = record
    for path, item in expected.items():
        old_version = REUSED_BATCHES.get((item["fire_id"], item["batch_id"]))
        if old_version is None:
            continue
        if path in records:
            raise ValueError("Reused batch unexpectedly has new-namespace status; inspect before proceeding")
        old_key = {**key(item), "production_version": old_version}
        record = (completed_cache or {}).get(path)
        if record is None:
            record = persistence.read_status(**old_key, client=client)
        expected_uri = f"gs://{config.GCS_BUCKET}/{production.parquet_path(**old_key)}"
        if (not record or record["status"] != "COMPLETE" or record["stage"] != "COMPLETE"
                or record["row_count"] != item["patch_count"] or record["output_uri"] != expected_uri):
            raise ValueError("Reused output must remain COMPLETE with the verified row count and URI")
        records[path] = record  # Virtual work-item key; actual URI retains its original version.
        if completed_cache is not None:
            completed_cache[path] = record
    # Count every visible active EE task, including other workflows, conservatively.
    active = sum(task.get("state") in ACTIVE for task in tasks)
    return records, active


def summarize(items, records, active, max_new, max_active):
    counts = Counter(r["status"] for r in records.values())
    return {"fires": 175, "batches": 175, "expected_exports": len(items),
            "completed": counts["COMPLETE"], "running_status": counts["RUNNING"],
            "failed": counts["FAILED"], "pending": len(items)-len(records),
            "visible_active_ee_tasks": active,
            "proposed_next_wave": proposed_wave(items, records, active, max_new, max_active)}


def wall_clock_progress(items, records, created_at, now=None):
    """Persisted completions only; exclude reused outputs from measured throughput."""
    now = now or datetime.now(timezone.utc)
    reused = sum(r["status"] == "COMPLETE" and r.get("production_version") != VERSION
                 for r in records.values())
    completed = [r for r in records.values() if r["status"] == "COMPLETE"
                 and r.get("production_version") == VERSION]
    hours = max(0, (now-datetime.fromisoformat(created_at)).total_seconds()/3600) if created_at else None
    rate = len(completed)/hours if hours else None
    remaining = len(items)-reused-len(completed)
    return {"new_persisted_products": len(completed), "reused_products": reused,
            "remaining_products": remaining, "wall_clock_start": created_at,
            "elapsed_wall_hours": hours, "persisted_new_products_per_hour": rate,
            "projected_remaining_hours": remaining/rate if rate else None,
            "timing_note": "Since run metadata creation, including preparation, polling, persistence and pauses. Projection is not a guarantee."}


def raw_columns(source, product):
    """Existing raw schema without constructing EE source images; graph-tested."""
    if source == "landsat": names = reductions.LANDSAT_PRODUCTS[product]
    elif source == "landfire":
        if product == "evt_categories": return ["groups"]
        names = {"evc": ["forest", "tree_cover", "support"], "height": ["tree_height", "support"],
                 "evt_support": ["evt_coverage", "support"]}[product]
    elif source == "terrain": names = ["elevation", "slope", "northness", "eastness", "support"]
    elif source == "daymet":
        names = list(daymet.HISTORICAL_BANDS + daymet.PRE_S1_BANDS + daymet.PRE_S1_ANOMALY_BANDS + daymet.S1_ANOMALY_BANDS)
        for period in ("pre_s1", "s1"):
            for band in daymet.SUMMARY_BANDS:
                names += [f"daymet_hist_{period}_{band}_daily_coverage", f"daymet_{period}_{band}_valid_days"]
    elif source == "lcms":
        names = [f"lcms_{window}_{name}" for window in ("pre_10yr", "post_y_to_y5")
                 for name in ("clear_disturbance", "other_loss", "complete_coverage", "valid_years")]
    elif source == "mtbs_severity": names = ["mtbs_"+n for n in reductions.SEVERITY_CLASSES]+["mtbs_valid","mtbs_meaningful_burn"]
    else: raise ValueError("Unknown source")
    return [reductions.RASTER_FULL_AREA]+[n+s for n in names for s in ("_sum","_area")]


def batch_patches(perimeter, fire_id, batch_id):
    """Exact canonical positive-area intersections, limited to one lattice tile."""
    match = re.fullmatch(r"r(-?\d+)_c(-?\d+)", batch_id)
    if not match: raise ValueError("Invalid batch ID")
    tr, tc = map(int, match.groups())
    rows, cols = np.meshgrid(np.arange(tr*120,(tr+1)*120), np.arange(tc*120,(tc+1)*120), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    cells = shapely.box(cols*90,rows*90,(cols+1)*90,(rows+1)*90)
    areas = shapely.area(shapely.intersection(cells, perimeter))
    keep = areas > 0
    rows, cols = rows[keep], cols[keep]
    return gpd.GeoDataFrame({"patch_id":[f"{fire_id}_r{r}_c{c}" for r,c in zip(rows,cols)],
        "fire_id":fire_id,"grid_row":rows,"grid_col":cols,"patch_area_m2":8100.,
        "fire_overlap_fraction":areas[keep]/8100.,"batch_id":batch_id},geometry=cells[keep],crs=config.GRID_CRS)


class BatchProducts:
    """Reuse exact batch geometry; reconciliation never constructs raster graphs."""
    def __init__(self, manifest):
        self.manifest = manifest.set_index("fire_id",drop=False)
        self.perimeters = manifest.to_crs(config.GRID_CRS).set_index("fire_id").geometry
        self.current = None
        self.products = None

    def prepare(self, item):
        identity = (item["fire_id"],item["batch_id"])
        if self.current != identity:
            self.fire = self.manifest.loc[item["fire_id"]].to_dict()
            self.patches = batch_patches(self.perimeters.loc[item["fire_id"]], *identity)
            if len(self.patches) != item["patch_count"]: raise ValueError("Canonical batch occupancy mismatch")
            self.current, self.products = identity, None
        return self.patches.patch_id.tolist()

    def reconciliation(self, item):
        return {"columns":raw_columns(item["source"],item["product"])}, self.prepare(item)

    def __call__(self, item):
        ids = self.prepare(item)
        if self.products is None:
            self.products = pipeline.build_raster_products(self.fire,self.patches)
            if {s:set(p) for s,p in self.products.items()} != {s:set(p) for s,p in INVENTORY.items()}:
                raise ValueError("Production inventory changed")
            for source, products in self.products.items():
                for product, descriptor in products.items():
                    if descriptor["columns"] != raw_columns(source,product): raise ValueError("Raw schema changed")
        return self.products[item["source"]][item["product"]], ids


def advance(items, records, tasks, client, payload, max_new, max_active, max_progress, *, reconcile=None):
    """One bounded pass. No retries; the existing controller owns all state transitions."""
    actions = []
    active = sum(t.get("state") in ACTIVE for t in tasks)
    by_id = {t["id"]: t for t in tasks}
    progressed = 0
    for item in items:
        record = records.get(production.status_path(**key(item)))
        if not record or record["status"] != "RUNNING" or not record.get("ee_task_id"):
            continue
        task = by_id.get(record["ee_task_id"])
        if task is None or task.get("state") in ACTIVE:
            continue  # Unknown task stays reserved, never resubmitted.
        if progressed >= max_progress:
            break
        descriptor, ids = (reconcile or payload)(item)
        result = work_items.advance_work_item(None, descriptor["columns"], **key(item),
            expected_patch_ids=ids, client=client, task_status_lookup=lambda _, t=task: t)
        actions.append({**key(item), "action": result["action"]})
        progressed += 1
        if result["action"] not in ("COMPLETE", "WAITING", "SKIPPED_COMPLETE"):
            return actions
    # Reserve missing task IDs too: a truncated task list is not evidence of inactivity.
    active += sum(r["status"] == "RUNNING" and r.get("ee_task_id") not in by_id for r in records.values())
    for item in proposed_wave(items, records, active, max_new, max_active):
        descriptor, ids = payload(item)
        result = work_items.advance_work_item(descriptor["collection"], descriptor["columns"],
            **key(item), expected_patch_ids=ids, client=client)
        actions.append({**key(item), "action": result["action"]})
        print(json.dumps(actions[-1]), flush=True)
        if result["action"] not in ("SUBMITTED", "SKIPPED_COMPLETE", "WAITING"):
            break  # Stop this wave on failure or ambiguity; no automatic retry.
    return actions


def run_session(step, *, submission_cap, poll_seconds, max_minutes,
                sleep=time.sleep, clock=time.monotonic):
    """Repeat existing waves within a finite submission/time budget; errors stop."""
    submitted = 0
    deadline = clock() + max_minutes * 60
    while clock() < deadline:
        report = step(submission_cap - submitted)
        actions = report.get("actions", [])
        submitted += sum(a["action"] == "SUBMITTED" for a in actions)
        report["session_newly_submitted"] = submitted
        report["session_submission_cap"] = submission_cap
        SNAPSHOT.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if any(a["action"] not in ("COMPLETE", "SUBMITTED", "WAITING", "SKIPPED_COMPLETE") for a in actions):
            return report
        if submission_cap > 0 and submitted >= submission_cap:
            return report  # Remote tasks continue; next invocation collects them.
        if not actions and not report["running_status"] and not report["visible_active_ee_tasks"]:
            return report
        remaining = deadline - clock()
        if remaining <= 0:
            return report
        sleep(min(poll_seconds, remaining))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "status", "run", "session"))
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--max-progress", type=int, default=12)
    parser.add_argument("--session-new-cap", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-minutes", type=int, default=60)
    parser.add_argument("--only-batch", help="Optional exact fire_id/batch_id for initial production throughput checks")
    args = parser.parse_args()
    if args.session_new_cap < 0 or args.poll_seconds < 30 or args.max_minutes <= 0:
        parser.error("Session cap must be nonnegative, poll >=30 seconds, duration positive")
    if min(args.max_new, args.max_active, args.max_progress) < 0 or args.max_active == 0:
        parser.error("Limits must be nonnegative and max-active positive")
    manifest, items = load_population()
    wave_items = items
    if args.only_batch:
        wave_items = [item for item in items if item["fire_id"]+"/"+item["batch_id"] == args.only_batch]
        if not wave_items:
            parser.error("Batch is not in the frozen California population")
    if args.mode == "dry-run":
        print(json.dumps({"mode": "OFFLINE; cloud status unknown", "fires": len(manifest),
            "batches": 175, "candidate_patches": 308048, "expected_exports": len(items), "completed": None,
            "reuse_work_items": 12, "production_version": VERSION,
            "active": None, "pending": None,
            "proposed_next_wave_if_no_saved_work": proposed_wave(wave_items, {}, 0, args.max_new, args.max_active),
            "note": "Use status for actual persisted counts. This command makes no cloud calls."}, indent=2))
        return
    # Single local operator only. A stale lock after forced termination requires review.
    lock = DATA / (VERSION + "_driver.lock")
    owns_lock = args.mode in ("run", "session")
    if owns_lock:
        with lock.open("x"):
            pass
    try:
        import ee
        from google.cloud import storage
        ee.Initialize(project=config.GCP_PROJECT)
        client = storage.Client(project=config.GCP_PROJECT)
        metadata = persistence.read_run_metadata(client=client, production_version=VERSION)
        if args.mode in ("run", "session"):
            fingerprint = {"tile_side_cells": 120, "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                           "batches_sha256": hashlib.sha256(OCCUPANCY.read_bytes()).hexdigest(),
                           "reuse": json.loads(REPORT.read_text())["reuse"],
                           "inventory": {s: list(p) for s, p in INVENTORY.items()}}
            if metadata and metadata["batching_configuration"] != fingerprint:
                raise ValueError("Existing production run metadata differs; stop without submitting")
            if metadata is None:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                metadata = persistence.write_run_metadata(git_commit=commit, batching_configuration=fingerprint,
                    created_at=datetime.now(timezone.utc), client=client, production_version=VERSION)
        payload = BatchProducts(manifest)
        completed_cache = {}  # COMPLETE is terminal; avoid rereading every completed JSON each minute.

        def step(new_limit):
            tasks = ee.data.getTaskList()  # One snapshot, not a polling loop.
            records, active = read_cloud_state(items, client, tasks, completed_cache)
            report = summarize(items, records, active, new_limit, args.max_active)
            report["production_version"] = VERSION
            report["proposed_next_wave"] = proposed_wave(wave_items, records, active, new_limit, args.max_active)
            report["checked_at"] = datetime.now(timezone.utc).isoformat()
            report["throughput"] = wall_clock_progress(items, records, metadata["created_at"] if metadata else None)
            ids = {r.get("ee_task_id") for r in records.values()}
            report["visible_recorded_task_eecu_seconds"] = sum(
                t.get("batch_eecu_usage_seconds", 0) or 0 for t in tasks if t.get("id") in ids)
            report["usage_note"] = "Visible task metadata only, not monthly quota or a spending cap. Check EE/Cloud consoles between waves."
            if args.mode in ("run", "session"):
                report["actions"] = advance(wave_items, records, tasks, client, payload,
                    new_limit, args.max_active, args.max_progress, reconcile=payload.reconciliation)
                report["counts_note"] = "Pre-action snapshot; actions and session submission count are current."
            return report

        if args.mode == "session":
            run_session(step, submission_cap=args.session_new_cap, poll_seconds=args.poll_seconds,
                        max_minutes=args.max_minutes)
        else:
            report = step(args.max_new)
            SNAPSHOT.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        if owns_lock:
            lock.unlink()


if __name__ == "__main__":
    main()
