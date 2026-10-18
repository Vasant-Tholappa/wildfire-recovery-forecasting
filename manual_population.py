"""Bounded production waves/sessions; dry-run is local-only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import geopandas as gpd
import pandas as pd

from wildfire_recovery_forecasting import config, persistence, pipeline, production, work_items
from wildfire_recovery_forecasting.grid import build_fire_grid, assign_spatial_batches

DATA = Path(__file__).resolve().parent / "data/processed"
MANIFEST = DATA / "phase3_ca175_population_v1.parquet"
REPORT = DATA / "phase3_ca175_population_v1.json"
OCCUPANCY = DATA / "phase3_ca175_batches_v1.parquet"
VERSION = "phase3_ca175_v1"
INVENTORY = {"landsat": ("nbrpre", "season1", "nbr5", "season5_count"),
             "landfire": ("evc", "height", "evt_support", "evt_categories"),
             "terrain": (None,), "daymet": ("combined",), "lcms": (None,),
             "mtbs_severity": (None,)}
ACTIVE = {"READY", "RUNNING", "CANCEL_REQUESTED"}
SNAPSHOT = DATA / "ca175_population_progress.json"


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
    batches = batches[batches.fire_id.isin(manifest.fire_id)].sort_values(["fire_id", "batch_id"])
    if (len(manifest) != 175 or not manifest.fire_id.is_unique or manifest.crs is None
            or len(batches) != 729 or batches.patch_count.sum() != 2334795
            or batches.duplicated(["fire_id", "batch_id"]).any()):
        raise ValueError("Frozen population/occupancy mismatch")
    items = [{"fire_id": row.fire_id, "batch_id": row.batch_id,
              "patch_count": int(row.patch_count), "source": source, "product": product}
             for row in batches.itertuples() for source, products in INVENTORY.items()
             for product in products]
    assert len(items) == 8748
    paths = [production.status_path(**key(item)) for item in items]
    assert len(set(paths)) == len(items)
    return manifest, items


def proposed_wave(items, records, active_count, max_new, max_active):
    """Never reselect saved work; ambiguous submission intents block new work."""
    if any(r["status"] == "RUNNING" and not r.get("ee_task_id") for r in records.values()):
        return []
    capacity = max(0, min(max_new, max_active - active_count))
    return [item for item in items if production.status_path(**key(item)) not in records][:capacity]


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
    # Count every visible active EE task, including other workflows, conservatively.
    active = sum(task.get("state") in ACTIVE for task in tasks)
    return records, active


def summarize(items, records, active, max_new, max_active):
    counts = Counter(r["status"] for r in records.values())
    return {"fires": 175, "batches": 729, "expected_exports": len(items),
            "completed": counts["COMPLETE"], "running_status": counts["RUNNING"],
            "failed": counts["FAILED"], "pending": len(items)-len(records),
            "visible_active_ee_tasks": active,
            "proposed_next_wave": proposed_wave(items, records, active, max_new, max_active)}


class BatchProducts:
    """Cache only the current fire grid and current batch's existing source graphs."""
    def __init__(self, manifest):
        self.manifest = manifest
        self.fire_id = self.batch_id = None

    def __call__(self, item):
        if self.fire_id != item["fire_id"]:
            self.fire = self.manifest.set_index("fire_id", drop=False).loc[item["fire_id"]].to_dict()
            self.grid = assign_spatial_batches(build_fire_grid(self.fire["geometry"], item["fire_id"],
                perimeter_crs=self.manifest.crs), tile_side_cells=120)
            self.fire_id, self.batch_id = item["fire_id"], None
        if self.batch_id != item["batch_id"]:
            patches = self.grid[self.grid.batch_id.eq(item["batch_id"])].copy()
            if len(patches) != item["patch_count"]:
                raise ValueError("Canonical grid does not match frozen occupancy")
            self.ids = patches.patch_id.tolist()
            self.products = pipeline.build_raster_products(self.fire, patches)
            if {s: set(p) for s, p in self.products.items()} != {s: set(p) for s, p in INVENTORY.items()}:
                raise ValueError("Production product inventory changed")
            self.batch_id = item["batch_id"]
        return self.products[item["source"]][item["product"]], self.ids


def advance(items, records, tasks, client, payload, max_new, max_active, max_progress):
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
        descriptor, ids = payload(item)
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
            "batches": 729, "expected_exports": len(items), "completed": None,
            "active": None, "pending": None,
            "proposed_next_wave_if_no_saved_work": proposed_wave(wave_items, {}, 0, args.max_new, args.max_active),
            "note": "Use status for actual persisted counts. This command makes no cloud calls."}, indent=2))
        return
    # Single local operator only. A stale lock after forced termination requires review.
    lock = DATA / "ca175_population_driver.lock"
    with lock.open("x"):
        pass
    try:
        import ee
        from google.cloud import storage
        ee.Initialize(project=config.GCP_PROJECT)
        client = storage.Client(project=config.GCP_PROJECT)
        if args.mode in ("run", "session"):
            fingerprint = {"tile_side_cells": 120, "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                           "inventory": {s: list(p) for s, p in INVENTORY.items()}}
            metadata = persistence.read_run_metadata(client=client, production_version=VERSION)
            if metadata and metadata["batching_configuration"] != fingerprint:
                raise ValueError("Existing production run metadata differs; stop without submitting")
            if metadata is None:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                persistence.write_run_metadata(git_commit=commit, batching_configuration=fingerprint,
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
            ids = {r.get("ee_task_id") for r in records.values()}
            report["visible_recorded_task_eecu_seconds"] = sum(
                t.get("batch_eecu_usage_seconds", 0) or 0 for t in tasks if t.get("id") in ids)
            report["usage_note"] = "Visible task metadata only, not monthly quota or a spending cap. Check EE/Cloud consoles between waves."
            if args.mode in ("run", "session"):
                report["actions"] = advance(wave_items, records, tasks, client, payload,
                    new_limit, args.max_active, args.max_progress)
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
        lock.unlink()


if __name__ == "__main__":
    main()
