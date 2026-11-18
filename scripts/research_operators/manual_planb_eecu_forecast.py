"""Local occupancy-weighted EECU scenarios; no EE/GCS access or production changes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/processed"
X = np.array([3600, 8100, 14400], dtype=float)


def exclude_retired_products(measured):
    """Filter historical measurements once, without modifying the saved benchmark."""
    return {tile: {key: cost for key, cost in values.items() if key != "landsat/nbrdrop"}
            for tile, values in measured.items()}


def scenarios(n, y):
    """Three-point interpolation plus explicit unmeasured small-batch scenarios."""
    slope, intercept = np.polyfit(np.log(X), np.log(y), 1)
    first_exponent = np.log(y[1] / y[0]) / np.log(X[1] / X[0])
    log_piecewise = np.exp(np.interp(np.log(n), np.log(X), np.log(y)))
    log_piecewise = np.where(n < X[0], y[0] * (n / X[0])**first_exponent,
                             log_piecewise)
    linear = np.interp(n, X, y)
    return {
        "power_fit": np.exp(intercept) * n**slope,
        "piecewise_power": log_piecewise,
        "linear_zero_anchor": np.where(n < X[0], y[0] * n / X[0], linear),
        "small_batch_floor": linear,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-daymet", action="store_true",
                        help="Replace only Daymet with the validated combined benchmark")
    parser.add_argument("--include-retired-nbrdrop", action="store_true",
                        help="Reproduce the historical forecast before approved Candidate A")
    args = parser.parse_args()
    benchmark = json.loads((DATA / "planb_benchmark_results.json").read_text())
    population = pd.read_parquet(DATA / "phase3_planb_tile120_batch_patch_counts.parquet")
    assert len(population) == 6301 and population.patch_count.sum() == 15275928
    assert not population.duplicated(["fire_id", "batch_id"]).any()
    assert population.patch_count.between(1, 14400).all()
    measured = {}
    for tile, expected_n in zip(("60", "90", "120"), X):
        batch = benchmark["batches"][tile]
        assert batch["patch_count"] == expected_n and len(batch["products"]) == 14
        values = {}
        for product in batch["products"]:
            task = product["ee_task"]
            assert task["state"] == "COMPLETED"
            key = product["source"] + ("/" + product["product"] if product["product"] else "")
            assert key not in values
            values[key] = float(task["batch_eecu_usage_seconds"])
            assert values[key] > 0
        measured[tile] = values
    assert all(set(v) == set(measured["60"]) for v in measured.values())
    n = population.patch_count.to_numpy(dtype=float)
    baseline_hours = {
        scenario: float(sum(scenarios(n, np.array([v[key] for v in measured.values()]))[scenario].sum()
                            for key in measured["60"]) / 3600)
        for scenario in ("piecewise_power", "small_batch_floor")
    }
    if args.combined_daymet:
        combined = json.loads((DATA / "daymet_combined_benchmark_results.json").read_text())
        for tile, expected_n in zip(("60", "90", "120"), X):
            batch = combined["batches"][tile]
            assert batch["patch_count"] == expected_n
            assert batch["expected_patch_ids"] == benchmark["batches"][tile]["expected_patch_ids"]
            assert batch["comparison"]["equivalent"] and batch["comparison"]["null_patterns_identical"]
            product, = batch["products"]
            assert (product["source"], product["product"]) == ("daymet", "combined")
            assert product["ee_task"]["state"] == "COMPLETED"
            cost = float(product["ee_task"]["batch_eecu_usage_seconds"])
            assert cost > 0
            del measured[tile]["daymet/primary"]
            del measured[tile]["daymet/coverage"]
            measured[tile]["daymet/combined"] = cost
    if not args.include_retired_nbrdrop:
        measured = exclude_retired_products(measured)
    totals = np.array([sum(v.values()) for v in measured.values()])
    aggregate_scenarios = scenarios(n, totals)
    product_rows = []
    central = np.zeros(len(n))
    upper = np.zeros(len(n))
    for key in measured["60"]:
        y = np.array([v[key] for v in measured.values()])
        predictions = scenarios(n, y)
        # Product-level interpolation permits an additive source breakdown.
        central += predictions["piecewise_power"]
        upper += predictions["small_batch_floor"]
        product_rows.append({"product": key, "measured_seconds": y.tolist(),
                             "central_hours": predictions["piecewise_power"].sum() / 3600,
                             "planning_hours": predictions["small_batch_floor"].sum() / 3600})
    # Linear interpolation is not a pointwise upper bound for concave segments.
    # Keep the original planning scenario; report its population sum separately.
    # Both interpolants reproduce all three measurements for each product.
    for values in zip(*(v.values() for v in measured.values())):
        np.testing.assert_allclose(scenarios(X, np.array(values))["piecewise_power"], values)
    population["central_eecu_seconds"] = central
    population["planning_eecu_seconds"] = upper
    ranges = [(1, 500), (501, 1000), (1001, 2500), (2501, 5000),
              (5001, 10000), (10001, 14400)]
    occupancy = []
    for lo, hi in ranges:
        mask = (n >= lo) & (n <= hi)
        occupancy.append({"range": f"{lo}-{hi}", "batches": int(mask.sum()),
                          "patches": int(n[mask].sum()),
                          "central_hours": central[mask].sum() / 3600,
                          "planning_hours": upper[mask].sum() / 3600})
    assert sum(v["batches"] for v in occupancy) == len(n)
    result = {
        "measured_total_seconds": dict(zip(("60", "90", "120"), totals.tolist())),
        "aggregate_model_hours": {k: float(v.sum() / 3600) for k, v in aggregate_scenarios.items()},
        "central_hours": float(central.sum() / 3600),
        "planning_hours": float(upper.sum() / 3600),
        "baseline_hours": baseline_hours,
        "combined_daymet": args.combined_daymet,
        "candidate_a": not args.include_retired_nbrdrop,
        "below_smallest_benchmark_batches": int((n < 3600).sum()),
        "below_smallest_benchmark_patch_share": float(n[n < 3600].sum() / n.sum()),
        "central_hours_at_or_above_3600": float(central[n >= 3600].sum() / 3600),
        "products": product_rows, "occupancy": occupancy,
        "assumptions": [
            "Central: sum product-level log-log interpolation; below 3600 extend first segment power law.",
            "Planning: product-level linear interpolation; below 3600 hold smallest benchmark cost constant.",
            "Power fit: log-log least squares through three totals; zero-anchor: linear segments through origin and measurements.",
            "One WOODBURY batch per size; no replicate or cross-fire uncertainty estimate.",
            "Occupancy-only transfer assumes patch count predicts work despite footprint, shape, scenes and geography differences.",
            "No retries, failed attempts, exploratory runs or downstream non-EE compute included.",
            "Scenarios are not confidence intervals or guaranteed upper/lower bounds.",
        ],
    }
    stem = "phase3_planb_eecu_forecast" + ("_combined_daymet" if args.combined_daymet else "")
    if not args.include_retired_nbrdrop:
        stem += "_candidate_a"
    population.to_csv(DATA / f"{stem}_batches.csv", index=False)
    (DATA / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
