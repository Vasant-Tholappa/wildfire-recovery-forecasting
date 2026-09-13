# Phase 3.4.6 — Production scaling decision

## Decision

Lock production spatial batching at:

- `tile_side_cells = 120`
- each modeling patch remains `90 m × 90 m`
- maximum patches per spatial batch = `120 × 120 = 14,400`

The 120 setting changes only execution packaging. It does not change the scientific modeling unit.

## Scaling evidence

Plan B contains 14 independent raster export products per spatial batch.

The WOODBURY stress benchmark tested increasingly large dense batches:

- tile 60: 3,600 patches — all 14 products completed
- tile 90: 8,100 patches — all 14 products completed
- tile 120: 14,400 patches — all 14 products completed
- tile 150: 16,169 retained patches in the selected dense batch — all submissions failed before Earth Engine task creation because the serialized request exceeded the 10,485,760-byte request-payload limit

The tile-120 benchmark therefore exercised the absolute maximum number of patches that any 120 × 120 production batch can contain.

## Population batch count

Exact local geometry counting over the 1,904-fire manifest produced:

- tile 90: 8,286 nonempty spatial batches
- tile 120: 6,301 nonempty spatial batches
- tile 150: 5,241 nonempty spatial batches

Historical tile-10, tile-20 and tile-30 totals were reproduced exactly before accepting these new counts.

## Population occupancy estimate

Total focal-fire area corresponds to approximately:

- 14,600,856 full 90 m × 90 m area-equivalent patches

A deliberately conservative buffered geometry calculation gives:

- 16,070,305 as the conservative retained-patch planning bound

Across 6,301 tile-120 batches this corresponds to approximately:

- ~2,317 patches/batch from raw fire area
- ~2,550 patches/batch using the conservative geometry bound

These are population-average planning values, not claims that every batch contains the same number of patches.

The important implication is that the average production batch should be far smaller than the 14,400-patch tile-120 stress benchmark.

## Wall-clock runtime conclusion

Measured 60/90/120 benchmarks show that the complete 14-product workload remains operationally manageable at tile 120.

Combining:

- 6,301 exact production batches
- population occupancy estimates
- measured benchmark execution scaling
- concurrent Earth Engine execution
- expected queue variation and isolated retries

gives an expected population runtime of roughly 4–5 days, with a practical planning envelope of approximately 5–6 days.

This satisfies the project launch gate of less than 7 days.

This is a runtime projection rather than a guaranteed completion time.

## Locked conclusion

`tile_side_cells = 120` is the production configuration.

No further tile-size scaling experiments are required.

Tile 150 is rejected under the current architecture because of the Earth Engine request-payload limit. Tile 90 is valid but creates substantially more work items without providing a necessary reliability advantage over the successfully tested tile-120 configuration.

Phase 3.4.6 is complete.

## Remaining launch gate

Wall-clock scalability and Earth Engine EECU quota are separate constraints.

The production run must not be launched until Phase 3.4.7 completes:

1. EECU quota forecasting and compute-efficiency preflight
2. bounded population orchestration / launcher implementation
3. final launch validation
4. production launch