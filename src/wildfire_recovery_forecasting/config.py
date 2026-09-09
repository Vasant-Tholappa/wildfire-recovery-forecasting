"""Production constants from docs/phase3_production_contract.md."""

GRID_CRS = "EPSG:5070"
PATCH_SIZE_M = 90

# Calendar boundaries as (month, day); the end date is inclusive.
GROWING_SEASON_START_MONTH_DAY = (6, 1)
GROWING_SEASON_END_MONTH_DAY = (9, 30)

NBR_PRE_SEARCH_SEASONS = 5
NBR_PRE_REQUIRED_USABLE_SEASONS = 3
# Per-pixel seasonal usability for NBRpre, not a final patch eligibility rule.
LANDSAT_MIN_VALID_ACQUISITION_DATES = 3

DAYMET_HISTORICAL_BASELINE_YEARS = 10
MTBS_PRIOR_FIRE_LOOKBACK_YEARS = (5, 7, 10)

# Identifiers for the production contract, output schema, and grid.
PRODUCTION_CONTRACT_VERSION = "v2"
SCHEMA_VERSION = "v1"
GRID_VERSION = "v1"

# Cloud execution contract: notes/phase3_4_production_contract.md.
GCP_PROJECT = "ee-wildfire-recovery-vasant"
GCS_BUCKET = "ee-wildfire-recovery-vasant-data"
GCS_REGION = "us-central1"
PRODUCTION_VERSION = "phase3_prod_v1"
