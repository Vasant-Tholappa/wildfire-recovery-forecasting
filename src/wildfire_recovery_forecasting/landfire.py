"""LF2014 vegetation images for later native-grid, area-weighted reduction.

EVC classes: https://developers.google.com/earth-engine/datasets/catalog/
LANDFIRE_Vegetation_EVC_v1_4_0
EVH decoding follows notebooks/03_landfire_poc.ipynb and contract v2.
No initialization, evaluation, clipping, resampling, or exports occur here.
"""

from collections.abc import Mapping

import ee


# Official Tree Cover / Tree Canopy classes, expressed as midpoint percent.
EVC_TREE_COVER_PCT = {
    101: 15, 102: 25, 103: 35, 104: 45, 105: 55,
    106: 65, 107: 75, 108: 85, 109: 95,
    151: 17.5, 152: 42.5, 153: 80,
}
# All other documented EVC classes are observed non-tree support, not missing.
# Unknown codes are masked rather than silently classified as nonforest.
EVC_VALID_CODES = (
    *range(11, 26), 31, 32, *range(60, 70), 75, 76, 78,
    *range(80, 86), 95, 100, *range(101, 110), *range(111, 120),
    *range(121, 130), 150, 151, 152, 153, 161, 162, 163, 171, 172,
)


def landfire_sources(fire: Mapping) -> dict[str, str]:
    """Return the three fixed collection IDs; reject stale manifest baselines."""
    if (fire.get("landfire_condition_year") != 2014
            or fire.get("landfire_version") != "v1.4.0"):
        raise ValueError("LANDFIRE requires manifest baseline 2014 / v1.4.0")
    return {
        product: f"LANDFIRE/Vegetation/{product}/v1_4_0"
        for product in ("EVT", "EVC", "EVH")
    }


def forest_indicator(evc: ee.Image) -> ee.Image:
    """One for official EVC tree classes, zero for known others, masked if missing."""
    return evc.remap(
        list(EVC_VALID_CODES),
        [int(code in EVC_TREE_COVER_PCT) for code in EVC_VALID_CODES],
    ).rename("forest_indicator")


def decode_tree_cover(evc: ee.Image) -> ee.Image:
    """Decode tree classes to midpoint percent; non-tree/unknown pixels stay masked."""
    return evc.remap(
        list(EVC_TREE_COVER_PCT), list(EVC_TREE_COVER_PCT.values()),
    ).rename("tree_cover_pct")


def decode_tree_height(evh: ee.Image) -> ee.Image:
    """Decode tree heights to metres, including the tentative >=99 m lower bound."""
    return evh.remap(list(range(101, 200)), list(range(1, 100))).rename("tree_height_m")


def build_landfire_images(fire: Mapping, geometry: ee.Geometry) -> dict[str, ee.Image]:
    """Build vegetation and QA images for a Western CONUS manifest row.

    Geometry selects the regional source image, as in Phase 2. Images remain
    unclipped so full boundary patches retain source coverage. Native source
    projections are inherited; later reductions must specify the source grid.

    EVT retains original categorical codes on all valid source pixels. Apply
    forest_mask for forest-supported categorical summaries. Tree height is
    already restricted to EVC forest support; tree cover defines that support.
    *_valid images are 0/1 coverage indicators, including zero outside coverage;
    measurement images are never unmasked to zero. evc_valid is essential to
    distinguish absent forest from unavailable forest classification.

    Later patch fractions use full patch area for forest/EVT/EVC coverage and
    forest area for cover/height validity. Means use each variable's valid
    forest area. This function performs no patch reduction or eligibility cut.
    """
    sources = landfire_sources(fire)
    images = {
        product: ee.Image(ee.ImageCollection(source).filterBounds(geometry).first())
        .select(product)
        for product, source in sources.items()
    }
    # Use the full source legend validated in Phase 2, not observed Highline codes.
    evt_codes = ee.List(images["EVT"].get("EVT_class_values"))
    evt = images["EVT"].remap(evt_codes, evt_codes).rename("evt")
    forest = forest_indicator(images["EVC"])
    cover = decode_tree_cover(images["EVC"])
    height = decode_tree_height(images["EVH"]).updateMask(forest)
    result = {
        "forest_indicator": forest,
        "forest_mask": forest.selfMask().rename("forest_mask"),
        "evt": evt,
        "tree_cover_pct": cover,
        "tree_height_m": height,
    }
    for name, measurement in {
        "evc_valid": forest, "evt_valid": evt,
        "tree_cover_valid": cover, "tree_height_valid": height,
    }.items():
        result[name] = measurement.mask().gt(0).unmask(0, sameFootprint=False).rename(name)
    return result
