"""Native 3DEP terrain images and coverage QA; no evaluation or exports."""

from math import pi

import ee


THREE_DEP_SOURCE = "USGS/3DEP/10m_collection"


def aspect_components(aspect: ee.Image) -> dict[str, ee.Image]:
    """Convert clockwise-from-north aspect degrees to masked unit components."""
    radians = aspect.multiply(pi / 180)
    return {
        "northness": radians.cos().rename("northness"),
        "eastness": radians.sin().rename("eastness"),
    }


def build_terrain_images(geometry: ee.Geometry) -> dict[str, ee.Image]:
    """Select the regional DEM as in Phase 2 and derive terrain before clipping.

    The caller initializes EE. Geometry selects the intersecting regional
    source image; full source support and its native projection are retained.
    Elevation is metres and slope is degrees. Raw aspect is not returned as a
    modeling feature. Its validity mask serves both northness and eastness.
    Terrain derivative edge masks are preserved. Only QA indicators fill
    missing coverage with zero; measurements remain masked. Forest support
    and explicit native-grid patch reductions belong to later orchestration.
    """
    elevation = ee.Image(
        ee.ImageCollection(THREE_DEP_SOURCE).filterBounds(geometry).first()
    ).select("elevation")
    products = ee.Terrain.products(elevation)
    slope = products.select("slope")
    aspect = products.select("aspect")
    result = {"elevation": elevation, "slope": slope, **aspect_components(aspect)}
    for name, image in {
        "elevation_valid": elevation, "slope_valid": slope, "aspect_valid": aspect,
    }.items():
        result[name] = image.mask().gt(0).unmask(0, sameFootprint=False).rename(name)
    return result
