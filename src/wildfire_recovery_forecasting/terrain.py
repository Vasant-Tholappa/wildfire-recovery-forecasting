"""Native 3DEP terrain images and coverage QA; no evaluation or exports."""

from math import pi

import ee


THREE_DEP_SOURCE = "USGS/3DEP/10m_collection"
# Include source neighbors for terrain's four-neighbor derivative stencil.
DEM_NEIGHBORHOOD_M = 100


def compose_elevation(primary: ee.Image, alternatives: ee.ImageCollection) -> ee.Image:
    """Keep primary valid pixels; fill gaps in ascending system:index priority.

    unmask(..., sameFootprint=False) extends coverage without replacing valid
    elevations or filling unsupported pixels with zero. Alternatives retain
    their masks; no overlap averaging occurs. Explicitly evaluate the composite
    on the primary's exact native grid before neighborhood derivatives. EE's
    default nearest-neighbor sampling aligns differing source transforms.
    """
    def fill(image, accumulated):
        return ee.Image(accumulated).unmask(ee.Image(image), sameFootprint=False)

    combined = ee.Image(alternatives.sort("system:index").iterate(fill, primary))
    return combined.reproject(primary.projection()).rename("elevation")


def aspect_components(aspect: ee.Image) -> dict[str, ee.Image]:
    """Convert clockwise-from-north aspect degrees to masked unit components."""
    radians = aspect.multiply(pi / 180)
    return {
        "northness": radians.cos().rename("northness"),
        "eastness": radians.sin().rename("eastness"),
    }


def build_terrain_images(geometry: ee.Geometry) -> dict[str, ee.Image]:
    """Compose native DEM coverage and derive terrain before forest masking.

    The caller initializes EE. Preserve the legacy first intersecting image
    wherever valid and use its projection as the reference. Ordered alternatives
    include a small halo so selection does not omit derivative neighbors.
    No clipping is applied; original valid overlap values win.
    Elevation is metres and slope is degrees. Raw aspect is not returned as a
    modeling feature. Its validity mask serves both northness and eastness.
    Terrain derivative edge masks are preserved. Only QA indicators fill
    missing coverage with zero; measurements remain masked. Forest support
    and explicit native-grid patch reductions belong to later orchestration.
    """
    source = ee.ImageCollection(THREE_DEP_SOURCE).select("elevation")
    primary = ee.Image(source.filterBounds(geometry).first())
    alternatives = source.filterBounds(geometry.buffer(DEM_NEIGHBORHOOD_M, maxError=1))
    elevation = compose_elevation(primary, alternatives)
    products = ee.Terrain.products(elevation)
    slope = products.select("slope")
    aspect = products.select("aspect")
    result = {"elevation": elevation, "slope": slope, **aspect_components(aspect)}
    for name, image in {
        "elevation_valid": elevation, "slope_valid": slope, "aspect_valid": aspect,
    }.items():
        result[name] = image.mask().gt(0).unmask(0, sameFootprint=False).rename(name)
    return result
