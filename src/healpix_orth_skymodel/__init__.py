"""Healpix skymodel → orthographic FITS image (work in progress)."""

from healpix_orth_skymodel import _jax_cpu  # noqa: F401 — JAX_PLATFORMS=cpu

from healpix_orth_skymodel.catalog import catalog_to_healpix
from healpix_orth_skymodel.projection import (
    healpix_to_j2000,
    lonlat_grid_from_wcs,
    validate_isometric_wcs,
)
from healpix_orth_skymodel.render import (
    catalog_to_theoretical_image,
    healpix_to_theoretical_image,
)

__all__ = [
    "healpix_to_j2000",
    "healpix_to_theoretical_image",
    "catalog_to_theoretical_image",
    "catalog_to_healpix",
    "lonlat_grid_from_wcs",
    "validate_isometric_wcs",
]
