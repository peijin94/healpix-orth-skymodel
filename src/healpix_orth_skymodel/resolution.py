"""Match Healpix nside to a FITS/WCS pixel scale."""

from __future__ import annotations

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def _nside_ideal_for_resolution_rad(resolution_rad: float) -> int:
    import healpy as hp

    nside = int(round(float(np.sqrt(np.pi / (3.0 * resolution_rad**2)))))
    if not hp.isnsideok(nside):
        raise ValueError(f"invalid nside {nside} for resolution {resolution_rad} rad")
    return nside


def _npix_fits_writable(nside: int) -> bool:
    import healpy as hp

    return hp.nside2npix(nside) % 1024 == 0


def nside_for_resolution_rad(resolution_rad: float) -> int:
    """
    Healpix ``nside`` whose mean pixel size (``healpy.nside2resol``) matches
    ``resolution_rad`` radians.

    If the ideal ``nside`` cannot be written with ``healpy.write_map`` (npix must
    be a multiple of 1024), search outward for the closest valid ``nside``.
    """
    import healpy as hp

    if resolution_rad <= 0:
        raise ValueError("resolution_rad must be positive")
    ideal = _nside_ideal_for_resolution_rad(resolution_rad)
    if _npix_fits_writable(ideal):
        return ideal

    best_nside: int | None = None
    best_err = float("inf")
    for delta in range(0, 4096):
        for nside in (ideal + delta, ideal - delta):
            if nside < 1 or not hp.isnsideok(nside):
                continue
            if not _npix_fits_writable(nside):
                continue
            err = abs(float(hp.nside2resol(nside)) - resolution_rad)
            if err < best_err:
                best_err = err
                best_nside = nside
        if best_nside is not None and delta > 0:
            break
    if best_nside is None:
        raise ValueError(f"no FITS-writable nside near ideal={ideal}")
    return best_nside


def nside_match_wcs(wcs: WCS, *, rtol: float = 1e-3) -> int:
    """
    Choose ``nside`` so Healpix pixel size matches the celestial WCS pixel scale.

    Uses the mean of ``|proj_plane_pixel_scales|`` (deg/pix) on an isometric WCS.
    """
    from healpix_orth_skymodel.projection import validate_isometric_wcs

    celestial = wcs.celestial if hasattr(wcs, "celestial") else wcs
    validate_isometric_wcs(celestial, rtol=rtol)
    scales_deg = np.abs(proj_plane_pixel_scales(celestial))
    scale_deg = float(np.mean(scales_deg))
    resolution_rad = np.deg2rad(scale_deg)
    return nside_for_resolution_rad(resolution_rad)


def healpix_resolution_arcsec(nside: int) -> float:
    """Mean Healpix pixel size in arcsec (``healpy.nside2resol``)."""
    import healpy as hp

    return float(hp.nside2resol(nside) * 206264.80624709636)
