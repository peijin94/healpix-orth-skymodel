"""Healpix map → FITS image (J2000 / ICRS tangent plane) via healpy."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

Interpolation = Literal["nearest", "linear"]


def _celestial_wcs_and_shape(header: Union[Header, WCS]) -> tuple[WCS, int, int]:
    if isinstance(header, WCS):
        wcs = header.celestial
        h = int(wcs.pixel_shape[1])
        w = int(wcs.pixel_shape[0])
    else:
        wcs = WCS(header).celestial
        h = int(header["NAXIS2"])
        w = int(header["NAXIS1"])
    return wcs, h, w


def validate_isometric_wcs(wcs: WCS, *, rtol: float = 1e-3) -> None:
    """
    Require square pixels (|cdelt1| ≈ |cdelt2|) on a 2D celestial WCS.

    LWA images use SIN/TAN-style projections with isotropic pixel scale; this does
    not assume a flat-sky RA/Dec linear approximation.
    """
    scales = np.abs(proj_plane_pixel_scales(wcs))
    if scales.size < 2:
        raise ValueError("WCS must have 2 celestial pixel scales")
    if not np.isclose(scales[0], scales[1], rtol=rtol):
        raise ValueError(
            f"Expected isometric pixels (equal |CDELT|), got scales {scales} deg/pix"
        )


def lonlat_grid_from_wcs(wcs: WCS, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """
    RA/Dec (deg) at each image pixel for an isometric 2D celestial WCS.

    Uses ``all_pix2world`` on the pixel index grid (valid for SIN/TAN/ZEA/STG etc.).
    """
    validate_isometric_wcs(wcs)
    yy, xx = np.indices((h, w), dtype=np.float32)
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    return np.asarray(ra, dtype=np.float32), np.asarray(dec, dtype=np.float32)


def _interp_healpy(
    healpix_map: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    nest: bool,
    interpolation: Interpolation,
) -> np.ndarray:
    import healpy as hp

    nside = hp.npix2nside(healpix_map.size)
    flat_ra = ra_deg.ravel()
    flat_dec = dec_deg.ravel()
    valid = np.isfinite(flat_ra) & np.isfinite(flat_dec)
    out = np.zeros(flat_ra.size, dtype=np.float32)
    if interpolation == "nearest":
        ipix = hp.ang2pix(
            nside, flat_ra[valid], flat_dec[valid], lonlat=True, nest=nest
        )
        out[valid] = healpix_map[ipix]
    elif interpolation == "linear":
        out[valid] = hp.get_interp_val(
            healpix_map,
            flat_dec[valid],
            flat_ra[valid],
            lonlat=True,
            nest=nest,
        )
    else:
        raise ValueError(f"Unknown interpolation {interpolation!r}")
    out[~valid] = np.nan
    return out.reshape(ra_deg.shape)


def healpix_to_j2000(
    healpix_map: np.ndarray,
    header: Union[Header, WCS],
    *,
    nest: bool = False,
    interpolation: Interpolation = "nearest",
) -> np.ndarray:
    """
    Sample a Healpix map onto the pixel grid defined by a FITS header / WCS (healpy).

    Assumes an **isometric** celestial WCS (equal pixel scale). For unresolved
    point-source catalogs use ``interpolation="nearest"``; use ``"linear"`` for
    smooth maps.
    """
    wcs, h, w = _celestial_wcs_and_shape(header)
    ra, dec = lonlat_grid_from_wcs(wcs, h, w)
    return _interp_healpy(
        healpix_map, ra, dec, nest=nest, interpolation=interpolation
    )
