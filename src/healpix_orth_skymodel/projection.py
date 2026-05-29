"""Healpix map → FITS image (J2000 / ICRS tangent plane)."""

from __future__ import annotations

import functools
import os

# Match healpy float64 precision in jax-healpy interpolation.
os.environ.setdefault("JAX_ENABLE_X64", "1")
from typing import Literal, Union

import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

Backend = Literal["auto", "jax", "healpy"]
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
    yy, xx = np.indices((h, w), dtype=np.float64)
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)


@functools.lru_cache(maxsize=8)
def _jax_linear_sampler(nest: bool):
    import jax
    import jax.numpy as jnp
    import jax_healpy as jhp

    @jax.jit
    def _sample(m: jnp.ndarray, ra_deg: jnp.ndarray, dec_deg: jnp.ndarray) -> jnp.ndarray:
        vals = jhp.get_interp_val(m, ra_deg, dec_deg, lonlat=True, nest=nest)
        ok = jnp.isfinite(ra_deg) & jnp.isfinite(dec_deg)
        return jnp.where(ok, vals, jnp.nan)

    return _sample


@functools.lru_cache(maxsize=8)
def _jax_nearest_sampler(nside: int, nest: bool):
    import jax
    import jax.numpy as jnp
    import jax_healpy as jhp

    @jax.jit
    def _sample(m: jnp.ndarray, ra_deg: jnp.ndarray, dec_deg: jnp.ndarray) -> jnp.ndarray:
        ipix = jhp.ang2pix(nside, ra_deg, dec_deg, lonlat=True, nest=nest)
        vals = m[ipix]
        ok = jnp.isfinite(ra_deg) & jnp.isfinite(dec_deg)
        return jnp.where(ok, vals, jnp.nan)

    return _sample


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
    out = np.zeros(flat_ra.size, dtype=np.float64)
    if interpolation == "nearest":
        ipix = hp.ang2pix(
            nside, flat_ra[valid], flat_dec[valid], lonlat=True, nest=nest
        )
        out[valid] = healpix_map[ipix]
    elif interpolation == "linear":
        colat = np.deg2rad(90.0 - flat_dec[valid])
        phi = np.deg2rad(flat_ra[valid])
        out[valid] = hp.get_interp_val(healpix_map, colat, phi, nest=nest)
    else:
        raise ValueError(f"Unknown interpolation {interpolation!r}")
    out[~valid] = np.nan
    return out.reshape(ra_deg.shape)


def _interp_jax(
    healpix_map: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    nest: bool,
    interpolation: Interpolation,
) -> np.ndarray:
    import jax.numpy as jnp

    import healpy as hp

    m = jnp.asarray(healpix_map, dtype=jnp.float64)
    ra = jnp.asarray(ra_deg, dtype=jnp.float64)
    dec = jnp.asarray(dec_deg, dtype=jnp.float64)
    if interpolation == "nearest":
        nside = hp.npix2nside(healpix_map.size)
        sampler = _jax_nearest_sampler(nside, nest)
    elif interpolation == "linear":
        sampler = _jax_linear_sampler(nest)
    else:
        raise ValueError(f"Unknown interpolation {interpolation!r}")
    out = sampler(m, ra, dec)
    return np.asarray(out, dtype=np.float32)


def healpix_to_j2000(
    healpix_map: np.ndarray,
    header: Union[Header, WCS],
    *,
    nest: bool = False,
    interpolation: Interpolation = "linear",
    backend: Backend = "auto",
) -> np.ndarray:
    """
    Sample a Healpix map onto the pixel grid defined by a FITS header / WCS.

    Uses [jax-healpy](https://github.com/CMBSciPol/jax-healpy) by default for
    bilinear interpolation (JIT-compiled). Assumes an **isometric** celestial WCS
    (equal pixel scale); typical LWA SIN/TAN images satisfy this.

    Parameters
    ----------
    healpix_map
        1D Healpix array (RING ordering unless ``nest=True``).
    header
        FITS header or `~astropy.wcs.WCS` for the output image (celestial axes).
    nest
        Healpix nested ordering if True.
    interpolation
        ``"linear"`` (default) or ``"nearest"``.
    backend
        ``"auto"`` or ``"jax"`` → jax-healpy; ``"healpy"`` → classic healpy.
    """
    wcs, h, w = _celestial_wcs_and_shape(header)
    ra, dec = lonlat_grid_from_wcs(wcs, h, w)

    use_jax = backend in ("auto", "jax")
    if use_jax:
        try:
            return _interp_jax(
                healpix_map, ra, dec, nest=nest, interpolation=interpolation
            )
        except ImportError:
            if backend == "jax":
                raise
    return _interp_healpy(
        healpix_map, ra, dec, nest=nest, interpolation=interpolation
    ).astype(np.float32)
