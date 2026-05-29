"""Catalog → Healpix flux map."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import astropy.wcs as wcs
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

Catalog = Literal["VLSSR", "NVSS"]
Deposition = Literal["nearest", "bilinear"]


def _filter_finite_sources(
    ra: np.ndarray,
    dec: np.ndarray,
    flux_jy: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop sources with non-finite flux or sky position (matches ``theoretical_sky_beam_function``)."""
    ok = np.isfinite(flux_jy) & np.isfinite(ra) & np.isfinite(dec)
    n_drop = int(np.size(ok) - np.count_nonzero(ok))
    if n_drop:
        import warnings

        warnings.warn(
            f"catalog_to_healpix: dropped {n_drop} sources with non-finite {label}",
            stacklevel=3,
        )
    return ra[ok], dec[ok], flux_jy[ok], n_drop


def _filter_sources_in_footprint(
    positions: SkyCoord,
    flux_jy: np.ndarray,
    imwcs: WCS,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Keep sources that project inside the image (same cuts as ``theoretical_sky_beam_function``)."""
    xy = np.stack(wcs.utils.skycoord_to_pixel(positions, imwcs), axis=1)
    scale = img_size / float(imwcs.pixel_shape[0])
    xy = xy * scale
    in_bounds = (
        np.isfinite(xy[:, 0])
        & np.isfinite(xy[:, 1])
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < img_size)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < img_size)
    )
    n_kept = int(np.count_nonzero(in_bounds))
    if n_kept == 0:
        raise ValueError("No catalog sources inside image footprint after WCS projection")
    pos = positions[in_bounds]
    ra = np.asarray(pos.icrs.ra.deg, dtype=np.float32)
    dec = np.asarray(pos.icrs.dec.deg, dtype=np.float32)
    return ra, dec, flux_jy[in_bounds], n_kept


def catalog_to_healpix(
    catalog: Catalog,
    *,
    nside: int,
    min_flux_mjy: float = 0.0,
    max_flux_jy: float = 35.0,
    catalog_path: str | Path | None = None,
    nest: bool = False,
    apply_beam: bool = False,
    obs_date: str | None = None,
    freq_hz: float | None = None,
    deposition: Deposition = "nearest",
    imwcs: Union[WCS, None] = None,
    img_size: int | None = None,
) -> np.ndarray:
    """
    Deposit unresolved catalog sources into a Healpix map (point flux, Jy).

    Uses the same VLSSr loader and in-footprint filter as
    ``theoretical_sky_beam_function``. Point sources use ``deposition="nearest"``
    by default (one Healpix pixel per source); ``"bilinear"`` spreads flux across
    four pixels before grid sampling and tends to blur / lose peaks.
    """
    if catalog != "VLSSR":
        raise NotImplementedError(f"catalog {catalog!r} — only VLSSR wired for now")

    from image_plane_correction.catalogs import reference_sources_vlssr

    path = catalog_path or (
        "/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt"
    )
    positions, fluxes = reference_sources_vlssr(min_flux=min_flux_mjy, path=str(path))
    flux_jy = np.clip(np.asarray(fluxes, dtype=np.float32), 0, max_flux_jy)

    finite_flux = np.isfinite(flux_jy)
    if not np.all(finite_flux):
        positions = positions[np.asarray(finite_flux)]
        flux_jy = flux_jy[finite_flux]

    if imwcs is not None and img_size is not None:
        ra, dec, flux_jy, _ = _filter_sources_in_footprint(
            positions, flux_jy, imwcs, img_size
        )
    else:
        ra = np.asarray(positions.icrs.ra.deg, dtype=np.float32)
        dec = np.asarray(positions.icrs.dec.deg, dtype=np.float32)

    ra, dec, flux_jy, _ = _filter_finite_sources(ra, dec, flux_jy, label="flux or coordinates")

    if apply_beam:
        if obs_date is None or freq_hz is None:
            raise ValueError("obs_date and freq_hz required when apply_beam=True")
        from pb_correct import _get_beam

        beam = _get_beam()
        resp = beam.get_response(ra, dec, obs_date, freq_hz)
        resp = np.asarray(resp, dtype=np.float32)
        resp = np.where(np.isfinite(resp), resp, 0.0)
        flux_jy = flux_jy * resp
        ra, dec, flux_jy, _ = _filter_finite_sources(
            ra, dec, flux_jy, label="flux after beam"
        )

    npix = hp.nside2npix(nside)
    m = np.zeros(npix, dtype=np.float32)
    if deposition == "nearest":
        ipix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
        np.add.at(m, ipix, flux_jy)
    elif deposition == "bilinear":
        pix, weights = hp.get_interp_weights(nside, dec, ra, lonlat=True, nest=nest)
        contrib = (weights * flux_jy[np.newaxis, :]).astype(np.float32)
        np.add.at(m, pix.ravel(), contrib.ravel())
    else:
        raise ValueError(f"Unknown deposition {deposition!r}")
    return m
