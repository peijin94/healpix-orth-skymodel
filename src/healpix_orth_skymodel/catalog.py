"""Catalog → Healpix flux map."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

Catalog = Literal["VLSSR", "NVSS"]


def catalog_to_healpix(
    catalog: Catalog,
    *,
    nside: int,
    min_flux_mjy: float = 10.0,
    max_flux_jy: float = 35.0,
    catalog_path: str | Path | None = None,
    nest: bool = False,
    apply_beam: bool = False,
    obs_date: str | None = None,
    freq_hz: float | None = None,
) -> np.ndarray:
    """
  Deposit unresolved catalog sources into a Healpix map (point flux, Jy).

  This matches the point-source step of ``theoretical_sky_beam_function`` before
  PSF convolution; beam and PSF belong in later pipeline stages.
  """
    if catalog != "VLSSR":
        raise NotImplementedError(f"catalog {catalog!r} — only VLSSR wired for now")
    import pandas as pd

    path = catalog_path or "/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt"
    df = pd.read_csv(path, sep=" ")
    df = df.sort_values("PEAK INT")
    df = df[df["PEAK INT"] >= min_flux_mjy * 10]
    fluxes = df["PEAK INT"].to_numpy(dtype=np.float64) / 10.0
    positions = SkyCoord(df.to_numpy()[:, 0:2], unit="deg")
    ra = positions.icrs.ra.deg
    dec = positions.icrs.dec.deg
    flux_jy = np.clip(np.asarray(fluxes, dtype=np.float64), 0, max_flux_jy)

    if apply_beam:
        if obs_date is None or freq_hz is None:
            raise ValueError("obs_date and freq_hz required when apply_beam=True")
        from pb_correct import _get_beam

        beam = _get_beam()
        resp = beam.get_response(ra, dec, obs_date, freq_hz)
        resp = np.asarray(resp, dtype=np.float64)
        resp = np.where(np.isfinite(resp), resp, 0.0)
        flux_jy = flux_jy * resp

    npix = hp.nside2npix(nside)
    m = np.zeros(npix, dtype=np.float64)
    ipix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
    np.add.at(m, ipix, flux_jy)
    return m
