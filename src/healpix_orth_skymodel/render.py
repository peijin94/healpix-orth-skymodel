"""Turn a Healpix catalog map into a FITS-grid theoretical sky (beam + PSF)."""

from __future__ import annotations

from typing import Union

import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS
from scipy.signal import fftconvolve

from healpix_orth_skymodel.projection import healpix_to_j2000


def _psf_kernel_2d(fits_path: str, imwcs: WCS, shape: tuple[int, int]) -> np.ndarray:
    from image_plane_correction.flow import cleaned_psf_from_fits
    from image_plane_correction.util import gkern

    psf = np.asarray(
        cleaned_psf_from_fits(fits_path, shape=shape, imwcs_override=imwcs),
        dtype=np.float64,
    )
    taper = gkern(psf.shape[0], psf.shape[0] / 4.0)
    kernel = np.asarray(taper, dtype=np.float64) * psf
    peak = kernel.max()
    if peak > 0:
        kernel /= peak
    return kernel


def healpix_to_theoretical_image(
    healpix_map: np.ndarray,
    header: Union[Header, WCS],
    *,
    fits_path: str | None = None,
    nest: bool = False,
) -> np.ndarray:
    """
    Project Healpix flux map onto the image grid and convolve with the CLEAN PSF.

    Matches the last two stages of ``theoretical_sky_beam_function`` (TAN grid +
    PSF FFT convolution). The Healpix map should already include per-source primary-
    beam weighting if comparing to ``use_best_pb_model=True`` calcflow output.
    """
    if isinstance(header, WCS):
        wcs = header.celestial
        h = int(wcs.pixel_shape[1])
        w = int(wcs.pixel_shape[0])
        hdr = header.to_header()
    else:
        wcs = WCS(header).celestial
        h = int(header["NAXIS2"])
        w = int(header["NAXIS1"])
        hdr = header

    if fits_path is None:
        raise ValueError("fits_path is required for PSF kernel (BMAJ/BMIN from header)")

    plane = healpix_to_j2000(
        healpix_map, wcs, nest=nest, interpolation="linear", backend="jax"
    )
    plane = np.nan_to_num(plane, nan=0.0, posinf=0.0, neginf=0.0)
    kernel = _psf_kernel_2d(fits_path, wcs, (h, w))
    out = fftconvolve(plane, kernel, mode="same")
    return np.asarray(out, dtype=np.float32)
