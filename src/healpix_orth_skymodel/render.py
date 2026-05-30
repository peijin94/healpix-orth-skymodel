"""Turn a Healpix catalog map into a FITS-grid theoretical sky (beam + PSF)."""

from __future__ import annotations

from healpix_orth_skymodel import _jax_cpu  # noqa: F401

import functools
from pathlib import Path
from typing import Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from healpix_orth_skymodel.projection import healpix_to_j2000

DEFAULT_PSF_KERNEL_SIZE = 64


@functools.lru_cache(maxsize=1)
def _jax_fftconvolve_same():
    import jax
    import jax.numpy as jnp
    from jax.scipy.signal import fftconvolve

    @jax.jit
    def _conv(plane: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        return fftconvolve(plane, kernel, mode="same")

    return _conv


def psf_fft_convolve(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a 2D image with a PSF kernel via ``jax.scipy.signal.fftconvolve`` (``mode='same'``)."""
    import jax.numpy as jnp

    conv = _jax_fftconvolve_same()
    plane_j = jnp.asarray(plane, dtype=jnp.float32)
    kernel_j = jnp.asarray(kernel, dtype=jnp.float32)
    out = conv(plane_j, kernel_j)
    return np.asarray(out, dtype=np.float32)


# Backward-compatible alias
psf_convolve = psf_fft_convolve


def _beam_compact_kernel_size(bmaj_deg: float, bmin_deg: float, imwcs: WCS) -> int:
    """Pixel side length for ~4σ beam support (same rule as calcflow benchmarks)."""
    from astropy.wcs.utils import proj_plane_pixel_scales

    pixel_scales = np.abs(proj_plane_pixel_scales(imwcs))
    sigma_y = (bmaj_deg / pixel_scales[1]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_x = (bmin_deg / pixel_scales[0]) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return max(9, 2 * int(np.ceil(4.0 * max(sigma_x, sigma_y))) + 1)


def _center_crop_or_pad(arr: np.ndarray, size: int) -> np.ndarray:
    """Return ``size × size`` array with ``arr`` centered (crop or zero-pad)."""
    h, w = arr.shape
    y0_src = max(0, (h - size) // 2)
    x0_src = max(0, (w - size) // 2)
    cropped = arr[y0_src : y0_src + size, x0_src : x0_src + size]
    ch, cw = cropped.shape
    if ch == size and cw == size:
        return cropped.astype(arr.dtype, copy=False)
    out = np.zeros((size, size), dtype=arr.dtype)
    y0_dst = (size - ch) // 2
    x0_dst = (size - cw) // 2
    out[y0_dst : y0_dst + ch, x0_dst : x0_dst + cw] = cropped
    return out


def _psf_kernel_2d(
    fits_path: str,
    imwcs: WCS,
    *,
    kernel_size: int = DEFAULT_PSF_KERNEL_SIZE,
) -> np.ndarray:
    """
    Build a tapered CLEAN PSF for FFT convolution.

    Synthesizes the beam on a compact grid, then center crop/pad to ``kernel_size``
    (default 64×64).
    """
    from image_plane_correction.flow import cleaned_psf_from_fits
    from image_plane_correction.util import gkern

    if kernel_size < 3:
        raise ValueError(f"kernel_size must be >= 3, got {kernel_size}")

    header = fits.getheader(fits_path)
    bmaj_deg = float(header["BMAJ"])
    bmin_deg = float(header["BMIN"])
    compact = _beam_compact_kernel_size(bmaj_deg, bmin_deg, imwcs)
    build_n = max(compact, kernel_size)

    psf = np.asarray(
        cleaned_psf_from_fits(
            fits_path, shape=(build_n, build_n), imwcs_override=imwcs
        ),
        dtype=np.float32,
    )
    taper = np.asarray(gkern(psf.shape[0], psf.shape[0] / 4.0), dtype=np.float32)
    kernel = taper * psf
    peak = float(kernel.max())
    if peak > 0:
        kernel = kernel / peak
    return _center_crop_or_pad(kernel, kernel_size)


def healpix_to_theoretical_image(
    healpix_map: np.ndarray | None,
    header: Union[Header, WCS],
    *,
    fits_path: str | None = None,
    nest: bool = False,
    use_image_splat: bool = True,
    catalog: str = "VLSSR",
    catalog_path: str | Path | None = None,
    min_flux_mjy: float = 0.0,
    max_flux_jy: float = 20.0,
    obs_date: str | None = None,
    freq_hz: float | None = None,
    img_size: int | None = None,
    psf_kernel_size: int = DEFAULT_PSF_KERNEL_SIZE,
) -> np.ndarray:
    """
    Build a theoretical sky on the FITS grid and convolve with the CLEAN PSF (FFT).

    By default (``use_image_splat=True``) point sources use the same sub-pixel
    image-plane deposition as ``theoretical_sky_beam_function``. Pass
    ``use_image_splat=False`` to sample a pre-built Healpix map with healpy.

    The Healpix map is ignored when ``use_image_splat=True``.
    """
    if isinstance(header, WCS):
        wcs = header.celestial
        h = int(wcs.pixel_shape[1])
        w = int(wcs.pixel_shape[0])
    else:
        wcs = WCS(header).celestial
        h = int(header["NAXIS2"])
        w = int(header["NAXIS1"])

    if fits_path is None:
        raise ValueError("fits_path is required for PSF kernel (BMAJ/BMIN from header)")

    size = img_size if img_size is not None else h

    if use_image_splat:
        if obs_date is None or freq_hz is None:
            raise ValueError("obs_date and freq_hz required when use_image_splat=True")
        from image_plane_correction.catalogs import theoretical_sky_point_plane

        plane = np.asarray(
            theoretical_sky_point_plane(
                wcs,
                catalog=catalog,  # type: ignore[arg-type]
                img_size=size,
                min_flux=min_flux_mjy,
                max_flux=max_flux_jy,
                path=str(catalog_path) if catalog_path is not None else None,
                obs_date=obs_date,
                freq_hz=freq_hz,
                use_best_pb_model=True,
            ),
            dtype=np.float32,
        )
    else:
        if healpix_map is None:
            raise ValueError("healpix_map is required when use_image_splat=False")
        plane = healpix_to_j2000(
            healpix_map, wcs, nest=nest, interpolation="nearest"
        )
        plane = np.nan_to_num(plane, nan=0.0, posinf=0.0, neginf=0.0)

    kernel = _psf_kernel_2d(fits_path, wcs, kernel_size=psf_kernel_size)
    return psf_fft_convolve(plane, kernel)


def catalog_to_theoretical_image(
    header: Union[Header, WCS],
    *,
    fits_path: str,
    catalog_path: str | Path,
    obs_date: str,
    freq_hz: float,
    min_flux_mjy: float = 0.0,
    max_flux_jy: float = 20.0,
    img_size: int | None = None,
    psf_kernel_size: int = DEFAULT_PSF_KERNEL_SIZE,
) -> np.ndarray:
    """Convenience wrapper: image-plane splat + PSF (no Healpix map)."""
    return healpix_to_theoretical_image(
        None,
        header,
        fits_path=fits_path,
        use_image_splat=True,
        catalog_path=catalog_path,
        min_flux_mjy=min_flux_mjy,
        max_flux_jy=max_flux_jy,
        obs_date=obs_date,
        freq_hz=freq_hz,
        img_size=img_size,
        psf_kernel_size=psf_kernel_size,
    )
