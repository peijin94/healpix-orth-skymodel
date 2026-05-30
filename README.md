# healpix-orth-skymodel

Build **theoretical OVRO-LWA skies** from a VLSS-style catalog: optionally deposit on a **Healpix** sphere, sample onto a FITS image grid (J2000 tangent plane) with **healpy**, and convolve with a CLEAN PSF via **JAX FFT** on CPU.

Designed to mirror `theoretical_sky_beam_function` in [`image-plane-correction-ovrolwa`](https://github.com/peijin94/image-plane-correction-ovrolwa) while allowing a reusable Healpix map for storage, Mollweide views, or faster reprojection onto other pointings.

## Render modes

| Mode | Function | Accuracy for point sources | Typical use |
|------|----------|--------------------------|-------------|
| **Image splat** (default) | `healpix_to_theoretical_image(..., use_image_splat=True)` | Matches calcflow (sub-pixel 4-pixel deposit + per-pixel beam) | Production theoretical sky |
| **Plain Healpix** | `healpix_to_theoretical_image(map, ..., use_image_splat=False)` | Nearest Healpix deposit + healpy nearest sample; can miss or broaden peaks | Fast preview, all-sky map reuse |

The Healpix FITS map is **ignored** when `use_image_splat=True`; splat calls `theoretical_sky_point_plane` from `image-plane-correction-ovrolwa` and only uses this package for PSF convolution.

## Matching image pixel scale

Use `nside_match_wcs` (or `nside_for_resolution_rad`) so Healpix pixel size matches the celestial WCS scale:

```python
from astropy.wcs import WCS
from healpix_orth_skymodel import nside_match_wcs, healpix_resolution_arcsec

wcs = WCS(fits_header).celestial
nside = nside_match_wcs(wcs)  # e.g. 1760 for ~120 arcsec/pix (4096² example)
print(healpix_resolution_arcsec(nside))
```

`healpy.write_map` requires `npix % 1024 == 0`. If the ideal `nside` is not FITS-writable, the helper picks the closest valid `nside` (e.g. ideal **1759** → **1760** for a 120″/pix image).

## API

```python
from healpix_orth_skymodel import (
    catalog_to_healpix,
    healpix_to_j2000,
    healpix_to_theoretical_image,
    nside_match_wcs,
)

# Optional: beam-weighted Healpix map (nearest deposit, FOV-filtered catalog)
m = catalog_to_healpix(
    "VLSSR",
    nside=nside,
    catalog_path="vlssr_radecpeak_unresolved.txt",
    apply_beam=True,
    obs_date=obs_date,
    freq_hz=freq_hz,
    imwcs=imwcs,
    img_size=4096,
    deposition="nearest",
)

# Plain: healpy sample + PSF (64×64 center kernel by default; ``psf_kernel_size=64``)
sky_plain = healpix_to_theoretical_image(
    m,
    fits_header,
    fits_path="image.fits",
    use_image_splat=False,
)

# Default: calcflow-equivalent splat + PSF (no map required)
sky = healpix_to_theoretical_image(
    None,
    fits_header,
    fits_path="image.fits",
    catalog_path="vlssr_radecpeak_unresolved.txt",
    obs_date=obs_date,
    freq_hz=freq_hz,
    use_image_splat=True,
)
```

Lower-level projection only:

```python
plane = healpix_to_j2000(m, imwcs, interpolation="nearest")
```

## Environment

- **JAX on CPU** — set `JAX_PLATFORMS=cpu` (importing the package applies this via `_jax_cpu.py`).
- **Primary beam** — set `OVRO_LWA_BEAM_H5` to the OVRO-LWA beam HDF5 when `apply_beam=True` or using splat mode.
- **Catalog / calcflow** — splat mode requires `image-plane-correction-ovrolwa` on `PYTHONPATH` or installed.

```bash
export JAX_PLATFORMS=cpu
export OVRO_LWA_BEAM_H5=/path/to/OVRO-LWA_MROsoil_updatedheight.h5
export PYTHONPATH="/path/to/healpix-orth-skymodel/src:/path/to/image-plane-correction-ovrolwa/src:$PYTHONPATH"
```

## Benchmarks (4096² example, CPU)

Indicative timings on a single node (`testdir/benchmark_theoretical_sky.py` in the parent dewarptest workspace):

| Pipeline | Time | Max flux (Jy) |
|----------|------|----------------|
| calcflow `theoretical_sky_beam_function` | ~12 s | 10.883 |
| Healpix **nside 1760** plain (proj + PSF) | ~4.5 s | ~11.0 |
| Image splat + PSF | ~11 s | 10.883 |

Plain Healpix at matched resolution is ~2.5× faster and lower memory, but peaks differ slightly from calcflow because of nearest-neighbor deposition.

### PSF convolution

Reference skies use **`jax.scipy.signal.fftconvolve`** only (`psf_fft_convolve`). The CLEAN beam is synthesized on a compact grid, then **center-cropped/padded to 64×64** (`psf_kernel_size=64`, default) before convolution — not a full `NAXIS1×NAXIS2` kernel (saves GPU memory on 4096² images).

Plots and map build: `python testdir/make_plots.py --nside auto --plain-healpix` (parent dewarptest workspace).

## Module layout

| Module | Role |
|--------|------|
| `catalog.py` | VLSS catalog → Healpix (FOV filter, beam, nearest/bilinear deposit) |
| `projection.py` | `healpix_to_j2000` — healpy sample onto isometric celestial WCS |
| `render.py` | `psf_fft_convolve` (JAX `fftconvolve`, 64×64 kernel) + `healpix_to_theoretical_image` |
| `resolution.py` | `nside_match_wcs`, `healpix_resolution_arcsec` |
| `_jax_cpu.py` | Force CPU JAX at import |

## Install

```bash
pip install -e .
# with image-plane-correction for splat mode:
pip install -e ../image-plane-correction-ovrolwa
```

Dependencies: `numpy`, `astropy`, `healpy`, `jax`, `scipy`. Splat mode additionally needs `image-plane-correction-ovrolwa` (JAX catalog deposition).

## Related

- [`image-plane-correction-ovrolwa`](https://github.com/peijin94/image-plane-correction-ovrolwa) — `calcflow`, `theoretical_sky_beam_function`, `theoretical_sky_point_plane`.
