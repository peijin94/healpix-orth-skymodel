# healpix-orth-skymodel

Build **theoretical LWA skies** on a **Healpix** sphere, then sample onto a FITS image grid (J2000 tangent plane) with **healpy**, and convolve with a CLEAN PSF via **JAX FFT**.

## API

```python
from healpix_orth_skymodel import catalog_to_healpix, healpix_to_theoretical_image

m = catalog_to_healpix(
    "VLSSR",
    nside=2048,
    catalog_path="...",
    apply_beam=True,
    obs_date=obs_date,
    freq_hz=freq_hz,
    imwcs=imwcs,
    img_size=4096,
)
sky = healpix_to_theoretical_image(
    None,
    fits_header,
    fits_path="image.fits",
    catalog_path="...",
    obs_date=obs_date,
    freq_hz=freq_hz,
    use_image_splat=True,  # default: same deposition as calcflow
)
```

## Pipeline

1. **`catalog_to_healpix`** — optional Healpix map for Mollweide / storage (nearest deposit).
2. **`healpix_to_theoretical_image`** (default) — **`theoretical_sky_point_plane`** image splat + JAX PSF (matches calcflow sources).
3. **`healpix_to_j2000`** — healpy sample of a Healpix map only when ``use_image_splat=False``.

Use **nside ≥ 2048** for ~2 arcmin LWA pixels.

Data: `../data/vlss_flux_healpix_nside2048.fits`

## Related

- `image-plane-correction-ovrolwa` — `theoretical_sky_beam_function` (direct image-plane splat).
- `../testdir/benchmark_theoretical_sky.py` — original vs Healpix path timing.

## Dev

```bash
conda activate /opt/devel/peijin/solarml
source testdir/jax_env.sh
pip install -e ./healpix-orth-skymodel
```

Dependencies: `numpy`, `astropy`, `healpy`, `jax` (PSF FFT only).
