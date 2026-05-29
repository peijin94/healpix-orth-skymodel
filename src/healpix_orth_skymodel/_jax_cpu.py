"""Force JAX to CPU before any jax import in this package."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
