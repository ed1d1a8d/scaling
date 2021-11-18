"""
# From https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
kfreq = jnp.fft.fftfreq(side_samples) * side_samples
kfreq2D = jnp.meshgrid(kfreq, kfreq)
knrm = jnp.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

kbins = jnp.arange(0.5, side_samples, 1)
kvals = 0.5 * (kbins[1:] + kbins[:-1])

import scipy.stats as stats

Abins, _, _ = stats.binned_statistic(
    knrm.flatten(),
    (jnp.abs(dft) ** 2).flatten(),
    statistic="mean",
    bins=kbins,
)

Abins *= jnp.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

plt.plot(kvals / 2 / jnp.pi, Abins)
plt.yscale("log");
plt.xlabel("frequency");
plt.title("Spectral density");
"""
