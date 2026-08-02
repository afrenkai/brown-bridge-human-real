# Retired observation-model experiments

The repository previously accumulated one-off StarGAN, continuous-VAE,
autoencoder, residual-diffusion, and SDE-refiner scripts. They were removed from
the canonical tree because experiments imported other experiments and the
renderer became conflated with the transport law. Git history retains the code.

The strongest final frozen-clock low-rank StarGAN residual run improved temporal
smoothness without changing endpoints:

- largest dog-probability jump: `0.4631 -> 0.4456`;
- DINO speed coefficient of variation: `1.1370 -> 1.0342`;
- active steps: `8.58 -> 9.47`;
- TV excess: `1.97e-4 -> 1.81e-4`.

It also slightly worsened semantic monotonicity and retained green/magenta pixel
artifacts. The conclusion was architectural: a low-rank residual can smooth a
fixed GAN path, but it cannot repair an observation manifold whose layerwise
style interpolation is itself artifact-prone.

Local results, if still present, are under ignored `outputs/` directories. They
are evidence for the pivot to a frozen diffusion VAE plus latent stochastic
bridge, not dependencies of the new implementation.
