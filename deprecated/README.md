# Deprecated experiments

These entry points are retained as historical records, not active methods.
They either train rectified-flow/flow-matching fields, depend on the retired
stochastic-interpolant variants, or were one-off capacity checks. The active
examples use explicit entropic OT couplings and exact or Brownian-bridge
diffusion paths.

Moved here:

- `train_bidirectional_kl_vae_rectified_flow.py`
- `train_bidirectional_kl_vae_stochastic_interpolant.py`
- `overfit_kl_vae_stochastic_interpolant.py`
- `train_bidirectional_kl_vae_anchored_sde.py`
- `render_empirical_spatial_ot_sde.py`
- `train_distilled_terminal_sde.py`
- `augmented_reciprocal_cat_dog.py`
- `overfit_single_latent_pair.py`
- `train_stargan_style_lightsb.py`
- `train_layerwise_style_bridge.py`
- `viz_ot.ipynb`
- `viz_ot_legacy.ipynb`

The files are intentionally not part of the documented execution path. Their
old imports may assume the former `experiments/` layout.
