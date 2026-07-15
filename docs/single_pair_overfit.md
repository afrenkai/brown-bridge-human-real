# Quarantined single-pair memorization check

This experiment deliberately overfits exactly two AFHQv2 training images:

- cat: `flickr_cat_000002.png`
- dog: `flickr_dog_000002.png`

The complete 32x32 latent autoencoder is optimized for 3,000 MPS updates on
only these endpoints. The path is the deterministic conditional latent bridge

\[
Z_t=(1-t)E_\theta(x_{cat})+tE_\theta(x_{dog}),\qquad
X_t=G_\theta(Z_t).
\]

Weak interior anchoring, latent-cycle consistency, and pixel-space second
difference penalties prevent the unsupervised interiors from degenerating;
endpoint reconstruction remains the dominant objective.

Endpoint L1 decreases from 0.01880 to 0.01120 and endpoint edge L1 from
0.01760 to 0.01418. Over 24 microsteps, mean adjacent-frame L1 is 0.02526.
The CIFAR-trained DINO dog probability changes from 0.055 at the cat endpoint
to 0.953 at the reconstructed dog endpoint, crossing decisively near the
middle of the path.

This checkpoint is a capacity/memorization sanity check only. It must never be
used as evidence of generalization.

## Quarantine

The AFHQ root now contains `exclusions.json`, with the absolute path and SHA256
of both images. `AFHQDomainDataset` automatically loads that manifest and
filters exact paths and matching content hashes. Post-exclusion counts are:

- training cats: 5,064
- training dogs: 4,677
- test cats: 493
- test dogs: 491

The selected filenames are absent from all four constructed datasets. The
hash check also prevents the same bytes from re-entering under the same
filename in another split.
