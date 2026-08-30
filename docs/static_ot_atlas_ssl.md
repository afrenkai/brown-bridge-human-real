# Static OT atlas: decision record

## Current decision

The long-term target is a family of source-only bijective maps between labelled
or discovered distributions. Each map should consume a continuous source state
and requested target condition. It must not retrieve a target example at
inference.

The map family should provide these structural contracts:

- identity for equal source and target conditions;
- an exact inverse for every ordered pair;
- composition through a shared canonical coordinate system; and
- retained source-specific coordinates that transport does not overwrite.

Optimal transport supplies finite training correspondences. It does not by
itself provide a decoder-compatible representation or population-level
bijection.

## Evidence retained from earlier text work

A 50,000-document OpenWebText T5 slot-codec experiment preserved enough meaning
for 75.8% held-out semantic retrieval. It failed the decoder gate:

- native frozen-T5 teacher-forced cross entropy: 0.0065;
- codec teacher-forced cross entropy: 3.2233;
- normalized autoregressive token edit distance: 0.8799;
- latent decode/re-encode cycle error: 0.0102.

Low latent cycle loss did not imply valid decoded text. The implementation was
removed from the active tree because it was a completed negative result with
undeclared dependencies. Git history preserves it.

A later SST-2 atlas entry point was also removed. It imported two modules that
were never committed, so a clean checkout could not run it.

## Gate for renewed language work

Before training a semantic atlas, a continuous language codec must pass held-out
checks for decoder cross entropy, autoregressive edit distance, off-manifold
interpolation, and semantic retrieval. Exact coordinate cycles are necessary
but not sufficient.

After the codec passes, promote the map into typed package modules rather than a
single experiment script. Add focused identity, inverse, and composition tests.
Keep deployment source-only and evaluate transport on disjoint held-out data.

The retained WikiText diffusion script is only a dynamic text-space baseline.
It is not the static atlas and does not satisfy the decoder contract.
