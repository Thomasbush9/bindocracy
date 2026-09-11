# AlphaFold 3

Image: `images/af3.sif` (4.1 GB)
Definition: `containers/af3.def`
Repository: `alphafold3/` (google-deepmind/alphafold3)

A registered tool: `tool: af3`, see `examples/af3.example.yaml`. On the labelled
Nipah-G set it scores **0.693** — above Protenix and Boltz-1, well below
Boltz-2, ESMFold2 and Chai-1. Worth stating plainly, because the name invites
the opposite expectation.

## The weights are not in the image

`WEIGHTS_TERMS_OF_USE.md` restricts distribution, so baking `af3.bin.zst` into
a SIF would make every copy of that file a redistribution. They are bound
read-only from `af3_models/` instead, and the image's `%test` asserts they are
**absent** so a later build cannot quietly embed them.

Use is non-commercial only, with further restrictions in §3 of the terms.

Parameters: 1,020,545,840 bytes, sha256 recorded in
`af3_models/af3.bin.zst.sha256`, fetched 2026-09-09 from the public URL the
repository README names.

## No genetic databases

AlphaFold 3's own data pipeline wants ~630 GB of BFD, UniRef and MGnify. The
campaign already has a ColabFold alignment, and AF3 accepts it inline as
`unpairedMsa` (`docs/input.md:206`), so the driver always passes
`--norun_data_pipeline` and hands it the campaign a3m.

That is not a config option, deliberately: making it one would let a run search
a database that is not installed and fail late.

HMMER is still built into the image, because `run_alphafold.py` imports the
pipeline module at start-up even when told not to run it.

## One trap

AF3 and every mosaic scorer are JAX programs, so a workflow running both
inherits mosaic's `JAX_COMPILATION_CACHE_DIR=/jax_cache` — a path that exists
only inside `mosaic.sif`. AF3 then dies with
`NOT_FOUND: /jax_cache/xla_gpu_per_fusion_autotune_cache_dir`, naming a
directory nobody configured. The launch pins the cache to the task's own work
directory, which also earns it back across designs of varying length.
