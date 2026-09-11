# Chai-1

Image: `images/chai1.sif` (10 GB)
Definition: `containers/chai1.def`
Repository: `chai-lab/` (upstream v0.6.1, `8d5ac0f9`)

Chai-1 co-folds a binder against a target. It is a registered tool, so a
campaign runs it like anything else: a `tool: chai1` model config and the usual
Snakemake path. See `examples/chai1.example.yaml`.

On the labelled Nipah-G set it scores **0.828**, level with ESMFold2 and second
only to Boltz-2 — see `docs/benchmark-nipah.md`.

## What is in the image

All eight inference assets are **embedded** (6.98 GB: six `models_v2/*.pt`, the
traced ESM2-3B embedder, `conformers_v1.apkl`), so the container digest alone
determines the result. Chai-1's assets are Apache-2.0, which permits that;
AlphaFold 3's are not, which is why `af3.sif` binds its weights instead.

Network MSA and template services are refused by `chai1_offline.py`, so a run
cannot silently fall back to a public server.

## The alignment

Chai wants `.aligned.pqt` keyed by a sha256 of the chain sequence, **not** an
a3m. Convert once per target:

```bash
mkdir -p /tmp/a3m && cp <target>.a3m /tmp/a3m/uniref90.a3m   # stem names the source DB
singularity run --cleanenv images/chai1.sif \
    a3m-to-pqt /tmp/a3m --output-directory chai_msas/<target>
```

`preflight` resolves the expected filename by Chai's own rule and refuses the
run if it is absent. That matters: Chai only *warns* when an alignment is
missing and then folds single-sequence, which looks like a quality result
rather than a plumbing failure.

Converted already: `chai_msas/dio3_cut/` (3,032 sequences) and
`chai_msas/nipah/` (64).

## Metrics

Fourteen per candidate, stored as `chai1_*`. Two exist because the first run
was wrong and the numbers said so:

- **ipTM is directional**, and Chai's own `iptm` is the *max* over chains. Both
  directions and their minimum are stored, so the flattering half of a
  disagreement is not the only record.
- **The clash matrix has intra-chain counts on its diagonal.** A real fold gave
  `[[17, 0], [0, 11]]` beside `has_inter_chain_clashes=False`; counting nonzero
  entries reported every design as having a clashing chain pair.

`aggregate_score` is deliberately not stored as `rank_composite` — that is
mosaic's formula and Chai's is its own.
