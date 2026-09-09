# Chai-1 as a scorer

Chai-1 co-folds a binder against the target and reports confidence. It is a
second scorer beside the mosaic one, over the same design set, the same metric
registry and the same `metrics.jsonl` contract.

## Why a separate plugin rather than a seventh `ScoringModelName`

`ScorerConfig.model` is one plugin over six models because those six share an
API: mosaic's `StructurePredictionModel`, one container, one driver, one set of
knobs. The model is a *field* there precisely because nothing else differs.

Chai-1 shares none of that. Its own image, its own entry point, its own
alignment format, its own sampler vocabulary (`num_diffn_timesteps`, not
`sampling_steps`). Adding it to the literal would put a second container and a
second featurization path behind a field every other value reads from mosaic,
and `ACCEPTS_TARGET_MSA` would start meaning two different things depending on
which value was selected.

What is worth sharing is shared: `Chai1OutputAdapter` **is** the scorer's
adapter with a different tool name, because the seam between a driver and the
database is the JSONL file, not the container that wrote it.

## The alignment is the part that bites

Chai resolves an MSA by hashing the chain's sequence and looking for
`<sha256>.aligned.pqt` in `--msa-directory`. When the file is absent it logs a
warning and folds that chain single-sequence
(`chai_lab/data/dataset/msas/load.py:52`).

A warning in a GPU job's stdout is not a failure anybody sees. The run finishes,
produces confident-looking numbers, and nothing in the output records that the
alignment was never used — the same failure that made OpenFold3 and Protenix
incomparable for a month. So `preflight_chai1` resolves the filename by Chai's
own rule before anything is allocated and refuses the run if it is missing.

Chai wants `.aligned.pqt`, **not** the campaign's `.a3m`. Convert once per
target:

```bash
mkdir -p /tmp/a3m
cp .../DIO3.a3m /tmp/a3m/uniref90.a3m      # stem names the source database
singularity run --cleanenv images/chai1.sif \
    a3m-to-pqt /tmp/a3m --output-directory chai_msas/dio3_cut
```

The campaign's alignment is already converted at
`binder_design/chai_msas/dio3_cut/` (3,032 sequences,
`cbd2ba0f...aligned.pqt`).

Only the target needs one. Chai will warn about the binder chain having no MSA;
that is correct and matches every other scorer here — a de novo binder has no
homologs by construction.

## Metrics

Fourteen per candidate structure, all registered in `adapters/scoring.py`, all
stored as `chai1_<key>`:

| key | source |
|---|---|
| `aggregate_score` | Chai's own ranking composite |
| `complex_ptm`, `iptm` | `ptm_scores.complex_ptm` / `.interface_ptm` |
| `bt_iptm`, `tb_iptm`, `iptm_min` | `per_chain_pair_iptm[query, key]`, both directions |
| `binder_ptm` | `per_chain_ptm[binder]` |
| `complex_plddt`, `binder_plddt` | per-token pLDDT, sliced by chain |
| `bt_pae`, `tb_pae` | per-token PAE, sliced by chain |
| `has_clashes`, `n_clashing_chain_pairs` | `clash_scores`, off-diagonal only |
| `binder_intra_clashes` | `chain_chain_clashes[binder, binder]` |

Two of these exist because the first smoke run's output was wrong, and the
numbers said so:

**ipTM is directional and Chai's own is the optimistic one.**
`interface_ptm` is documented as the *max* TM score over chains
(`chai_lab/ranking/ptm.py:100`), and `per_chain_pair_iptm` is
`[query_chain, key_chain]`. On a real fold the two directions were 0.470 and
0.395 with `iptm` reporting 0.470. Storing only `iptm` keeps the flattering
half of a disagreement, so both directions and their minimum are stored, the
same reasoning that puts `ipsae_min` in the registry.

**The clash matrix has intra-chain counts on its diagonal.** A real fold gave
`chain_chain_clashes = [[17, 0], [0, 11]]` beside
`has_inter_chain_clashes = False`: seventeen clashes inside the target,
eleven inside the binder, none across the interface. Counting nonzero entries
without masking the diagonal reported every design as having a clashing chain
pair. The diagonal is now masked for the interface count and the binder's own
self-clashes are stored separately, where they are a real design liability
rather than a phantom interface problem.

`aggregate_score` is deliberately **not** stored as `rank_composite`.
`rank_composite` is mosaic's formula (`iptm + 0.5*tb_ipsae + 0.5*bt_ipsae`);
Chai's aggregate is its own. One column holding two definitions would look
joinable and would not be.

**Not emitted:** `iplddt` and the ipSAE family. iplddt needs interface residues
from coordinates, and ipSAE is mosaic's computation over its own PAE
convention. Emitting either from a different definition under the same registry
key is the silent-mismatch failure the registry exists to prevent.

Every structure is a replicate, stored as its own row. `num_trunk_samples ×
num_diffn_samples` of them per design — five by default, which is five times the
rows and roughly five times the walltime of a one-sample mosaic run.

## Chain order

The driver writes the target first and the binder second, so Chai names the
target chain **A** — the same letter `general.target.chain_id` uses, which is
what any later epitope mapping will be written against.

The driver assumes tokens are the two protein chains concatenated in that order
and asserts `n_tokens == len(target) + len(binder)`. Chai tokenizes one token
per residue for standard protein residues; if that stops being true the assert
fails the design rather than silently mis-slicing every per-chain metric.

## Protocol knobs

All required, none defaulted, for the reason `scorer/config.py` gives at length.
Chai's own defaults are `num_diffn_timesteps=200` and `num_diffn_samples=5` —
much larger budgets than the mosaic scorer's smoke runs used. `low_memory` is
excluded from `protocol_hash`: it trades speed for memory without changing the
prediction.

`num_trunk_recycles` is **not** normalised by `RECYCLING_OFFSET`. That table
exists because OpenFold3 and ESMFold2 add a trunk pass internally while Boltz
does not; Chai's count is its own. Comparing a Chai recycle budget to a Boltz
one is a judgement for whoever reads the numbers, not something a config can
launder into equivalence.

## What the container gives us that mosaic currently does not

Chai's eight inference assets are embedded in the image and mounted read-only,
so the container digest alone determines what ran. There is no `dev_source`
overlay to record, and no gap between `container_digest` and what actually
executed. That is the state `docs/scoring-stage.md` wants the mosaic scorer to
reach by rebuilding `mosaic.sif`.

## Running it

```bash
bindocracy config load --general general.yaml --model configs/chai1.yaml
```

then the ordinary Snakemake path. See `examples/chai1.example.yaml`.
