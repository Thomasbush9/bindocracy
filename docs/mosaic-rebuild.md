# Rebuilding `mosaic.sif`

**Proposal, 2026-09-09.** Not applied. It changes what a scoring run means, so
it is a decision rather than a chore.

## Why

Every scoring run today is split across two artifacts. `container_digest` names
`mosaic.sif`; `dev_source_sha256` names a tree bound over it at run time. The
image alone does not determine the result, and the image alone is what a reader
a year from now will have.

Measured on 2026-09-08, image versus the checkout at
`mosaic_setup/mosaic/src`:

| | |
|---|---|
| files in the image / in the checkout | 94 / 96 |
| files that differ | **12** |
| `models/af2_msa.py` | **absent from the image** |
| `grep -rl require_msa /opt/mosaic/src/` | **no matches** |
| `models/af2.py:391` | `assert not c.use_msa, "AF2 interface does not support MSA yet"` |

The MSA routing exists in the image as dead code: `msa.py` is there and nothing
imports `require_msa`. That is the defect that had OpenFold3 and Protenix
quietly querying `api.colabfold.com` while the other models read the campaign
alignment.

Two capabilities are gated on this rebuild:

- **AF2 with the target MSA.** Verified working against the checkout on
  2026-09-09 — 3/3 designs, complex and monomer, no assertion. Against the
  shipped image it still asserts. `ACCEPTS_TARGET_MSA["af2"]` is now `True`,
  which is only true with `dev_source` set or after this rebuild.
- **Retiring `runtime.dev_source` entirely**, so one digest describes a run.

## Blocking prerequisite: the source is uncommitted

`git status` in `mosaic_setup/mosaic` shows **nine modified files under
`src/`**, and they are exactly the ones the harness depends on:

```
 M src/mosaic/msa.py                 M src/mosaic/models/of3.py
 M src/mosaic/structure_prediction.py  M src/mosaic/models/protenix.py
 M src/mosaic/models/boltz1.py       M src/mosaic/models/promera.py
 M src/mosaic/models/boltz2.py       M src/mosaic/models/opendde.py
 M src/mosaic/models/esmfold2.py
```

Building now would bake a state that exists on one filesystem and nowhere else.
The commit hash recorded in the image's labels would be `cb2213e`, which does
**not** contain these changes — the label would be a lie, which is worse than
no label.

**Commit them first**, on `singularity-cluster`. Note the `af2-msa` branch is
not the answer here: it is *behind* `singularity-cluster`, lacking the 90-line
`require_msa` routing in `msa.py` and the `af2_msa_test.py`. The MSA work is
already merged into the branch we build from.

## The changes to `singularity/mosaic.def`

The `%files` section already copies `src`, so a rebuild picks up `af2_msa.py`
and the routing with no edit. Two additions are worth making, both about
stopping this class of failure from recurring silently.

### 1. Record what went in

```
%labels
    SourceCommit   <git rev-parse HEAD>
    SourceBranch   singularity-cluster
    SourceTreeSha  <sha256 of sorted relpath:filedigest over src/**/*.py>
```

The tree digest is what `tools/scorer/preflight.py::source_tree_digest`
already computes for `dev_source`. Putting the same number in the image lets a
run assert that the image it used and the tree it was built from are the same
thing, rather than trusting a branch name.

### 2. Make the build fail if the MSA routing is dead

This is the important one. The current image is not broken in a way anything
notices — it imports, it runs, it produces numbers. Add to `%test`:

```sh
# The routing must be live, not merely present. The shipped image has msa.py
# and zero importers, which is how six models came to disagree about which
# alignment they were folding against while every one of them succeeded.
for module in of3 protenix esmfold2 boltz1 boltz2 promera opendde; do
    grep -q "require_msa" /opt/mosaic/src/mosaic/models/$module.py || {
        echo "FAIL: models/$module.py does not route through require_msa"; exit 1; }
done

# AF2's MSA path must be present, not the old assertion.
test -f /opt/mosaic/src/mosaic/models/af2_msa.py || {
    echo "FAIL: models/af2_msa.py missing; AF2 cannot take a target MSA"; exit 1; }
grep -q "does not support MSA yet" /opt/mosaic/src/mosaic/models/af2.py && {
    echo "FAIL: af2.py still carries the no-MSA assertion"; exit 1; }

# The server fallback must be off by default.
/opt/mosaic/.venv/bin/python -c "
from mosaic.msa import ALLOW_SERVER_ENV
import os; assert os.environ.get(ALLOW_SERVER_ENV, '0') == '0'
print('MSA server fallback off by default')"
```

A build that cannot satisfy these should not produce a SIF.

### 3. Nothing else changes

Base image, `uv sync --frozen`, the dm-haiku patch and the CA-bundle handling
all stay. This is a rebuild, not a redesign.

## After building

1. Point one scorer config at the new image with `dev_source` **unset**, and
   re-run the 3-design verification (af2 with MSA, protenix base, promera).
2. If it matches, drop `dev_source` from every config in `configs/scoring/`.
3. `runtime.dev_source` stays in the config *schema* — it is the right tool for
   the next time something needs to be tested ahead of an image — but no
   production run should set it.

## What this does not fix

`docs/scoring-stage.md` §10 items 1 and 2 are independent of the image: the
mosaic driver now saves structures (2026-09-09) but `readers.epitope` and
`readers.inverse_folding` are still accepted and silently dropped by the
launcher. Rebuilding does not touch either.
