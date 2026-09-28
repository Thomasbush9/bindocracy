# Adding a custom optimizer

How to get from "I have an idea for improving designs" to children in the
database, in the order the work actually happens.

[custom-optimization.md](custom-optimization.md) is the **contract**: the exact
rows, the six rejection rules, why `loss_models` exists. Read it once. This
page is the **procedure**, and it assumes you have.

The unit you build is a **kit**: a directory holding a manifest and an
entrypoint. Everything true of the optimizer wherever it runs goes in the
manifest; everything true only of your machine stays in the campaign config.
That line is the whole design, and the test for which side a field falls on is:
*would a second site have to change it?*

```
mosaic-af2-refine/
|-- kit.yaml        what it is, what it needs, what it costs
`-- optimizer.py    the entrypoint
```

Four worked kits ship under `examples/kits/`; inspect their declarations before reuse.

---

## 1. Write the entrypoint

Start from `tests/fixtures/optimize/point_mutate.py`. It is the shortest thing
that exercises every part of the contract and it is covered by the test suite,
so it is the one example that cannot silently rot.

```python
from bindocracy_io import RejectCandidate, run_optimization

def optimize(parent, context, args):
    ...
    yield {"sequence": child, "metrics": {"loss": value}}

if __name__ == "__main__":
    run_optimization(optimize)
```

The helper owns the transport: it parses `--inputs`, `--outputs` and
`--context`, loads the context once, validates that every metric is a finite
number, and **attaches `parent_index` and the child ordinal itself**. You
cannot name a parent outside your shard and cannot emit a duplicate
`(parent_index, child)` pair, because you never write either.

Three things are still yours, and each is a place scripts go wrong.

**Distinguish a refusal from a bug.** Raise `RejectCandidate` when *this
parent* cannot be optimized and the run should carry on. Let everything else
propagate. Catching broadly is tempting and it is how a broken optimizer comes
to look like a biological result: a version of `mosaic-af2-refine` with a
two-line unpacking bug wrapped every parent in `except Exception`, wrote four
tidy refusal rows, and produced a run that read as "AF2 declined to improve
these binders" when it had never run a gradient step. Keep the caught set
narrow enough to name, one clause per reason.

**Report `start_loss` in the same units as `loss`.** A final loss means
nothing without the number it started from, and only if the two subtract. The
same optimizer once reported `start_loss` around 12 beside a `loss` around
-1.3, because one was the nine-term training objective and the other the
ranking objective. Both numbers were correct and their difference was
meaningless.

**Write artifacts relative to `context["structure_dir"]`.** An absolute path is
rejected so the run directory can be moved. `known-issues.md` records that
having already cost this campaign a day.

A script that reads and writes the JSONL rows by hand is still valid and always
will be, which is what lets a kit wrap a binary or a script in another
language. `examples/kits/surface-tuner/` is kept on that path on purpose, so
the raw contract keeps a user. Use the helper unless you cannot.

---

## 2. Write the manifest

```yaml
kit_schema: 1
name: example-optimizer
version: 1.0.0
kind: optimizer
summary: A reviewed optimizer with a portable execution contract.
entrypoint: optimizer.py
python: ">=3.12"
requires: []
inputs: [sequence]
loss_models: []
capabilities: {changes_length: false, max_children: 1}
metrics:
  diagnostic: {direction: none, description: A reported diagnostic.}
cost: {gpu: none, compile_seconds: 0, per_parent_seconds: 0}
args: {}
dependency_args: {}
```

This illustrates the schema, not a measured workload. Export the authoritative
JSON Schema with `bindocracy --json config schema --kind kit`. Unknown fields,
duplicate YAML keys, aliases, unsafe paths and non-finite numeric values are
refused. Optional `parameters` declare types, required values, enums and numeric
bounds; `args` holds defaults. Without an explicit declaration, a default's
scalar type constrains overrides. Boolean true emits a flag; false omits it.
`inputs`, `outputs` and `context` are reserved harness arguments.

**The approved manifest owns inputs, metric declarations and `loss_models`.**
The resolver copies them into the generated optimizer configuration. Campaign
requests must not restate these fields, the entrypoint, arguments or driver.
Reviewing and approving the complete bundle digest is the explicit affirmation;
a matching name or version alone is not approval. An empty `loss_models` list
remains a substantive author claim, not independent evidence.

**Cost metadata is an author claim, not qualification.** Put benchmark context
in `cost.measured_on`; an empty mapping is unmeasured. The current resolver
neither benchmarks a kit nor derives a scheduler reservation from these fields.
Campaign-owned resources remain explicit.

### Dependencies

Name what you need. Do not name where it lives.

```yaml
requires:
  - name: mosaic
    kind: container
    version: ">=2026.08.18"
    verify: {import: mosaic.models.af2_msa}
    why: AF2 multimer with a per-chain MSA.
```

Resolution checks declared files and captures local dependency identity without
importing modules, starting containers, installing packages or contacting a
store. `verify.import`, dependency versions and container Python compatibility
remain **unknown** until independently exercised; a successful parse is not
runtime qualification.

| `kind` | identity recorded by the resolver | supplied by |
|---|---|---|
| `container` | filesystem-stat identity, not a content digest | the site |
| `weights` | filesystem-stat inventory and required-file presence | the site |
| `source_overlay` | complete file-content inventory | the site |
| `harness_module` | declared files plus actual helper and driver hashes | the harness |

Mark optional requirements explicitly. `bindocracy-io` is harness-owned;
bindings cannot replace it with a publisher's or site's copy. At most one
container may declare `provides: interpreter`; it requires an explicit
interpreter binding. Other container dependencies can be passed through
`dependency_args`. The supported runtime overlay target is
`/opt/mosaic/src/mosaic`; other targets are refused, not silently dropped.

Dependency symlinks are currently refused, including links inside weight caches.
Materialize a regular-file tree before binding it. Content-hashed dependency
trees are limited to 512 MiB and 100,000 entries; stat inventories to 100,000
entries. Stat identities detect ordinary filesystem drift but do not
authenticate container or weight bytes.

---

## 3. Bind the dependencies on your machine

One file per site, campaign-owned. `examples/kits/bindings.example.yaml` is a
template.

```yaml
bindings_schema: 1
mosaic:
  kind: container
  path: /absolute/path/to/images/mosaic.sif
  interpreter: python
alphafold-params:
  kind: weights
  path: /absolute/path/to/weights/alphafold
```

Nothing here is portable, and nothing here belongs in a kit.

---

## 4. Inspect, review and resolve

```bash
uv run bindocracy --json kit inspect /path/to/kit
uv run bindocracy --json kit check /path/to/kit --bindings /path/to/bindings.yaml
```

Neither command executes publisher code. `check` reports unapproved execution
as unknown and exits 1 while retaining its JSON report. Missing or invalid
requirements are errors; unprobed runtime capabilities are never reported ready.

Review the full kit tree before authorizing it. Every regular file contributes
to its digest, including dotfiles and bytecode; there are no silent exclusions.
Keep credentials, site bindings, generated outputs and caches outside the kit.
Symlinks, special files and unsafe archive paths are rejected. A kit may contain
at most 100 MiB of file content, 4,096 files and 4,096 directories; `kit.yaml`
is limited to 1 MiB.

The campaign request contains campaign choices, not a second kit contract:

```yaml
schema_version: 1
name: reviewed-optimizer
tool: optimize
design_set: /absolute/path/to/design-set.json
sharding: {jobs: 1}
resources: {gpus: 1, cpus: 4, memory_gb: 16, walltime: "00:30:00"}
```

Use resource values appropriate to your independently validated workload.

```bash
uv run bindocracy --json kit resolve /path/to/reviewed/kit \
  --cache /path/to/kit-cache --bindings /path/to/bindings.yaml \
  --config /path/to/request.yaml --parameters /path/to/parameters.yaml \
  --output-dir /path/to/new-resolution --approve sha256:REVIEWED_DIGEST
```

`--bindings` and `--parameters` are optional when unnecessary. The output
directory must be new and outside the kit. Resolution writes `kit-lock.json`
and `optimize.yaml`, pinning the full cached bundle, effective parameters,
dependency evidence, harness files and reviewed digest. Inspect the lock with
the `kit-lock` JSON Schema. This approval authorizes reviewed code; it is not a
security attestation, runtime test or scientific validation.

Use the generated model YAML with ordinary `config check`, campaign readiness,
planning and collection. Campaign-plan approval remains a separate step.
Frozen-plan verification and launch recheck the lock, all bundle files and
dependency identities. Supporting files and sibling imports remain available
because execution uses the complete cached bundle, not a flattened entrypoint.
Keep that cache and the bound site assets: a run directory alone is not a
self-contained copy of every dependency. Independent raw-script configurations
remain supported without a kit.

Contract validation with `scripts/validate_optimizer.py` executes code. Run it
only after review; a successful contract fixture proves row handling, not
runtime compatibility or scientific quality. Containers and read-only mounts
are not a sandbox.

---

## 5. Publish and fetch immutable releases

```bash
uv run bindocracy --json kit publish /path/to/kit \
  --store /path/to/store --publisher your-lab
uv run bindocracy --json kit list --store https://kits.example.org
uv run bindocracy --json kit fetch your-lab/example-optimizer@1.0.0 \
  --store https://kits.example.org --cache /path/to/kit-cache \
  --digest sha256:INDEPENDENTLY_VERIFIED_DIGEST
```

Publishing writes a filesystem store; a web server can serve those same files
over HTTPS. Fetching requires an exact reference and independently trusted
digest, verifies the extracted inventory, and caches it at
`CACHE/DIGEST_HEX/kit`. Re-publishing identical content is idempotent; changing an
existing release requires a new version. Fetch does not authorize execution.
Pass its returned local `root` to `kit resolve`; cached kits work without the
source store. No lock contains a floating `latest` reference.

### Store integration contract

`catalog.json` has this shape:

```json
{
  "schema_version": 1,
  "releases": [{
    "reference": "your-lab/example-optimizer@1.0.0",
    "publisher": "your-lab",
    "name": "example-optimizer",
    "version": "1.0.0",
    "digest": "sha256:DIGEST_HEX",
    "archive": "blobs/DIGEST_HEX.zip"
  }]
}
```

The ZIP contains `kit.yaml` and the bundle's files directly at its root, without
a wrapper directory. The bundle digest is **not** the ZIP digest: it is SHA-256
of the UTF-8 canonical JSON mapping of relative POSIX file paths to
`sha256:HEX` file-content hashes. JSON keys are sorted, separators are `,` and
`:`, non-ASCII characters are retained as UTF-8, and no newline is appended.
Archive timestamps and compression do not affect identity.

Catalogs are capped at 4 MiB, archives at 110 MiB, and HTTPS operations use a
30-second socket timeout. Redirects must remain credential-free HTTPS.
Duplicate archive members, traversal, symlinks, encrypted entries and oversized
expanded bundles are refused.

**Publisher namespaces are claims, not authenticated identities.** This
filesystem/static-HTTPS implementation does not provide accounts, namespace
ownership, signatures, authenticated uploads, search UI or a trust registry.
A hosted store must supply its own publication authorization and trusted
digest distribution. Keep author claims, runtime qualification evidence and
scientific evaluations distinct; listing a release establishes none of them.

**Score the children with a model that is not in `loss_models`.** That is the
one rule that survives everything else on this page.

---

## Things that will bite you

**Editing a config re-plans every run in the same index.** The model YAML is a
declared input to `prepare_run`, so changing it re-runs preparation for every
entry that points at it, mints new run IDs over finished run directories, and
resubmits their jobs. Give a new run its own index entry, or expect to cancel.

**A dry run proves the rows, not the science.** Both bugs that cost this
campaign a GPU job passed every dry run: one was an arity mismatch in a mosaic
call, the other a units mismatch between two metrics. Run two or three parents
before a whole set, and read `start_loss` against `loss` when they land.

**Compilation is paid per binder length, not per task.** A shard spanning many
lengths pays it again at each one. The design set is ordered by length and
shards are contiguous precisely so it is paid as rarely as possible; a cost
model that ignores this will under-reserve a mixed shard.
