# Weights and offline-readiness report

This report describes the images currently installed in `images/`. "Embedded"
means the required checkpoints were copied into the SIF at build time. Runtime
caches may still be created in a writable directory even when weights are
embedded.

| Image | Weight status | Runtime network needed? | Notes |
|---|---|---|---|
| `genie3.sif` | Embedded | No for normal runs | Genie 3, ColabFold/AF2 multimer, and ProteinMPNN assets are included. |
| `freebindcraft.sif` | Embedded | No for normal runs | AF2/ColabDesign and MPNN assets are included. PyRosetta is intentionally absent. |
| `caliby.sif` | Embedded for supported modes | No | Caliby variants, sidechain packer, ProteinMPNN, and Protpardelle assets are included. AF2 self-consistency weights are intentionally excluded. |
| `boltzgen.sif` | Embedded | No | Standard BoltzGen checkpoints/data and Hugging Face assets are included. |
| `protein_hunter.sif` | Embedded for supported modes | No | Boltz2, Chai/Chai-ESM, and LigandMPNN/ProteinMPNN assets are included. AlphaFold3 and PyRosetta are intentionally excluded. |
| `proteina_complexa.sif` | Embedded | No | Complexa, AF2, ESM2, RF3, ProteinMPNN, LigandMPNN, Foldseek, and pipeline data are included. |
| `pxdesign.sif` | Embedded | No for inference; yes for live MSA search | PXDesign, Protenix, AF2, ProteinMPNN, and CCD data are included. `prepare-msa` contacts the configured MSA service; precompute MSAs for offline jobs. |
| `switchcraft.sif` | Embedded | No for normal runs | Boltz and LigandMPNN assets are included. |
| `mosaic.sif` | External by design | No after the external cache is populated | Use `mosaic-exec.sh`; it binds Boltz, AF2, Protenix, and Hugging Face weight directories into the image. |
| `rfd.sif` | **Embedded** (corrected 2026-08-27) | No | All nine checkpoints ship at `/app/RFdiffusion/models`. **No `/models` bind is needed.** |

## Corrections

**`rfd.sif` weights are embedded, not external.** This table previously said the
opposite, as does `CONTAINER_WEIGHTS_REPORT.md`. Both were describing
`RFDiffusion/rfdiff.def` — the ESMFold2 variant that was never built — rather
than the installed image, which is a straight conversion of
`rosettacommons/rfdiffusion` and ships `Complex_base_ckpt.pt`,
`Complex_beta_ckpt.pt`, `Base_ckpt.pt`, `Complex_Fold_base_ckpt.pt`,
`InpaintSeq_ckpt.pt`, `InpaintSeq_Fold_ckpt.pt`, `ActiveSite_ckpt.pt`,
`Base_epoch8_ckpt.pt` and `RF_structure_prediction_weights.pt`. Verified by
listing the image and by a successful design run. Treat the SIF and the adjacent
`.def` as two separate artifacts.

**`genie3.sif` has the weights but cannot use them on a GPU.** Genie 3,
ColabFold/AF2 and ProteinMPNN assets are all present, but the image's JAX is
CPU-only (`jax-cuda12-plugin` missing), so the AF2 evaluation stage silently
runs on the CPU. Weight completeness and runnability are different questions;
this row says nothing about the latter. See
[`known-issues.md` §2.3](known-issues.md).

## Decision feedback

Embedding weights is the safest default for a stable shared suite: jobs are
offline, immutable, and easier to reproduce. Its costs are very large SIFs,
slow rebuilds, and duplicated weights across images.

External weights are preferable when several images share very large models or
checkpoints change frequently. They reduce image size but make each run depend
on a second versioned artifact and exact bind paths. Mosaic is intentionally in
this category; `rfd.sif` currently is as well.

For external weights, record a manifest with file paths, sizes, and checksums:

```bash
find /path/to/weights -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > weights.sha256
```

Do not silently download checkpoints during a production GPU job. Populate and
verify caches first, then use offline mode where supported.

## Model expansion staging — 2026-09-08

This is asset preparation, **not** a switch of the active scoring protocol.
`mosaic.sif` is not rebuilt in this step. Chai's image will be built by the user
after inspecting its definition; the existing scoring configs are unchanged.

### Protenix Base: acquired and converted for the existing Mosaic image

- Model: `protenix_base_default_v1.0.0` (368,484,735 Torch parameters).
- Directory: `binder_design/mosaic_setup/weights/protenix/`, already bound
  read-only to `~/.protenix` by `mosaic-exec.sh`.
- New files: `protenix_base_default_v1.0.0.pt` (1,475,950,125 bytes),
  `.eqx` (1,474,291,978 bytes), `.skeleton.pkl` (39,090 bytes), and
  `.manifest.json` with URLs, sizes, SHA256, backend revision and conversion
  details. Existing Mini and shared reference data were not replaced.
- Official checkpoint:
  <https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt>.
- Conversion used the image's existing `protenij` backend at
  `d48a0b0456b582cef10f1bacfb19941a04f7e8c8`, not a newly installed dependency.
  Strict Torch state loading and an exact converted-array round trip passed.
  A second scheduled CPU smoke loaded the final published Base filenames with
  socket connections and backend downloads disabled.
- Acquisition scripts:
  `mosaic_setup/mosaic/singularity/fetch-protenix-base.{py,sbatch}`.
  The fetch refuses to overwrite existing converted artifacts. It needs CPU
  memory for conversion, not a GPU.

No Mosaic rebuild is required to load these assets. **The bindocracy scoring
driver still selects Mini**; selecting Base in scoring is a separate change for
the next evaluation. GPU folding quality/performance has not been measured here.

### Chai-1: downloaded build inputs and inspectable definition

- Checkout: `binder_design/chai-lab/`, official release `v0.6.1`, revision
  `8d5ac0f93e9b6ea4c3a6545c253a6381c0f3694b`.
- Assets: `chai-lab/downloads/`, the upstream default layout. All eight
  inference assets are present: six `models_v2/*.pt` modules, the traced
  ESM2-3B embedding model, and `conformers_v1.apkl`. Total: **6,979,394,752 bytes**.
- `assets-manifest.json` records official HTTPS URLs, sizes and locally
  computed SHA256; `assets.sha256` is the checksum list. These hashes identify
  the acquired bytes; upstream did not supply independent SHA256 values.
- `fetch_assets.py` downloads into partial files before publishing. Once a
  manifest exists, rerunning verifies it rather than blessing changed bytes.
  `source.sha256` separately pins the upstream source copied into the image.

Inspect **`chai-lab/chai1.def`** before building. Its base CUDA 12.4.1 image is
pinned by digest; CPython 3.11.11, Torch 2.5.1+cu124 and Python dependencies are
pinned, with dependency hashes in `container-requirements.lock`. Ubuntu package
security updates remain build-time resolved.

The definition embeds the already-downloaded assets in
`/opt/chai-lab/downloads`; no external weight bind is needed. `chai1_offline.py`
routes inference through local-only asset lookup, refuses network MSA/template
flags and blocks Requests HTTP calls. Incidental caches use private writable
scratch rather than the read-only asset tree. Keep the documented `--cleanenv`
launch to avoid host Python/cache overrides.

After inspection, build **from the Chai clone root**, in a suitable scheduled
CPU build allocation with network access for OS/Python packages:

```bash
singularity build --fakeroot ../images/chai1.sif chai1.def
```

After building, on a GPU allocation with writable input/output storage:

```bash
singularity run --nv --cleanenv \
  --bind /absolute/work:/work ../images/chai1.sif \
  fold /work/input.fasta /work/output --msa-directory /work/msas
```

`/work/msas` must contain Chai's sequence-hash-named `.aligned.pqt` files;
an A3M path cannot be passed directly. The image exposes upstream conversion:

```bash
singularity run --cleanenv --bind /absolute/work:/work ../images/chai1.sif \
  a3m-to-pqt /work/a3ms --output-directory /work/msas
```

Each conversion input directory is for **one query sequence**. Upstream infers
source databases from filenames and defaults unknown names to UniRef90; retain
accurate source metadata when adapting ProtForge's output rather than assuming
an arbitrary combined A3M already carries that information.

Verification before build: all eight asset hashes/sizes and pinned source
hashes passed; the exact locked Python/Torch/Chai environment installed and
passed dependency/import checks; the offline CLI help, a two-sequence local
A3M-to-Parquet conversion, and rejection of the network-MSA flag were exercised.
Definition file inputs exist and its shell sections pass syntax checks.
**No Chai SIF was built and no GPU fold was run.** The definition's `%test`
checks imports and asset integrity without requiring a GPU; run a real offline
GPU fold after building. Chai is not yet integrated into the bindocracy scorer.

### DeepMind AlphaFold 3: source only

- Checkout: `binder_design/alphafold3/`.
- Upstream: <https://github.com/google-deepmind/alphafold3>.
- Pinned detached revision: `c0f97eda2f1f482fd94d3a38bece18c7069b4a5c`.
- No DeepMind weights downloaded and no AF3 image built in this step.

**The existing OpenFold3 weights cannot be used as DeepMind AF3 weights.**
OpenFold3 is a separately trained implementation, not a filename-compatible
distribution of DeepMind's parameters. The local Mosaic OpenFold3 cache contains
its own Equinox tree and skeleton; DeepMind loads its named Haiku parameter
records with the layer shapes documented in
[`alphafold3/docs/model_parameters.md`](../../alphafold3/docs/model_parameters.md).
Renaming a checkpoint or changing its serialization does not establish model
compatibility or make it a DeepMind AF3 result. Continue using OF3 through the
existing OF3 backend.

The current upstream README provides a direct Google-hosted AF3 parameter
download. Actual AF3 inference would require those official parameters and a
separately validated runtime. Source is Apache-2.0; parameters and outputs have
separate [terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md),
including non-commercial-use and redistribution restrictions. Do not conflate
the source license with permission to redistribute a weight-embedded image.

The deferred scientific/workflow priorities are recorded in
[`scoring-stage.md` §10](scoring-stage.md#10-deferred-priorities-before-adding-new-losses).
