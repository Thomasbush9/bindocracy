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
