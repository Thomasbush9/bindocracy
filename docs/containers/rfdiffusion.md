# RFdiffusion

Image: `images/rfd.sif`
Repository: `RFDiffusion/`

This installed image is a direct conversion of
`rosettacommons/rfdiffusion`. Its runscript launches
`/app/RFdiffusion/scripts/run_inference.py` in the image's virtual environment.

The adjacent `RFDiffusion/rfdiff.def` describes a newer, custom ESMFold2
extension, but that definition was not used to build the installed SIF. Document
and use the installed image and definition as separate artifacts. Do not assume
`/opt/run_esmfold2.py` exists in this image.

## Two corrections (2026-08-27)

**This image does not run on H100.** It carries torch 1.12.1+cu116, whose
`arch_list` stops at sm_86 with no PTX to JIT forward from, so every CUDA op
fails with `no kernel image is available for execution on the device` — verified
down to a bare host-to-device tensor copy by
`bindocracy/launching_scripts/rfdiffusion/diagnose_h100.sbatch`. Submit it to
A100 (sm_80) until the image is rebuilt:

```bash
sbatch --partition=kempner --gres=gpu:nvidia_a100-sxm4-40gb:1 run_rfdiffusion.sbatch
```

**Checkpoints ARE embedded — the section below is wrong.** All nine ship at
`/app/RFdiffusion/models` and no `/models` bind is needed. The external-weights
description applies to `RFDiffusion/rfdiff.def`, the ESMFold2 variant that was
never built.

Two further traps, both in
[`known-issues.md`](../known-issues.md): the runscript `eval`s its arguments, so
Hydra's `${now:%Y-%m-%d}` breaks the command line; and without `ppi.hotspot_res`
the code silently selects the **monomer** checkpoint, so
`inference.ckpt_override_path=/app/RFdiffusion/models/Complex_base_ckpt.pt` is
mandatory for hotspot-free binder design.

## Prepare external checkpoints (obsolete — see above)

RFdiffusion checkpoints are not embedded. Put the upstream checkpoint files in
a versioned host directory that will be bound to `/models`; record checksums
beside it. The selected checkpoint depends on the design mode and Hydra
configuration.

## Unconditional generation example

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
SIF="$BINDER_ROOT/images/rfd.sif"
WORK=/absolute/path/to/rfdiffusion_run
WEIGHTS=/absolute/path/to/rfdiffusion_weights

mkdir -p "$WORK"
singularity run --cleanenv --nv \
  --bind "$WORK:/work" \
  --bind "$WEIGHTS:/models" \
  "$SIF" \
  inference.model_directory_path=/models \
  inference.output_prefix=/work/design \
  inference.num_designs=10 \
  'contigmap.contigs=[100-100]'
```

This asks for ten 100-residue backbones. Start with one design for a smoke test.

## Motif scaffolding pattern

Bind the input PDB into the image and describe the fixed motif plus generated
segments with RFdiffusion's contig syntax:

```bash
INPUT=/absolute/path/to/motif.pdb

singularity run --cleanenv --nv \
  --bind "$WORK:/work" \
  --bind "$WEIGHTS:/models" \
  --bind "$INPUT:/work/motif.pdb:ro" \
  "$SIF" \
  inference.model_directory_path=/models \
  inference.input_pdb=/work/motif.pdb \
  inference.output_prefix=/work/scaffold \
  inference.num_designs=1 \
  'contigmap.contigs=[10-40/A10-25/10-40]'
```

The contig above is only a syntax example; replace chain/residue ranges with the
actual motif. For binder design, partial diffusion, active-site models, or
symmetry, follow the checked-out RFdiffusion documentation and use the matching
checkpoint. Quote Hydra list expressions so the shell does not interpret them.

## Verify image provenance

This image and its adjacent definition currently differ. Before changing or
rebuilding it, save the installed provenance:

```bash
singularity inspect --runscript "$SIF"
singularity inspect --deffile "$SIF" > rfd-installed.def.txt
sha256sum "$SIF" "$WEIGHTS"/* > rfd-runtime.sha256
```
