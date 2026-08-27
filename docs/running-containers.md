# Running containers on the cluster

## Login node versus compute node

Use the login node for lightweight work only: edit YAML/JSON/FASTA files,
inspect an image, create a job script, and run `sbatch`. Run builds, inference,
design, validation, MSA search, and GPU checks on compute nodes.

Do not run `singularity run --nv ...` directly on a login node. For an
interactive test, first request an interactive compute allocation according to
the cluster's current policy, then run the same command there.

## Standard paths

```bash
export BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
export IMAGE_ROOT="$BINDER_ROOT/images"
```

Files under the lab's `/n` storage are normally visible inside these containers
at the same absolute path. Always launch from a writable working directory.
Explicit `--bind host:container` mappings are still useful when an application
expects a fixed in-container path, as Mosaic does for its external weights.

## Generic GPU job

Create a job such as `run-design.sbatch` and adjust the account/partition,
resources, image, and application command:

```bash
#!/bin/bash
#SBATCH --job-name=binder-design
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

IMAGE_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images
WORK=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/runs/my-run

mkdir -p "$WORK" "$WORK/logs" "$WORK/cache"
cd "$WORK"

export TMPDIR="$WORK/cache/tmp"
export XDG_CACHE_HOME="$WORK/cache/xdg"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

singularity run --cleanenv --nv "$IMAGE_ROOT/boltzgen.sif" --help
```

Submit from the login node:

```bash
mkdir -p logs
sbatch run-design.sbatch
```

`--nv` exposes the allocated NVIDIA GPU and driver to the image. It is required
for GPU applications but not for `singularity inspect`. The images deliberately
use CUDA 12-compatible frameworks; the host NVIDIA driver still must be new
enough for the specific CUDA 12 runtime in an image.

## Environment isolation

Host Conda modules and `PYTHONPATH` can contaminate a container process. Prefer
`--cleanenv`, then pass back only variables the application needs:

```bash
export SINGULARITYENV_MY_SETTING=value
singularity run --cleanenv --nv image.sif command
```

Singularity forwards variables named `SINGULARITYENV_NAME` as `NAME` inside the
container. Apptainer installations may also accept `APPTAINERENV_NAME`.

## Before a long campaign

Run a small smoke test on the same GPU class planned for production:

```bash
singularity inspect --helpfile image.sif
singularity test image.sif
nvidia-smi
singularity run --cleanenv --nv image.sif --help
```

Then submit one design, check the output and logs, and only then expand to a
large job array. A successful image build verifies installation, not scientific
inputs, writable output paths, GPU memory, or every optional pipeline branch.

## Reproducibility record

For every campaign, keep the following beside the outputs:

- the exact SIF path, size, and checksum (`sha256sum image.sif`);
- the submitted SLURM script and resolved configuration;
- input sequences/structures/MSAs and random seeds;
- the `singularity inspect --deffile image.sif` output;
- GPU type, job ID, and relevant log files.

SIFs are immutable, but filenames can be replaced. A checksum is the reliable
identity of the environment that produced a result.
