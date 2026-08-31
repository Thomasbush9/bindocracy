"""Build the command for one Protein-Hunter task.

The target enters as an argv value rather than a path, which is unique among
the tools here: Protein-Hunter reads no structure and opens no FASTA of its
own. The sequence on this command line is the one preflight read from the
campaign FASTA, and the manifest's digest of that file is what ties the two
together.

A driver is used for one reason: the ColabFold MSA cache has to be written into
the task's own save directory before the pipeline starts, and there is no flag
that would let the container do it.
"""

from __future__ import annotations

import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import read_single_fasta
from bindocracy.runs.inputs import TargetDigest
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.protein_hunter.config import ProteinHunterConfig

# The runscript's third mode: run the container's own interpreter on a script.
CONTAINER_PYTHON_MODE = "python"


def protein_hunter_launch_spec(
    general: GeneralConfig, model: ProteinHunterConfig, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """One task: the archived driver, seeding a cache and running the pipeline."""
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    task_dir = run_dir / task.directory
    # The archived driver, not the authored one: the working tree may have
    # moved on since this run was planned.
    driver = run_dir / manifest.provenance["driver"].path
    sampling = model.sampling

    argv = [
        "singularity", "run", "--cleanenv", "--nv",
        str(model.runtime.container),
        CONTAINER_PYTHON_MODE, str(driver),
        "--save-dir", str(task_dir),
        "--name", _run_name(manifest),
        # The target itself, not a path to it. Nothing downstream re-reads it.
        "--protein-seqs", _target_sequence(manifest),
        "--msa-mode", model.msa.mode,
        "--num-designs", str(sampling.trajectories_per_job),
        "--num-cycles", str(sampling.cycles),
        "--min-protein-length", str(sampling.min_binder_length),
        "--max-protein-length", str(sampling.max_binder_length),
        "--percent-x", str(sampling.percent_x),
        "--omit-aa", sampling.omit_aa,
        "--temperature", str(sampling.temperature),
        "--diffuse-steps", str(sampling.diffuse_steps),
        "--recycling-steps", str(sampling.recycling_steps),
        "--high-iptm-threshold", str(model.filters.high_iptm_threshold),
        "--high-plddt-threshold", str(model.filters.high_plddt_threshold),
    ]
    # The campaign epitope, mapped at planning and carried in the manifest so
    # the launch cannot re-derive it differently from what was recorded.
    contacts = str(manifest.workflow.get("contact_residues") or "")
    if contacts:
        argv += [
            "--contact-residues", contacts,
            "--contact-cutoff", str(model.contacts.cutoff),
            "--max-contact-filter-retries", str(model.contacts.max_retries),
        ]
        argv.append("--contact-filter" if model.contacts.filter else "--no-contact-filter")

    if model.msa.mode == "mmseqs":
        argv += [
            "--a3m", str(_msa_path(manifest)),
            "--max-seqs", str(model.msa.max_seqs),
        ]

    node_tmp = _node_tmp(model, manifest, task_id)
    return LaunchSpec(
        argv=tuple(argv),
        env=_protein_hunter_environment(node_tmp),
        # The runscript mkdirs its cache tree under TMPDIR on its first line,
        # under `set -eu`, so an absent TMPDIR kills the job before Python.
        mkdirs=(node_tmp, node_tmp / "protein-hunter-cache"),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs,),
    )


def _run_name(manifest: RunManifest) -> str:
    """What Protein-Hunter names its structure files after.

    Taken from the run's own workflow metadata so the names the adapter
    reconstructs and the names the tool writes come from one place.
    """
    name = manifest.workflow.get("design_name")
    if not name:
        raise ValueError(
            f"run {manifest.run_id} records no design_name; it was planned "
            "before the name was stored and cannot be launched from"
        )
    return str(name)


def _target_sequence(manifest: RunManifest) -> str:
    """The target, re-read from the FASTA this run digested, and checked.

    Everywhere else the target reaches a tool as a file the tool opens, and the
    manifest's digest of that file is the whole guarantee. Here it reaches the
    tool as a string in an argument vector, so the string is hashed and
    compared against what the run recorded designing against. Nothing else
    would notice a task launched with the wrong sequence.
    """
    digest = manifest.inputs.get("target_fasta")
    if digest is None:
        raise ValueError(f"run {manifest.run_id} digested no target FASTA")
    sequence = read_single_fasta(Path(digest.uri))

    recorded = manifest.target
    if recorded is not None and TargetDigest.of(recorded.name, sequence) != recorded:
        raise ValueError(
            f"run {manifest.run_id} designs against {recorded.name} "
            f"({recorded.length} aa), but {digest.uri} now reads as "
            f"{len(sequence)} aa. This tool passes the target on the command "
            "line, so this is the only place the substitution would show."
        )
    return sequence


def _msa_path(manifest: RunManifest) -> Path:
    digest = manifest.inputs.get("target_msa")
    if digest is None:
        raise ValueError(
            f"run {manifest.run_id} was planned without an alignment, so it "
            "cannot be launched in mmseqs mode"
        )
    return Path(digest.uri)


def _node_tmp(model: ProteinHunterConfig, manifest: RunManifest, task_id: int) -> Path:
    """Node-local scratch for one task, unique so two jobs cannot collide."""
    return (
        model.runtime.node_tmp_root
        / f"bindocracy-protein-hunter-{manifest.run_id[:8]}-{task_id:04d}"
    )


def _protein_hunter_environment(node_tmp: Path) -> dict[str, str]:
    """Everything the image derives from TMPDIR, moved onto node-local disk.

    `PROTEIN_HUNTER_RUNTIME_CACHE` defaults to `${TMPDIR}/protein-hunter-cache`
    and XDG_CACHE_HOME, HF_HOME and TORCH_HOME are all derived from it, so
    TMPDIR alone decides where four caches land. The model weights are in the
    image, so nothing here is downloaded -- these are incidental framework
    caches that should not be on Lustre or on the home quota.
    """
    environment = {
        "TMPDIR": str(node_tmp),
        "SINGULARITYENV_TMPDIR": str(node_tmp),
        "SINGULARITYENV_PROTEIN_HUNTER_RUNTIME_CACHE": str(
            node_tmp / "protein-hunter-cache"
        ),
        # The host's RHEL CA path does not exist inside this image.
        "SINGULARITYENV_SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "SINGULARITYENV_CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
    }
    # `--cleanenv` drops the variable Slurm sets to name the allocated device.
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = devices
    return environment
