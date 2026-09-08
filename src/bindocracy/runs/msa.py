"""Prepare the campaign alignment before planning any MSA-consuming run.

Use Mosaic's imported ProtForge SLURM script unchanged, in a private attempt
folder: that script deletes its search directory and replaces its output.
Never expose that output to scorers until the job and query validation succeed.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from bindocracy.config.models import GeneralConfig, ResourceConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    read_single_fasta,
    require_alignment_of,
)
from bindocracy.runs.inputs import digest_of
from bindocracy.runs.manifest import write_json_atomic


def prepare_target_msa(
    general: GeneralConfig,
    *,
    script: Path,
    image: Path,
    database: Path,
    resources: ResourceConfig,
) -> Path:
    """Reuse a matching A3M or wait for ProtForge's local GPU search.

    The explicit command is the opt-in. No search happens during config loading
    or DAG evaluation. Existing alignments are never overwritten. Failed jobs
    leave their archived inputs and logs beside the destination for diagnosis.
    """
    destination = general.target.msa
    if destination is None or not destination.is_absolute():
        raise ConfigPreflightError("set target.msa to an absolute output A3M path first")
    sequence = read_single_fasta(general.target.sequence_fasta)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode: deleting a lock file can admit a third writer while
    # a second process still waits on the old inode.
    with destination.with_suffix(destination.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            require_alignment_of(destination, sequence, described_as="existing target MSA")
            return destination

        script, image, database = script.resolve(), image.resolve(), database.resolve()
        for label, path in (("ProtForge MSA script", script), ("MSA image", image)):
            if not path.is_file():
                raise ConfigPreflightError(f"{label} not found: {path}")
        if not database.is_dir():
            raise ConfigPreflightError(f"MSA database directory not found: {database}")
        if resources.gpus != 1:
            raise ConfigPreflightError("the imported ProtForge MSA script uses exactly one GPU")

        attempt = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        fasta = attempt / "target.fasta"
        archived_script = attempt / "msa-search.sbatch"
        # Freeze both before queueing: editing the campaign during the wait must
        # not cause the search to use an unrecorded target or script.
        fasta.write_text(f">target\n{sequence}\n")
        shutil.copyfile(script, archived_script)
        generated = attempt / "target.a3m"
        log = attempt / "search.log"
        argv = [
            "sbatch",
            "--wait",
            "--parsable",
            "--export=ALL",
            f"--account={general.cluster.account}",
            f"--partition={general.cluster.default_partition}",
            "--gres=gpu:1",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.memory_gb}G",
            f"--time={resources.walltime}",
            f"--output={log}",
            f"--error={log}",
            f"--chdir={attempt}",
            str(archived_script),
        ]
        # Set named variables through the process environment, not a comma-
        # delimited --export value: filesystem paths can themselves have commas.
        search_env = {
            "TARGET_FASTA": str(fasta),
            "TARGET_NAME": "target",
            "MSA_OUT": str(attempt),
            "MSA_IMAGE": str(image),
            "MSA_DB": str(database),
        }
        provenance = {
            "target_fasta": digest_of(fasta).model_dump(mode="json"),
            "script": digest_of(archived_script, uri=str(script)).model_dump(mode="json"),
            # As for scoring runs, image/database bytes belong to the asset
            # catalogue; do not hash a multi-terabyte search database here.
            "image": str(image),
            "database": str(database),
            "argv": argv,
            "environment": search_env,
            "log": str(log),
        }
        record = attempt / "preparation.json"
        write_json_atomic(record, provenance)
        result = subprocess.run(
            argv,
            env={**os.environ, **search_env},
            capture_output=True,
            text=True,
            check=False,
        )
        provenance.update(
            returncode=result.returncode,
            submission_stdout=result.stdout.strip(),
            submission_stderr=result.stderr.strip(),
        )
        write_json_atomic(record, provenance)
        if result.returncode:
            raise ConfigPreflightError(
                f"target MSA search failed (exit {result.returncode}); "
                f"see {log} and {record}: {result.stderr.strip()}"
            )
        if not generated.is_file():
            raise ConfigPreflightError(f"target MSA search produced no A3M; see {log}")
        require_alignment_of(generated, sequence, described_as="generated target MSA")
        if read_single_fasta(general.target.sequence_fasta) != sequence:
            raise ConfigPreflightError("target FASTA changed during MSA generation; not publishing")
        provenance["alignment"] = digest_of(generated, uri=str(destination)).model_dump(mode="json")
        write_json_atomic(record, provenance)
        # Publish without clobbering an alignment created outside this command.
        # Both paths are on the same filesystem; link is atomic and exclusive.
        os.link(generated, destination)
        generated.unlink()
        return destination
