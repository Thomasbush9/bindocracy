"""Offline, fail-closed materialization of existing target inputs.

No folding, alignment search, chain selection, renumbering or inferred pairing.
A complete private directory is validated before an exclusive atomic publish.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import warnings
from contextlib import suppress
from enum import StrEnum
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    read_single_fasta,
    require_alignment_of,
)
from bindocracy.tools.chai1.preflight import expected_pqt_basename
from bindocracy.tools.pxdesign.preflight import MSA_FILES


class TargetFormat(StrEnum):
    CHAI = "chai"
    PXDESIGN = "pxdesign"
    PDB = "pdb"
    CIF = "cif"


CHAI_SOURCES = ("uniref90", "uniprot", "bfd_uniclust", "mgnify")
MANIFEST = "materialization.json"
_PROTEIN = "ACDEFGHIKLMNPQRSTVWY"


def _digest(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _alignment(path: Path, sequence: str) -> list[tuple[str, str]]:
    require_alignment_of(path, sequence, described_as="target MSA")
    records: list[tuple[str, str]] = []
    header: str | None = None
    body: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(body)))
            header, body = line[1:], []
            if not header.strip():
                raise ConfigPreflightError("A3M has an empty record header")
        elif header is None:
            raise ConfigPreflightError("A3M sequence or metadata precedes its first header")
        else:
            body.append(line)
    if header is not None:
        records.append((header, "".join(body)))
    if not records or records[0][1] != sequence:
        raise ConfigPreflightError("A3M query must exactly match FASTA, without gaps or insertions")
    for index, (_, row) in enumerate(records, 1):
        # Dots have incompatible consumer meanings (Chai ignores them; Protenix
        # retains them). Refuse rather than silently rewrite either alignment.
        if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYXa-z-]+", row):
            raise ConfigPreflightError(f"A3M row {index} has empty or invalid sequence data")
        if len(re.sub(r"[a-z]", "", row)) != len(sequence):
            raise ConfigPreflightError(f"A3M row {index} has a mismatched match-state width")
        # Chai stores counts as uint8 and clamps to 255. Do not accept a lossy
        # handoff that appears faithful in the parquet but changes on ingestion.
        if any(len(run) > 255 for run in re.findall(r"[a-z]+", row)):
            raise ConfigPreflightError(f"A3M row {index} has an insertion longer than 255")
    return records


def _chai(path: Path, records: list[tuple[str, str]], source: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ConfigPreflightError(
            "format chai requires the runtime dependency pyarrow>=17"
        ) from exc
    # Chai 0.6.1 AlignedParquetModel: precisely these four string columns.
    # Empty pairing keys are its documented NO_PAIRING_KEY, not guessed taxa.
    table = pa.table(
        {
            "sequence": pa.array([row for _, row in records], type=pa.string()),
            "source_database": pa.array(
                ["query"] + [source] * (len(records) - 1), type=pa.string()
            ),
            "pairing_key": pa.array([""] * len(records), type=pa.string()),
            "comment": pa.array([header for header, _ in records], type=pa.string()),
        }
    )
    pq.write_table(table, path, compression="NONE", version="2.6")
    if not pq.read_table(path).equals(table):
        raise ConfigPreflightError("Chai parquet did not retain its input records")


def _read_structure(path: Path, kind: str):
    try:
        from biotite.structure.io import pdb, pdbx
    except ImportError as exc:
        raise ConfigPreflightError(
            "structure formats require the runtime dependency biotite>=1.7"
        ) from exc
    extra = ["atom_id", "b_factor", "occupancy", "charge"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if kind == "pdb":
            document = pdb.PDBFile.read(path)
            if document.get_model_count() != 1:
                raise ConfigPreflightError("structure materialization requires exactly one model")
            atoms = document.get_structure(model=1, altloc="all", extra_fields=extra)
        else:
            document = pdbx.CIFFile.read(path)
            if len(document) != 1:
                raise ConfigPreflightError(
                    "structure materialization requires exactly one CIF block"
                )
            site = document.block["atom_site"]
            if set(site["pdbx_PDB_model_num"].as_array(str)) != {"1"}:
                raise ConfigPreflightError(
                    "structure materialization requires exactly one model numbered 1"
                )
            atoms = pdbx.get_structure(document, model=1, altloc="all", extra_fields=extra)
    return document, atoms


def _identity(atoms) -> list[tuple]:
    fields = (
        "chain_id",
        "res_id",
        "ins_code",
        "res_name",
        "hetero",
        "atom_name",
        "element",
        "atom_id",
        "charge",
    )
    return list(zip(*(atoms.get_annotation(field).tolist() for field in fields), strict=True))


def _validate_structure(atoms, sequence: str, chain: str) -> None:
    import numpy as np
    from biotite.sequence import ProteinSequence

    if atoms.array_length() == 0 or not np.isfinite(atoms.coord).all():
        raise ConfigPreflightError("structure is empty or has non-finite coordinates")
    identities = _identity(atoms)
    altlocs = [str(value).strip(" .?") for value in atoms.altloc_id]
    keys = [(row[:6], alt) for row, alt in zip(identities, altlocs, strict=True)]
    if len(set(keys)) != len(keys):
        raise ConfigPreflightError("structure contains duplicate atom identities")
    residues: dict[tuple, str] = {}
    for row in identities:
        chain_id, number, insertion, name, hetero = row[:5]
        if chain_id == chain and not hetero:
            key = (number, insertion)
            if key in residues and residues[key] != name:
                raise ConfigPreflightError("structure has conflicting residue identities")
            residues[key] = name
    try:
        observed = "".join(ProteinSequence.convert_letter_3to1(name) for name in residues.values())
    except KeyError as exc:
        raise ConfigPreflightError("target chain contains a nonstandard protein residue") from exc
    if observed != sequence:
        raise ConfigPreflightError(
            f"structure chain {chain!r} does not exactly match target FASTA; no cropping or missing residues are inferred"
        )


def _structure(
    source: Path, source_kind: str, output: Path, kind: str, sequence: str, chain: str
) -> None:
    import numpy as np
    from biotite.structure.io import pdb, pdbx

    document, atoms = _read_structure(source, source_kind)
    _validate_structure(atoms, sequence, chain)
    if source_kind == kind:
        output.write_bytes(source.read_bytes())
        return
    # This is a coordinate-input converter, not a crystallographic metadata
    # translator. Refuse information the chosen writer cannot round-trip.
    if any(str(value).strip(" .?") for value in atoms.altloc_id):
        raise ConfigPreflightError("conversion would lose alternate-location identifiers")
    if atoms.box is not None:
        raise ConfigPreflightError("conversion of unit-cell metadata is not supported")
    if source_kind == "pdb":
        allowed = {"ATOM", "HETATM", "TER", "END", "MODEL", "ENDMDL", ""}
        if any(line[:6].strip() not in allowed for line in document.lines):
            raise ConfigPreflightError(
                "PDB contains metadata/connectivity records this conversion cannot retain"
            )
        atom_lines = [line for line in document.lines if line.startswith(("ATOM  ", "HETATM"))]
        if any(line[72:76].strip() for line in atom_lines):
            raise ConfigPreflightError("conversion would lose PDB segment identifiers")
        models = [line for line in document.lines if line.startswith("MODEL")]
        if models and any(line[10:14].strip() != "1" for line in models):
            raise ConfigPreflightError("conversion would change model numbering")
        target = pdbx.CIFFile()
        pdbx.set_structure(target, atoms)
        target.block["atom_site"]["id"] = pdbx.CIFColumn(atoms.atom_id)
        # PDB without COMPND metadata declares no molecular entity assignments.
        # Biotite otherwise invents one entity per chain; retain this uncertainty.
        target.block["atom_site"]["label_entity_id"] = pdbx.CIFColumn(
            np.zeros(len(atoms), dtype=int),
            mask=np.full(len(atoms), pdbx.MaskValue.MISSING),
        )
    else:
        if set(document.block) != {"atom_site"}:
            raise ConfigPreflightError(
                "mmCIF contains metadata/connectivity categories this conversion cannot retain"
            )
        site = document.block["atom_site"]
        if "label_entity_id" in site and any(
            value not in {"", ".", "?"} for value in site["label_entity_id"].as_array(str)
        ):
            raise ConfigPreflightError("PDB conversion cannot retain mmCIF entity assignments")
        from decimal import Decimal

        for field, places in (
            ("Cartn_x", 3),
            ("Cartn_y", 3),
            ("Cartn_z", 3),
            ("occupancy", 2),
            ("B_iso_or_equiv", 2),
        ):
            if field not in site:
                raise ConfigPreflightError(
                    f"mmCIF is missing {field}; no numeric values are inferred"
                )
            for raw in site[field].as_array(str):
                value = Decimal(raw)
                if not value.is_finite() or value != value.quantize(Decimal(10) ** -places):
                    raise ConfigPreflightError(f"PDB cannot retain the precision of mmCIF {field}")
        for suffix in ("asym_id", "seq_id", "comp_id", "atom_id"):
            author, label = f"auth_{suffix}", f"label_{suffix}"
            if (
                author not in site
                or label not in site
                or not np.array_equal(site[author].as_array(str), site[label].as_array(str))
            ):
                raise ConfigPreflightError(
                    "mmCIF author/label identities differ or are missing; PDB cannot retain both"
                )
        allowed = {
            "group_PDB",
            "id",
            "type_symbol",
            "label_atom_id",
            "label_alt_id",
            "label_comp_id",
            "label_asym_id",
            "label_entity_id",
            "label_seq_id",
            "pdbx_PDB_ins_code",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "occupancy",
            "B_iso_or_equiv",
            "pdbx_formal_charge",
            "auth_seq_id",
            "auth_comp_id",
            "auth_asym_id",
            "auth_atom_id",
            "pdbx_PDB_model_num",
        }
        if set(site) - allowed:
            raise ConfigPreflightError("mmCIF has atom fields PDB cannot retain")
        target = pdb.PDBFile()
        target.set_structure(atoms, hybrid36=False)
    target.write(output)
    _, restored = _read_structure(output, kind)
    _validate_structure(restored, sequence, chain)
    if _identity(atoms) != _identity(restored) or any(
        not np.array_equal(getattr(atoms, field), getattr(restored, field))
        for field in ("coord", "occupancy", "b_factor")
    ):
        raise ConfigPreflightError("conversion would lose atom identity or numeric precision")


def _publish(stage: Path, destination: Path) -> None:
    # Reserve exclusively before POSIX rename, which otherwise replaces existing
    # empty directories. Lustre does not necessarily support RENAME_NOREPLACE.
    # Readers see an empty reservation or the complete tree, never partial files.
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ConfigPreflightError(
            f"refusing to overwrite existing output directory: {destination}"
        ) from exc
    reserved = destination.stat()
    try:
        stage.rename(destination)
    except BaseException:
        # Preserve the original failure and any intervening files/replacement.
        with suppress(OSError):
            current = destination.lstat()
            if (current.st_dev, current.st_ino) == (reserved.st_dev, reserved.st_ino):
                destination.rmdir()
        raise


def materialize_target(
    general: GeneralConfig,
    *,
    output_dir: Path,
    formats: list[TargetFormat | str],
    msa_source: str | None = None,
) -> dict:
    """Produce an immutable directory, or reuse a byte-for-byte matching one."""
    selected = sorted({TargetFormat(value).value for value in formats})
    if not selected:
        raise ConfigPreflightError("choose at least one --format")
    destination = output_dir.absolute()
    if destination.is_symlink():
        raise ConfigPreflightError("output directory must not be a symlink")
    target = general.target
    sources = {"sequence_fasta": target.sequence_fasta.resolve()}
    if {"chai", "pxdesign"} & set(selected):
        if target.msa is None:
            raise ConfigPreflightError("alignment formats require an existing target.msa")
        sources["msa"] = target.msa.resolve()
    structure_sources = {}
    for kind in ("pdb", "cif"):
        if kind in selected:
            source = getattr(target, f"structure_{kind}")
            source_kind = kind
            if source is None:
                source_kind = "cif" if kind == "pdb" else "pdb"
                source = getattr(target, f"structure_{source_kind}")
            if source is None:
                raise ConfigPreflightError(f"format {kind} requires an existing target structure")
            structure_sources[kind] = (source.resolve(), source_kind)
            sources[f"structure_{source_kind}"] = source.resolve()
    source_records = {key: {"path": str(path), **_digest(path)} for key, path in sources.items()}
    sequence = read_single_fasta(target.sequence_fasta)
    if set(sequence) - set(_PROTEIN):
        raise ConfigPreflightError("target must use the 20 standard protein amino acids")
    records = _alignment(sources["msa"], sequence) if "msa" in sources else []
    if "chai" in selected:
        if msa_source is None:
            msa_source = sources["msa"].stem.removeprefix("hits_").removesuffix("_hits")
        if msa_source not in CHAI_SOURCES:
            raise ConfigPreflightError(
                f"Chai needs the real MSA source; set --msa-source to one of {', '.join(CHAI_SOURCES)} (no database is inferred from sequence)"
            )
    elif msa_source is not None:
        raise ConfigPreflightError("--msa-source applies only to format chai")
    request = {
        "schema_version": 1,
        "target": target.model_dump(mode="json"),
        "formats": selected,
        "msa_source": msa_source,
        "sources": source_records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        transformations = {}
        if "chai" in selected:
            directory = stage / "chai"
            directory.mkdir()
            _chai(directory / expected_pqt_basename(sequence), records, msa_source)
            transformations["chai"] = (
                "Chai-1 0.6.1 aligned parquet; unchanged A3M rows; no pairing keys"
            )
        if "pxdesign" in selected:
            directory = stage / "pxdesign"
            directory.mkdir()
            (directory / MSA_FILES[0]).write_text(
                "".join(f">{header}\n{row}\n" for header, row in records)
            )
            (directory / MSA_FILES[1]).write_text(f">query\n{sequence}\n")
            for name in MSA_FILES:
                _alignment(directory / name, sequence)
            transformations["pxdesign"] = (
                "full unpaired A3M plus query-only pairing A3M; no paired homologs inferred"
            )
        for kind, (source, source_kind) in structure_sources.items():
            try:
                from biotite import InvalidFileError
                from biotite.structure import BadStructureError

                _structure(
                    source, source_kind, stage / f"target.{kind}", kind, sequence, target.chain_id
                )
            except ImportError as exc:
                raise ConfigPreflightError(
                    "structure formats require the runtime dependency biotite>=1.7"
                ) from exc
            except ConfigPreflightError:
                raise
            except (
                ValueError,
                KeyError,
                IndexError,
                ArithmeticError,
                Warning,
                InvalidFileError,
                BadStructureError,
            ) as exc:
                raise ConfigPreflightError(
                    f"invalid or lossy {source_kind}-to-{kind} structure: {exc}"
                ) from exc
            transformations[kind] = (
                f"{source_kind}-to-{kind}; all atoms and author identities retained; no renumbering"
            )
        if {"pdb", "cif"} <= set(selected):
            import numpy as np

            _, pdb_atoms = _read_structure(stage / "target.pdb", "pdb")
            _, cif_atoms = _read_structure(stage / "target.cif", "cif")
            if _identity(pdb_atoms) != _identity(cif_atoms) or not np.array_equal(
                pdb_atoms.coord, cif_atoms.coord
            ):
                raise ConfigPreflightError(
                    "requested PDB and mmCIF disagree on atom identities or coordinates"
                )
        outputs = {
            str(path.relative_to(stage)): _digest(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {**request, "transformations": transformations, "outputs": outputs}
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if any(
            _digest(path) != {key: record[key] for key in ("sha256", "size_bytes")}
            for name, path in sources.items()
            for record in [source_records[name]]
        ):
            raise ConfigPreflightError("source changed during materialization; not publishing")
        reused = destination.exists()
        if reused:
            if (
                destination.is_symlink()
                or not destination.is_dir()
                or not (destination / MANIFEST).is_file()
            ):
                raise ConfigPreflightError(f"refusing unrelated output directory: {destination}")
            expected_files = {*outputs, MANIFEST}
            actual = {str(path.relative_to(destination)) for path in destination.rglob("*")}
            expected_tree = {str(path.relative_to(stage)) for path in stage.rglob("*")}
            if actual != expected_tree or any(path.is_symlink() for path in destination.rglob("*")):
                raise ConfigPreflightError(
                    "existing materialization contains missing, extra or symlinked paths"
                )
            for name in expected_files:
                if _digest(destination / name) != _digest(stage / name):
                    raise ConfigPreflightError(
                        f"existing materialization differs or was tampered with: {name}"
                    )
        else:
            _publish(stage, destination)
    return {
        "output_dir": str(destination),
        "manifest": str(destination / MANIFEST),
        "reused": reused,
        "paths": {name: str(destination / name) for name in outputs},
        "provenance": manifest,
    }
