"""Turning a scorer's numbers into metric rows.

DRAFT -- not wired into anything yet. See docs/scoring-stage.md.

Every generation adapter builds `MetricRecord`s by hand today, and each one
re-decides the same three things: what the metric is called, which direction is
better, and what to do when a measurement failed. For seven tools reporting
their own scores that duplication was tolerable. For a common scorer it is not,
because the whole value of the stage is that one number means one thing.

So the direction of a metric is declared once, in a registry, and a scorer that
emits an unregistered name is refused rather than defaulted to `none`. A metric
stored with the wrong direction sorts backwards, and nothing about the row says
so -- it is the same class of silent-wrong-answer bug the harness spends most
of its preflight budget on.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bindocracy.adapters.base import CollectionError
from bindocracy.store.records import (
    MetricDirection,
    MetricRecord,
    MetricStatus,
    stable_id,
)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """What one measurement is, independent of who measured it.

    `key` is the bare name a scorer reports (`iptm`); the stored name is that
    prefixed with the scoring model (`esmfold2_iptm`). Keeping the two apart is
    what lets the same specification describe six models without repetition,
    and what makes it impossible to store an unprefixed name by accident.
    """

    key: str
    direction: MetricDirection
    unit: str | None = None
    description: str = ""

    def stored_name(self, model: str) -> str:
        return f"{model}_{self.key}"


def _max(key: str, description: str, unit: str | None = None) -> MetricSpec:
    return MetricSpec(key, MetricDirection.MAX, unit, description)


def _min(key: str, description: str, unit: str | None = None) -> MetricSpec:
    return MetricSpec(key, MetricDirection.MIN, unit, description)


# The thirteen interface metrics the existing benchmark already computes, plus
# the four monomer ones. Names match `mosaic/benchmark/model_matrix.py` exactly
# so a stored column and a benchmark column are the same quantity; renaming
# them here would silently break every comparison against work already done.
#
# PAE metrics are stored raw -- lower is better -- rather than pre-negated. The
# benchmark orients them at plot time and a database that stores an already
# flipped number cannot be joined against one that does not.
COMPLEX_METRICS: tuple[MetricSpec, ...] = (
    _max("rank_composite", "iptm + 0.5*tb_ipsae + 0.5*bt_ipsae, the design-path composite"),
    _max("ipsae_min", "the lower of the two directional ipSAE values"),
    _max("bt_ipsae", "binder-to-target ipSAE"),
    _max("tb_ipsae", "target-to-binder ipSAE"),
    _max("iptm", "interface pTM over both chains"),
    _max("bt_iptm", "binder-to-target ipTM"),
    _max("binder_ptm", "pTM of the binder chain alone"),
    _max("binder_plddt", "mean pLDDT over binder residues"),
    _max("complex_plddt", "mean pLDDT over the whole complex"),
    _max("iplddt", "mean pLDDT over interface residues"),
    _min("bt_pae", "mean binder-to-target predicted aligned error", "angstrom"),
    _min("tb_pae", "mean target-to-binder predicted aligned error", "angstrom"),
    _min("ptm_energy", "pTM energy, lower is a better interface"),
    # Reported by co-folding backends that publish a whole-complex pTM and
    # their own ranking composite. `rank_composite` above is mosaic's formula
    # and is not interchangeable with a model's native aggregate, so the two
    # are separate keys rather than one column holding two definitions.
    _max("complex_ptm", "pTM over the whole complex"),
    _max("aggregate_score", "the model's own ranking composite, as it defines it"),
    # Chai-1 reports steric clashes alongside confidence. A clashing interface
    # can still score well on ipTM, so this is stored rather than folded into
    # a confidence number.
    _min("has_clashes", "1 if any inter-chain steric clash was detected, else 0"),
    _min("n_clashing_chain_pairs", "distinct chain pairs with inter-chain clashes"),
    _min("binder_intra_clashes", "steric clashes of the binder with itself"),
    # ipTM is directional: the score restricting to chain c against everything
    # else depends on which chain is the query. Chai's `interface_ptm` is the
    # MAX over chains, so it is optimistic by construction. Both directions and
    # their minimum are stored for the same reason `ipsae_min` exists.
    _max("tb_iptm", "target-to-binder ipTM"),
    _max("iptm_min", "the lower of the two directional ipTM values"),
)

MONOMER_METRICS: tuple[MetricSpec, ...] = (
    _max("mono_plddt", "mean pLDDT of the binder folded alone"),
    _max("mono_ptm", "pTM of the binder folded alone"),
    _min("mono_pae", "mean within-binder predicted aligned error", "angstrom"),
    _min("mono_rg", "radius of gyration of the binder alone", "angstrom"),
)

# Computed from the scored structure rather than predicted, so they carry no
# model prefix when stored -- see `epitope_metric_name`.
EPITOPE_METRICS: tuple[MetricSpec, ...] = (
    _max("epitope_coverage", "fraction of campaign hotspots the binder contacts"),
    _max("n_epitope_contacts", "hotspot residues contacted at the distance cutoff"),
    _max("n_interface_residues", "target residues contacted at the distance cutoff"),
    _min("epitope_offset", "mean distance from the binder to the nearest hotspot", "angstrom"),
)

# Sequence-only, no structure and no model. These are also the negative
# controls: the existing benchmark found four of six models failing to beat
# net charge alone, so storing them beside every real metric is what makes that
# check possible without re-running anything.
SEQUENCE_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("length", MetricDirection.NONE, "residues", "binder length"),
    MetricSpec("net_charge", MetricDirection.NONE, None, "net charge at pH 7"),
    MetricSpec("molecular_weight", MetricDirection.NONE, "dalton", "molecular weight"),
    _min("hydrophobic_fraction", "fraction of residues in AILMFWVY"),
    _min("n_cysteines", "free cysteine count, an expression liability"),
    _min("n_glycosylation_motifs", "N-X-S/T sequons"),
    _min("max_low_complexity_run", "longest single-residue run"),
)

_REGISTRY: dict[str, MetricSpec] = {
    spec.key: spec
    for spec in (*COMPLEX_METRICS, *MONOMER_METRICS, *EPITOPE_METRICS, *SEQUENCE_METRICS)
}


def spec_for(key: str) -> MetricSpec:
    """The specification for a metric key, or a refusal.

    Deliberately not forgiving. A scorer that reports something new adds it to
    the registry in the same commit, which is one line and forces the direction
    question to be answered by somebody who knows the answer.
    """
    try:
        return _REGISTRY[key]
    except KeyError:
        raise CollectionError(
            f"unregistered metric {key!r}; add a MetricSpec in adapters/scoring.py "
            "rather than storing a metric whose direction nothing records"
        ) from None


def registered_keys() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def epitope_metric_name(spec: MetricSpec, model: str) -> str:
    """Epitope metrics are prefixed by the model whose structure they measured.

    They are geometric, not predicted, but they are computed *from* a particular
    model's structure and two models disagree about where the binder sits. An
    unprefixed `epitope_coverage` would collapse those into one column and make
    the disagreement invisible.
    """
    return spec.stored_name(model)


def metric_records(
    *,
    run_id: str,
    design_id: str,
    model: str,
    values: Mapping[str, float | None],
    replicate: int,
    measured_at: datetime,
    details: Mapping[str, Any] | None = None,
) -> tuple[MetricRecord, ...]:
    """One replicate's worth of measurements for one design.

    A value that is None, NaN or infinite is stored with status `failed` rather
    than dropped. An absent row and a failed measurement are different facts:
    the first says nothing ran, the second says something ran and produced
    nonsense, and only the second is a reason to distrust the model rather than
    the harness.
    """
    records: list[MetricRecord] = []
    for key, raw in values.items():
        spec = spec_for(key)
        name = spec.stored_name(model)
        usable = raw is not None and math.isfinite(raw)
        records.append(
            MetricRecord(
                metric_id=stable_id("metric", run_id, design_id, name, str(replicate)),
                run_id=run_id,
                design_id=design_id,
                name=name,
                value=float(raw) if usable else None,
                unit=spec.unit,
                direction=spec.direction,
                replicate=replicate,
                status=MetricStatus.OK if usable else MetricStatus.FAILED,
                details=dict(details) if details else None,
                measured_at=measured_at,
            )
        )
    return tuple(records)


def scored_design_ids(records: Iterable[MetricRecord]) -> tuple[str, ...]:
    """Distinct designs that got at least one usable measurement.

    This is what a scoring run reports as `n_produced`. Counting metric rows
    instead would report six times the truth at six samples, and counting
    attempted designs would report a failed fold as a success -- the exact
    conflation harness-design §2 exists to prevent.
    """
    return tuple(
        sorted({record.design_id for record in records if record.status == MetricStatus.OK})
    )
