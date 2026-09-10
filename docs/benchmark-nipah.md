# What the scorers are worth, measured

Nipah-G labelled set: **434 designs, 56 binders**. Every model scored at the
protocol the published numbers used — recycling 3, one sample, seed 42, complex
only. Reproduce with:

```bash
python scripts/benchmark_auc.py \
    --existing .../mosaic_setup/benchmark/labelled_scores \
    --new build-logs/nipah_bench \
    --labels benchmark_sets/nipah_434_labels.csv \
    --fasta benchmark_sets/nipah_434.fasta \
    --out build-logs/nipah-auc.png
```

## Results, 2026-09-09

![Nipah-G AUC by model](figures/nipah-auc.png)

| model | AUC | best metric | note |
|---|---|---|---|
| boltz2 | **0.841** | `iplddt` | |
| esmfold2 | **0.833** | `tb_pae` | intended held-out judge |
| **chai1** | **0.828** | `bt_iptm` | added by this harness |
| **af3** | 0.693 | `complex_ptm` | added by this harness |
| protenix (mini) | 0.680 | `tb_pae` | |
| boltz1 | 0.671 | `tb_pae` | |
| af2 | 0.647 | `ptm_energy` | |
| **promera** | 0.610 | `tb_pae` | added; **not comparable**, see below |
| of3 | 0.597 | `ipsae_min` | |

**Control bar: 0.642** — the best sequence-only property (length). Net charge
and molecular weight are close behind. These cannot know anything about the
target, and a model that does not clear them has not earned its GPU time.

## Reading it

**Chai-1 is a genuine addition.** 0.828 is level with ESMFold2 and second only
to Boltz-2. That matters beyond the ranking: ESMFold2 is meant to be the
held-out judge, and until now nothing else in the panel was good enough to take
over if it were ever compromised by being used upstream.

**AF3 is mid-pack at 0.693.** Above Protenix and Boltz-1, nowhere near the top.
Worth saying plainly, because the name invites the opposite expectation.

**Promera's 0.610 is not a verdict on Promera.** mosaic refuses to hand it the
campaign alignment (`models/promera.py:77-85` raises rather than silently
running its own ColabFold search), so it folded MSA-free while every other model
had the 64-sequence Nipah alignment. It is the only scorer here running a
different protocol. Read the number as *not comparable*, not as *worse*.

**Five of nine sit within 0.06 of the control bar.** That is the same finding
the original six produced, now on a wider panel. The response is not more
models — it is the negative controls in `scoring-stage.md` §10. Without a
false-positive rate, an ipTM threshold is a number with no denominator.

## Why `--tag macc` is the default

`labelled_scores/` holds three runs of the same six models: `acc_`, `macc_` and
`accsmoke_`. Only **`macc_`** — matched compute, the same trunk passes and the
same diffusion steps for every backend — reproduces the AUCs the benchmark
reports. Reading `acc_` gives Protenix 0.552 where the report says 0.680, with
nothing in the output to say why.

`scripts/benchmark_auc.py` reproduces all six published numbers exactly against
`macc_`, which is what puts the three new bars on the same axis as the old ones
rather than beside a differently-computed number.

## Method notes

- Each model is scored by its **best single metric**, which is how the existing
  benchmark reports it. Fixing one metric across models would penalise the ones
  that do not emit it — Promera and AF3 have no ipSAE.
- AUC is Mann-Whitney U / (n_pos × n_neg) with averaged ranks for ties,
  transcribed from `mosaic/benchmark/common.py::auc`. Several confidence scores
  saturate near 1.0 on easy designs and a naive implementation rewards that.
- AF3 completed 431 of 434; three designs failed and are excluded from its AUC
  rather than counted as failures.
- Chai-1 and AF3 ran at their native 200-step diffusion samplers, not the
  25-step figure the mosaic backends share. Trunk passes, sample count and seed
  are matched; the sampler budget is not, because 25 steps is far outside the
  regime either was tuned for.

---

# The GFP sanity check

A second, cheaper question: **does each model produce the right fold at all?**
GFP is an 11-strand beta barrel, known since 1996, that every one of these
models has seen thousands of. Something that is not a barrel is a wiring
failure rather than a modelling opinion — and unlike the benchmark above, this
needs no labels and no controls.

![GFP folded by every scorer](figures/gfp-gallery.png)

Folded alone, with GFP's own 770-sequence alignment where the model can use
one. RMSD is to the most confident prediction, which stands in for a reference
structure.

| model | pLDDT | RMSD to consensus | barrel? |
|---|---|---|---|
| protenix (base) | 95.8 | reference | yes |
| af2 | 95.4 | 3.9 Å | yes |
| boltz1 | 94.4 | 1.9 Å | yes |
| boltz2 | 92.4 | 3.5 Å | yes |
| chai1 | 92.0 | 3.7 Å | yes |
| af3 | 91.4 | 4.0 Å | yes |
| esmfold2 | 41.6 | 6.9 Å | roughly |
| of3 | 38.5 | 23.9 Å | **no** |
| promera | 33.7 | 20.7 Å | **no** |

**Six of nine agree on the same barrel** within 4 Å at pLDDT above 91. That is
the result worth having: six independent implementations, three containers, and
they converge on the same structure.

**OpenFold3 does not fold GFP.** 23.9 Å from consensus and visibly not a
barrel, with the alignment loaded (its pLDDT moved 34.6 → 38.5 when the MSA was
supplied, so it consumed it and still failed). This agrees with its last place
on the benchmark above, at 0.597 — below the sequence-only control bar. Two
independent measurements now point the same way, which is worth more than
either alone.

**Promera does not either**, but it is the one model that could not be given
the alignment at all, so the comparison is not like-for-like.

**ESMFold2 gets the topology roughly right at pLDDT 42.** The structure is
recognisably a barrel; the confidence is not. That deserves attention rather
than a shrug, because ESMFold2 is the intended held-out judge and scored 0.833
on the benchmark. A judge whose confidence is uninformative on a protein it
should find trivial is worth understanding before trusting it to arbitrate.

## What the alignment was worth

The first run of this test folded everything single-sequence, and measured the
wrong thing:

| model | single-sequence | with 770-sequence MSA |
|---|---|---|
| af2 | 38.5 | **95.4** |
| boltz1 | 38.4 | **94.4** |
| af3 | 26.9 | **91.4** |
| chai1 | 49.5 | **92.0** |
| boltz2 | 93.7 | 92.4 |

AF2 at 38.5 was never broken — it had no homologs. Boltz-2 barely moved, which
is consistent with it being the least MSA-dependent of the group and explains
why it was the lone confident model before the alignment existed.

The monomer reader passes no alignment by default, and that is correct for
scoring designs: a de novo binder has no homologs by construction. It is wrong
for a control protein. `--monomer-msa` exists for exactly this and is a separate
flag from `--target-msa` so that a design cannot take that path by accident.
