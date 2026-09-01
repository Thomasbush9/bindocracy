# Drivers

Production code that runs **inside a container**, invoked by the harness. A
driver is archived per run and executed from the archive, so what a run used
stays inspectable after the working tree moves on.

| Driver | Tool | Contract |
|---|---|---|
| `mosaic/hallucinate_binders.py` | Mosaic | `designs.jsonl` + `status.json`; see its module docstring |
| `genie3/run_genie3.py` | Genie 3 | renders `<task>/experiment.yaml` from the archived template, then runs `genie3 run`; see its module docstring |
| `protein_hunter/run_boltz_design.py` | Protein-Hunter | seeds the ColabFold MSA cache for this task, then runs the Boltz design pipeline |
| `freebindcraft/run_freebindcraft.py` | FreeBindCraft | renders `<task>/<target>.json` and `<task>/<advanced>.json` from the archived templates, then runs `bindcraft`; see its module docstring |

They exist for different reasons. Mosaic's *is* the science — a copy of the
hallucination loop, run because no packaged entry point does what the lab wants.
The other three are glue: Genie 3's writes a config per task because the CLI
takes a config file and nothing else, Protein-Hunter's writes an MSA cache into
the task's own save directory because there is no flag for a precomputed
alignment, and FreeBindCraft's writes a settings pair per task because the
output directory, the design count, the epitope and the trajectory budget are
all keys inside JSON documents rather than flags. All three sit next to
container fixes that only work in the same process as the tool.

A driver cannot import `bindocracy` — it runs in the tool's own image, which
knows nothing about this package. Everything it needs arrives as an argument.
That is also why `tests/test_driver_contract.py` parses the connector's real
argument vector with the driver's real parser: the two never meet at runtime.

Not to be confused with `../legacy/`, which holds the hand-written `sbatch`
launchers from the alpha benchmark. Those are kept as a reference for what the
harness had to absorb, and are not used by it.
