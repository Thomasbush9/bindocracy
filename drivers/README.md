# Drivers

Production code that runs **inside a container**, invoked by the harness. A
driver is archived per run and executed from the archive, so what a run used
stays inspectable after the working tree moves on.

| Driver | Tool | Contract |
|---|---|---|
| `mosaic/hallucinate_binders.py` | Mosaic | `designs.jsonl` + `status.json`; see its module docstring |
| `genie3/run_genie3.py` | Genie 3 | renders `<task>/experiment.yaml` from the archived template, then runs `genie3 run`; see its module docstring |

The two exist for different reasons. Mosaic's *is* the science — a copy of the
hallucination loop, run because no packaged entry point does what the lab wants.
Genie 3's is glue: its CLI takes a config file and nothing else, so a run that
fans out over tasks needs something to write one config per task, next to the
container fixes that only work in the same process as the tool.

A driver cannot import `bindocracy` — it runs in the tool's own image, which
knows nothing about this package. Everything it needs arrives as an argument.
That is also why `tests/test_driver_contract.py` parses the connector's real
argument vector with the driver's real parser: the two never meet at runtime.

Not to be confused with `../legacy/`, which holds the hand-written `sbatch`
launchers from the alpha benchmark. Those are kept as a reference for what the
harness had to absorb, and are not used by it.
