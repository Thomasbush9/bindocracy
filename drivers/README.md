# Drivers

Production code that runs **inside a container**, invoked by the harness. A
driver is archived per run and executed from the archive, so what a run used
stays inspectable after the working tree moves on.

| Driver | Tool | Contract |
|---|---|---|
| `mosaic/hallucinate_binders.py` | Mosaic | `designs.jsonl` + `status.json`; see its module docstring |

A driver cannot import `bindocracy` — it runs in the tool's own image, which
knows nothing about this package. Everything it needs arrives as an argument.
That is also why `tests/test_driver_contract.py` parses the connector's real
argument vector with the driver's real parser: the two never meet at runtime.

Not to be confused with `../legacy/`, which holds the hand-written `sbatch`
launchers from the alpha benchmark. Those are kept as a reference for what the
harness had to absorb, and are not used by it.
