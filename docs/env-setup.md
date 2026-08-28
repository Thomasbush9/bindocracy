# The bindocracy environment

A single `uv`-managed virtualenv at `bindocracy/.venv`, Python 3.12. It holds the
**driver** side of the project — the workflow engine, the database, the parsers,
the CLI. It deliberately holds none of the design models: those live in their own
Singularity images under `binder_design/images/` and are invoked as containers.
See `running-containers.md`.

## Creating / restoring it

```bash
cd /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy
export UV_CACHE_DIR=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/.uv-cache
uv sync
```

`uv sync` reads `uv.lock` and reproduces the environment exactly, so **commit
`uv.lock`**. Add dependencies with `uv add <pkg>` (which updates both files)
rather than by hand-editing `pyproject.toml`.

Use it either by activating (`source .venv/bin/activate`) or, preferably in
scripts, by calling the interpreter directly: `.venv/bin/python`, `.venv/bin/snakemake`.

### Why `UV_CACHE_DIR` is redirected

uv's default cache is `~/.cache/uv`, which counts against the FASRC home quota.
The wheel cache for this environment is ~850 MB and the venv itself another
~840 MB, so both are kept in lab space. Put the export in your `~/.bashrc` if you
do not want to repeat it. The cache lives at `binder_design/.uv-cache`, one level
*above* the repo, so it is outside version control by construction.

## What is in it, and why

| Package | Why it is here |
| --- | --- |
| `snakemake` 9.26 | the workflow engine: DAG, re-entrancy, `--rerun-incomplete` |
| `snakemake-executor-plugin-slurm` 2.8 | Snakemake 9 moved executors out of core. Pulls `…-slurm-jobstep` automatically — do not depend on that one directly |
| `duckdb` 1.5 | the store. Reads the tools' heterogeneous CSVs **in place**, so ingest is SQL rather than nine bespoke parsers |
| `pandas`, `pyarrow` | hand results to matplotlib; Parquet so the unified design table is one file, not nine |
| `harlequin` 2.11 | terminal SQL IDE, DuckDB adapter built in |
| `biotite` 1.7 | reads **mmCIF as well as PDB** — see below |
| `pyyaml`, `pydantic` | parse and *validate* the per-tool configs |
| `jinja2` | render one target spec into each tool's own config and sbatch dialect |
| `typer`, `rich`, `matplotlib` | CLI, terminal output, figures |

Dev group (`uv sync --group dev`): `pytest`, `ruff`.

### Why biotite specifically

BoltzGen writes **CIF only, no PDB**, and Caliby's `out_pdb` column holds `.cif`
paths despite the name. The hand-rolled CA-atom PDB parser in
`launching_scripts/common/seq_qc.py` cannot read either, so two of the nine tools
were unreachable for any structure-level analysis. biotite reads both formats:

```python
import biotite.structure.io.pdbx as pdbx, biotite.structure as struc
arr = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
seq = struc.to_sequence(arr[arr.chain_id == "A"])[0][0]
```

It emits a `UserWarning` about falling back from `auth_comp_id` to `label_comp_id`
on these files. That is expected — the tools write minimal CIFs without the auth
columns — and the fallback is correct.

**Chain conventions differ per tool and must be recorded, not assumed.** BoltzGen
writes binder = A, target = B. Proteina-Complexa is the reverse. RFdiffusion
writes binder = A, target = B. Caliby joins chains with `:` alphabetically.

## Inspecting the database

```bash
.venv/bin/harlequin path/to/designs.duckdb
```

This was chosen over DuckDB's own `ui` extension (`duckdb -ui`) because harlequin
is a TUI: it runs in the SSH session you already have, with no port forwarding
and no browser, and the extension does not have to be downloaded at first use.

DuckDB can also query the raw tool outputs without any ingest step, which is the
fastest way to answer a one-off question:

```sql
SELECT id, designed_sequence, design_to_target_iptm
FROM read_csv_auto('outputs/boltzgen/results/final_ranked_designs/final_designs_metrics_*.csv')
ORDER BY design_to_target_iptm DESC LIMIT 10;
```

## The Python 3.6 boundary — read this before editing helper scripts

The FASRC **login-node system Python is 3.6.8**. Everything in
`launching_scripts/common/` (`inventory.py`, `seq_qc.py`, `summarize.py`,
`make_figures.py`, `build_report.py`) is deliberately 3.6-compatible and
stdlib-only so it runs under `/usr/bin/python3` with no environment at all —
no `from __future__ import annotations`, no `X | None`.

Those constraints do **not** apply to anything under `src/bindocracy/`, which
always runs under `.venv/bin/python` (3.12).

Do not casually mix the two. If a helper is migrated into the package, every
sbatch script that calls it must be updated to activate the venv first.

## Snakemake on Kempner

The SLURM executor is registered and its flags are available under
`snakemake --executor slurm --help`. Two are directly relevant to the cluster
policy of **max 10 concurrent jobs**:

- `--slurm-array-jobs` / `--slurm-array-limit` — submit as an array with a
  concurrency cap, rather than N independent jobs
- `-j / --jobs` — Snakemake's own cap on simultaneous submissions

Account is `kempner_bsabatini_lab`, partition `kempner_h100` (with the documented
RFdiffusion exception on `kempner`/A100 — see `known-issues.md`). These belong in
a Snakemake profile rather than on the command line; that profile is not written
yet.
