# Adding a generation tool

A guide for whoever does this next, human or agent. It covers **generation
only** — filtering, clustering, evaluation, and optimization are not built.

Adding a tool is **one package and one line**. If you find yourself editing
anything else under `src/bindocracy/`, stop: either the contract is missing
something (say so, don't work around it) or the change belongs in the tool.

```text
src/bindocracy/tools/<tool>/
├── __init__.py     exports the plugin
├── config.py       what its YAML is allowed to say
├── preflight.py    what must be true before a GPU is touched
├── launch.py       the command for one task
├── adapter.py      its output → normalized records
└── plugin.py       the five methods that tie those together
```

Then one line in `src/bindocracy/tools/__init__.py`:

```python
register(YourPlugin)
```

A tool outside this repository registers itself the same way, by being named in
`BINDOCRACY_PLUGINS`. `tests/toytool/` is a complete worked example of exactly
that, and `tests/test_third_tool.py` drives it through the real workflow.

---

## Before you write anything

**Run the tool by hand once and keep the output.** Every adapter here was
written against observed output, and every time the docs disagreed with the
files, the files were right. `docs/harness-design.md` §3 lists cases where
upstream documentation described directories the code never creates and flags
that were dead.

**Read `docs/known-issues.md`.** Most of it is not about protein design. Node-
local `TMPDIR`, the container CA bundle, and `--cleanenv` dropping
`CUDA_VISIBLE_DEVICES` have each killed a real run here.

**Answer these four questions.** They are what the contract asks for, and the
tools already in the tree answer them differently:

| Question | Mosaic | BoltzGen | Genie 3 | PXDesign |
|---|---|---|---|---|
| What form of the target? | sequence + MSA | structure (CIF) | a problem set: renumbered PDB + FASTA + epitope | structure (CIF) + a precomputed MSA directory |
| What is consumed and archived? | a driver script it executes | a design spec bound into the container | both — a driver, and the experiment YAML it renders per task | an input spec passed to `pipeline -i` |
| What does it write? | JSON lines, one per design | a 237-column CSV | a 51-column CSV with one row *per design per fold* | a 31-column CSV, always exactly `--N_sample` rows |
| Does it judge its own output? | no | yes — `produced` and `passed` differ | yes — a v0 success reducer | yes — **four** filters that disagree |

PXDesign is the shortest of the four and the one to copy first: no driver, no
rendering, and every value the harness owns is a CLI flag. What it costs
instead is config surface, because three of its CLI defaults are wrong for a
campaign — a preset that configures no filters, a sampling schedule the CLI
overwrites, and a seed taken from the clock — and each is answered by a
required or explicitly-defaulted field rather than left to the default.

Genie 3 is the one to read if your tool has no CLI override for its output
directory. It has none for its seed or its sample count either, so a driver
renders one config per task from the archived template, and preflight refuses a
template that sets any of those three itself. Two answers to "how many designs
did this run ask for" is worse than none.

---

## The five methods

`ToolPlugin` (`src/bindocracy/tools/base.py`) is the whole contract.

### 1. `config_type` — what the YAML may say

Subclass `ToolConfig`, which fixes `schema_version`, `name`, `tool`, and
`resources`. Everything else is yours. `tool` must be a `Literal`, because it
is what selects your plugin.

```python
class YourConfig(ToolConfig):
    schema_version: Literal[1]
    tool: Literal["yourtool"]
    sampling: YourSamplingConfig
    runtime: YourRuntimeConfig
```

Configs are `extra="forbid"` and frozen, so a typo fails in a second rather
than forty minutes into a GPU job.

**Bump `schema_version` when you add a required field.** A stored config that
cannot be re-validated is unreadable data, and the version is what makes the
error say why.

### 2. `preflight` — refuse, don't warn

Runs after structural validation, before anything is planned. Assert what your
tool actually needs and nothing else: `TargetConfig.msa` and `structure_cif`
are both optional precisely so each tool can require what it reads.

The assertions worth writing are the ones that would otherwise fail *silently*.
BoltzGen's spec carries its own structure path, so preflight checks it is the
campaign's target — without that, a spec pointing elsewhere designs against the
wrong protein and the output looks entirely normal.

### 3. `tool_plan` — the shape of a run

```python
ToolPlan(
    jobs=...,                  # tasks, each one Slurm job
    designs_per_task=...,      # final outputs asked for
    generated_per_task=...,    # candidates attempted, if that differs
    designs_file="...",        # per-task output, relative to the task dir
    archives={"driver": path}, # inputs consumed by the run
    container=...,
    workflow={...},            # anything worth recording about the run
    inputs={"label": path},    # every file the tool reads
)
```

Two things people get wrong here:

**`archives` is for what the run *consumes*, not what describes it.** The
driver or spec is copied into `provenance/` and executed from the copy, so what
a run used stays inspectable when the working tree moves. Your harness config
is *not* archived — it is stored whole in the database.

**`inputs` is what your tool actually opens.** It is digested at planning and
verified immediately before launch, so a FASTA replaced in place stops the run
instead of quietly changing it. Declaring files you do not read makes runs fail
for no reason.

### 4. `launch_spec` — one task's command

Returns argv, environment, resources, log path, and expected outputs. **It
takes the manifest, not the loaded config**, and rebuilds the config from what
the manifest stored via `self.configs_of(manifest)`. A planned run must launch
what it was planned with, even if the YAML has changed since.

Never call `sbatch`. Snakemake submits; you describe.

Use `slurm_resources(cluster, resources)` unless your tool genuinely needs
something else.

### 5. `adapter` — output to records

Implement `OutputAdapter.collect(run_dir, run) -> CollectedRun`. Use the shared
helpers in `adapters/common.py` for manifest reading, artifacts, and the run
window; write the parsing yourself, because that is where tools differ.

Four rules, each of which cost something to learn:

**Skip bad rows; don't fail the run.** One malformed line must not cost the
designs written before and after it. Count what you skipped in `count_details`.

**Native IDs must be unique within a run.** If your tool numbers designs per
task, qualify them (`task-0000-...`). BoltzGen did not, and a two-task run
silently collected half its designs.

**Producing and passing are different facts.** A candidate your tool's own
filters rejected is still `produced`; the verdict is a `DecisionRecord`. A rank
needs the `scope_id` of the pool it was ranked *within* — per task, if your tool
ranks per task.

**Everything must be deterministic.** Record IDs come from `stable_id(...)` over
stable inputs, and timestamps from the files or the manifest — never
`utcnow()`. Re-collecting an unchanged directory must produce an identical
bundle, or re-ingestion looks like a conflicting rewrite.

---

## Does the tool write its own `status.json`?

Most tools do not, and that is fine: the harness records the outcome around the
process. Write your own **only if you know more than your exit code does** —
Mosaic's driver does, because only it can distinguish a walltime kill that kept
nine designs (`partial`) from a crash that kept none (`failed`).

A status the tool wrote is never overwritten.

An ordinary tool failure is **not** a workflow failure. It is recorded, and
collection still runs, so the failed run reaches the database as history.

---

## Testing it

Mirror the existing tests; each file name says what seam it guards.

| Test | What it must catch |
|---|---|
| `test_<tool>_launch.py` | wrong argv, wrong resources, a preflight that would let a silent-wrong-answer through |
| `test_<tool>_adapter.py` | truncation, invalid rows, duplicate IDs, missing tasks, non-determinism |
| `test_third_tool.py` | that your plugin needs no change to the library |

**Use real output as the fixture.** `tests/fixtures/boltzgen/` is four
unmodified rows from a real benchmark run. Writing it that way immediately
caught a bug: `X` is a legal letter, so a sequence regex accepted unknown
residues that cannot be ordered.

**Do not let a fixture write files the tool never produces.** The BoltzGen
fixture wrote a `status.json` BoltzGen does not write, which hid the fact that a
successful run would fail Snakemake for want of a declared output.

**Check your test fails.** Break the thing on purpose and confirm the right
test goes red. Several bugs here survived a green suite.

Then:

```bash
./check.sh          # ruff, pytest, snakemake --lint
```

---

## A first run

```yaml
# an execution index, kept beside the campaign data — not in this repository
database: /absolute/path/to/campaign.duckdb
run_root: /absolute/path/to/runs
general_config: /absolute/path/to/configs/general_config.yaml
runs:
  - name: yourtool-smoke-01
    config: /absolute/path/to/configs/yourtool/smoke.yaml
```

```bash
uv run snakemake --dry-run --configfile <index>.yaml   # expect one job per task
uv run snakemake --configfile <index>.yaml --profile workflow/profiles/slurm
```

Start with one task and one design. The first real run of a new tool is for
finding out that the container starts, not for making binders.

Afterwards:

```sql
SELECT tool, status, n_requested, n_attempted, n_produced, n_passed FROM runs;
SELECT producing_tool, count(*) FROM design_history GROUP BY 1;
```

---

## Things that are not your job

- **Submitting.** Snakemake owns it.
- **Writing to DuckDB.** One serialized rule does, and only it.
- **Naming run directories.** The workflow index does.
- **Deciding a run succeeded.** Collection does, from the artifacts — exit
  status is corroboration, not evidence (`harness-design.md` §6).

## Where to look when stuck

| You want | Read |
|---|---|
| the contract | `src/bindocracy/tools/base.py` |
| a simple worked example | `tests/toytool/plugin.py` |
| a tool with a driver | `src/bindocracy/tools/mosaic/` |
| a tool with its own filters | `src/bindocracy/tools/boltzgen/` |
| a tool whose config must be written per task | `src/bindocracy/tools/genie3/` |
| a tool that scores each design several times | `src/bindocracy/tools/genie3/adapter.py` |
| a tool whose CLI defaults are actively wrong | `src/bindocracy/tools/pxdesign/config.py` |
| what the tables mean | `docs/database.md` |
| what has already gone wrong | `docs/known-issues.md` |
