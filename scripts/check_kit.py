#!/usr/bin/env python3
"""Check a kit against this machine, and against the config that points at it.

Three questions, all answerable in a second and none of them needing a GPU:

1. **Is the kit well formed?** Required fields, an entrypoint that exists, a
   direction on every metric.
2. **Is everything it needs installed here?** Each `requires:` entry is
   resolved against the campaign's bindings and then *verified* -- a file that
   has to exist, or a module that has to import inside the container. A name
   with no binding is reported the way the harness will eventually report it:
   as something a store could be asked for, not as a crash.
3. **Has the campaign config drifted from the kit?** Today the config restates
   the metrics, the inputs and the loss models, because the harness does not
   read `kit.yaml` yet. Restating is fine; disagreeing silently is not. This
   is what keeps the duplication honest until the harness absorbs it.

    python optimizers/kitcheck.py --kit optimizers/mosaic-af2-refine \\
        --bindings configs/kit_bindings.yaml \\
        --config configs/optimize/af2_refine.yaml

`verify.import` runs inside the bound container, so it needs singularity on
PATH. Pass `--no-probe` to skip it and check only what is on disk.

This is a standalone stand-in for harness preflight, deliberately: it proves
the manifest is checkable before anything under `src/` learns to read one.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import yaml

REQUIRED_KIT_FIELDS = ("kit_schema", "name", "version", "kind", "entrypoint",
                       "requires", "inputs", "loss_models", "metrics", "cost")
DIRECTIONS = {"min", "max", "none"}


class Report:
    """Findings, printed in one place so a run says everything it found."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.missing: list[tuple[str, dict]] = []
        self.notes: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def absent(self, name: str, requirement: dict) -> None:
        self.missing.append((name, requirement))

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def ok(self) -> bool:
        return not self.failures and not self.missing


def main() -> int:
    args = parse_args()
    report = Report()

    kit_dir = Path(args.kit).resolve()
    kit = load_yaml(kit_dir / "kit.yaml", report, "kit")
    if kit is None:
        return finish(report, None)

    check_shape(kit, kit_dir, report)
    loaded = load_yaml(Path(args.bindings), report, "bindings") if args.bindings else {}
    # Every binding is a mapping; anything else at the top level is a header
    # key like `bindings_schema`, not a name a kit can require.
    bindings = {
        name: entry for name, entry in (loaded or {}).items() if isinstance(entry, dict)
    }
    check_requirements(kit, bindings, report, probe=not args.no_probe)

    if args.config:
        config = load_yaml(Path(args.config), report, "config")
        if config is not None:
            check_agreement(kit, config, report)
            estimate_walltime(kit, config, report)

    return finish(report, kit)


# ---------------------------------------------------------------------------
# 1. shape
# ---------------------------------------------------------------------------


def check_shape(kit: dict, kit_dir: Path, report: Report) -> None:
    for field in REQUIRED_KIT_FIELDS:
        if field not in kit:
            report.fail(f"kit.yaml has no `{field}`")

    if kit.get("kit_schema") != 1:
        report.fail(f"kit_schema {kit.get('kit_schema')!r} is not 1")

    entrypoint = kit_dir / str(kit.get("entrypoint", ""))
    if not entrypoint.is_file():
        report.fail(f"entrypoint not found at {entrypoint}")

    for name, spec in (kit.get("metrics") or {}).items():
        direction = (spec or {}).get("direction")
        if direction not in DIRECTIONS:
            # The same rule the harness applies, for the same reason: a number
            # stored the wrong way round sorts backwards and nothing in the row
            # says so.
            report.fail(
                f"metric {name!r} has direction {direction!r}; must be one of "
                f"{sorted(DIRECTIONS)}"
            )

    capabilities = kit.get("capabilities") or {}
    if "max_children" not in capabilities:
        report.fail("capabilities has no `max_children` ceiling")


# ---------------------------------------------------------------------------
# 2. dependencies
# ---------------------------------------------------------------------------


def check_requirements(kit: dict, bindings: dict, report: Report, *, probe: bool) -> None:
    requires = kit.get("requires") or []
    for requirement in requires:
        name = requirement.get("name")
        binding = bindings.get(name)

        if binding is None:
            if requirement.get("optional"):
                report.note(f"{name}: optional and not bound, skipped")
            else:
                report.absent(name, requirement)
            continue

        path = Path(str(binding.get("path", "")))
        if not path.exists():
            report.fail(f"{name}: bound to {path}, which does not exist")
            continue

        verify = requirement.get("verify") or {}

        for relative in verify.get("files") or []:
            if not (path / relative).exists():
                report.fail(f"{name}: {path}/{relative} is missing")

        module = verify.get("import")
        if module and probe:
            ok, detail = probe_import(module, requirement, requires, bindings, report)
            if not ok:
                report.fail(f"{name}: `import {module}` failed inside the container. {detail}")
        elif module:
            report.note(f"{name}: `import {module}` not probed (--no-probe)")


def probe_import(module: str, requirement: dict, requires: list, bindings: dict, report: Report):
    """Run the import inside the bound container, with any overlay applied.

    The overlay matters: on this campaign the shipped image fails the very
    import the kit verifies, and the `mosaic-src` requirement is what fixes
    it. Probing without the overlay would report a false failure and probing
    the host would report a false success.
    """
    container = bindings.get(requirement["name"]) or {}
    image = container.get("path")
    interpreter = container.get("interpreter", "python")

    if not shutil.which("singularity"):
        report.note(f"singularity not on PATH; `import {module}` not probed")
        return True, ""

    argv = ["singularity", "exec", "--cleanenv"]
    for name, binding in bindings.items():
        if (binding or {}).get("kind") != "source_overlay":
            continue
        target = next(
            (r.get("target") for r in requires if r.get("name") == name and r.get("target")),
            None,
        )
        if target:
            argv += ["--bind", f"{binding['path']}:{target}:ro"]
    argv += [str(image), interpreter, "-c", f"import {module}"]

    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=600, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"probe could not run: {error}"
    if done.returncode == 0:
        return True, ""
    tail = (done.stderr or done.stdout or "").strip().splitlines()
    return False, (tail[-1] if tail else "no output")


# ---------------------------------------------------------------------------
# 3. agreement with the campaign config
# ---------------------------------------------------------------------------


def check_agreement(kit: dict, config: dict, report: Report) -> None:
    kit_metrics = {k: (v or {}).get("direction") for k, v in (kit.get("metrics") or {}).items()}
    cfg_metrics = {k: (v or {}).get("direction") for k, v in (config.get("metrics") or {}).items()}
    if kit_metrics != cfg_metrics:
        only_kit = sorted(set(kit_metrics) - set(cfg_metrics))
        only_cfg = sorted(set(cfg_metrics) - set(kit_metrics))
        flipped = sorted(
            k for k in set(kit_metrics) & set(cfg_metrics) if kit_metrics[k] != cfg_metrics[k]
        )
        if only_kit:
            report.fail(f"config is missing metric(s) the kit declares: {only_kit}")
        if only_cfg:
            report.fail(f"config declares metric(s) the kit does not: {only_cfg}")
        for key in flipped:
            report.fail(
                f"metric {key!r} points {cfg_metrics[key]!r} in the config and "
                f"{kit_metrics[key]!r} in the kit"
            )

    if list(kit.get("inputs") or []) != list(config.get("inputs") or []):
        report.fail(
            f"inputs disagree: kit {kit.get('inputs')} vs config {config.get('inputs')}"
        )

    # Restated, not inherited -- so it must be present and it must agree.
    if "loss_models" not in config:
        report.fail("config does not restate `loss_models`; it is never inherited")
    elif sorted(config["loss_models"]) != sorted(kit.get("loss_models") or []):
        report.fail(
            f"loss_models disagree: kit {sorted(kit.get('loss_models') or [])} vs "
            f"config {sorted(config['loss_models'])}. If that is deliberate, the kit "
            "is the wrong one or its version needs to move."
        )

    capabilities = kit.get("capabilities") or {}
    ceiling = capabilities.get("max_children")
    wanted = config.get("max_children")
    if ceiling is not None and wanted is not None and wanted > ceiling:
        report.fail(f"config asks for {wanted} children; the kit tops out at {ceiling}")

    if capabilities.get("changes_length") is False and config.get("length_delta", 0) != 0:
        report.fail(
            f"config allows length_delta {config['length_delta']} but the kit "
            "declares it cannot change length"
        )


def estimate_walltime(kit: dict, config: dict, report: Report) -> None:
    """The resources argument, made arithmetic.

    The kit supplies two measured constants and the campaign supplies the
    shape. Neither half is a walltime on its own, which is exactly why the
    field belongs to the campaign and the constants belong to the kit.
    """
    cost = kit.get("cost") or {}
    compile_s = cost.get("compile_seconds")
    per_parent = cost.get("per_parent_seconds")
    if compile_s is None or per_parent is None:
        return

    manifest = Path(str(config.get("design_set", "")))
    if not manifest.is_file():
        report.note("design set not readable; no walltime estimate")
        return

    n_parents = int(json.loads(manifest.read_text()).get("n_designs", 0))
    jobs = int(((config.get("sharding") or {}).get("jobs")) or 1)
    per_shard = math.ceil(n_parents / jobs) if jobs else n_parents
    needed = compile_s + per_parent * per_shard

    reserved = (config.get("resources") or {}).get("walltime")
    report.note(
        f"cost model: {n_parents} parents over {jobs} shard(s) = {per_shard} each, "
        f"≈ {needed / 60:.1f} min per task (reserved {reserved})"
    )
    if reserved:
        report.note(
            "  the kit cannot know that number; it supplies the constants and the "
            "campaign supplies the parent and shard counts"
        )


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def load_yaml(path: Path, report: Report, what: str):
    if not path.is_file():
        report.fail(f"no {what} file at {path}")
        return None
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        report.fail(f"{what} file {path} is not valid YAML: {error}")
        return None


def finish(report: Report, kit: dict | None) -> int:
    if kit:
        print(f"kit {kit.get('name')}@{kit.get('version')}  ({kit.get('kind')})")

    for note in report.notes:
        print(f"  note     {note}")

    for name, requirement in report.missing:
        kind = requirement.get("kind", "dependency")
        version = requirement.get("version", "any")
        print(f"  MISSING  {name} ({kind} {version}) is required and not bound")
        why = (requirement.get("why") or "").strip().replace("\n", " ")
        if why:
            print(f"           {why}")
        if requirement.get("provided_by") == "harness":
            # Not something a store hands out. If this is missing, the harness
            # is too old, and fetching a copy would paper over that.
            print("           supplied by the harness itself; upgrade bindocracy rather than fetching it")
        else:
            print(f"           bind it in the bindings file, or fetch {name} once there is a store")

    for failure in report.failures:
        print(f"  FAIL     {failure}")

    print("\n" + ("PASS -- this kit is installed and the config agrees with it"
                  if report.ok else "NOT READY"))
    return 0 if report.ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kit", required=True, help="Directory holding kit.yaml.")
    parser.add_argument("--bindings", help="This machine's symbol-to-path file.")
    parser.add_argument("--config", help="A campaign optimize YAML to cross-check.")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip container imports; check only what is on disk.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
