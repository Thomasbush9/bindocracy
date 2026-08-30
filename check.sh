#!/usr/bin/env bash
#
# The standard check. Everything CI would run, in one command:
#
#     ./check.sh
#
# Ruff never sees the Snakefile: Snakemake's syntax is not ordinary Python, and
# `snakemake --lint` is the tool that understands it.
set -euo pipefail
cd "$(dirname "$0")"

echo "== ruff"
# drivers/ is production code that runs inside a container, so it is linted
# with everything else -- it was outside this command until 2026-08-30, and a
# real import-order finding had been sitting in the Mosaic driver.
uv run ruff check src tests drivers

echo "== pytest"
uv run pytest -q

echo "== snakemake --lint"
# The lint needs an index whose paths resolve, so build a throwaway campaign
# from the same fixtures the tests use. Linting the placeholder template would
# only prove that placeholders are not real paths.
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
uv run python - "$scratch" <<'PY'
import sys, pathlib, yaml
sys.path.insert(0, "tests")
from conftest import write_configs
root = pathlib.Path(sys.argv[1])
general, model = write_configs(root)
(root / "index.yaml").write_text(yaml.safe_dump({
    "database": str(root / "campaign.duckdb"),
    "run_root": str(root / "runs"),
    "general_config": str(general),
    "runs": [{"name": "lint", "config": str(model)}],
}))
PY
uv run snakemake --lint --snakefile workflow/Snakefile \
    --configfile "$scratch/index.yaml" >/dev/null

echo "== ok"
