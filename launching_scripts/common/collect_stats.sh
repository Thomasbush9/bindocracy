#!/usr/bin/env bash
#
# Record what one benchmark job actually consumed.
#
#     collect_stats.sh <tool-name> <jobid> [<jobid> ...]
#
# Writes outputs/<tool>/resources/{jobstats.txt,sacct.txt,summary.tsv}.
#
# WHY BOTH TOOLS: `jobstats` gives the utilisation view (GPU %, GPU memory,
# CPU efficiency) but only for a finished job and only for a while afterwards.
# `sacct` gives the billing/allocation view (MaxRSS, elapsed, exit code) and is
# retained far longer. Neither alone answers "what should I request next time".
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

TOOL="${1:?usage: collect_stats.sh <tool> <jobid> [...]}"
shift
DEST="$OUT_ROOT/$TOOL/resources"
mkdir -p "$DEST"

: > "$DEST/jobstats.txt"
: > "$DEST/sacct.txt"

for JOB in "$@"; do
    {
        echo "===== jobstats $JOB ====="
        jobstats "$JOB" 2>&1 || echo "(jobstats unavailable for $JOB)"
        echo
    } >> "$DEST/jobstats.txt"

    {
        echo "===== sacct $JOB ====="
        sacct -j "$JOB" --units=G -o \
            JobID%20,JobName%18,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,MaxVMSize,AveCPU,NodeList%20 2>&1
        echo
        echo "----- TRES -----"
        sacct -j "$JOB" -o JobID%20,AllocTRES%80 2>&1
        echo
    } >> "$DEST/sacct.txt"
done

# One machine-readable row per job for the cross-tool ranking table.
{
    printf 'tool\tjobid\tstate\telapsed\treq_mem\tmax_rss\talloc_cpus\n'
    for JOB in "$@"; do
        sacct -j "$JOB" --units=G -n -P -o JobID,State,Elapsed,ReqMem,MaxRSS,AllocCPUS \
            | awk -F'|' -v t="$TOOL" '$1 ~ /\.batch$/ {
                  split($1, a, "."); printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", t, a[1], $2, $3, $4, $5, $6 }'
    done
} > "$DEST/summary.tsv"

echo "wrote $DEST/{jobstats.txt,sacct.txt,summary.tsv}"
cat "$DEST/summary.tsv"
