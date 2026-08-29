#!/usr/bin/env bash
#
# Sample GPU utilisation and memory for the life of the job.
#
#     source common/gpu_trace.sh
#     gpu_trace_start "$OUT/resources/gpu_trace.csv"
#     ... the real work ...
#     gpu_trace_stop
#
# WHY THIS EXISTS: `jobstats` reported "GPU utilization 0% <-- GPU was not used"
# for a job whose GPU memory peaked at 76 GB. Its sampler is far too coarse for
# jobs of a few minutes, and it reports only a maximum for memory, so it cannot
# distinguish a tool that needs 70 GB throughout from one that touches it once.
# A 15 s trace answers both, and costs nothing.
#
# CAVEAT that no sampler can remove: JAX preallocates ~75% of the device by
# default, so for the JAX-based tools (Mosaic, PXDesign, Genie3's AF2 stage)
# the memory column measures the allocator, not the model. Set
# XLA_PYTHON_CLIENT_PREALLOCATE=false to make the number mean demand — at some
# cost in speed and fragmentation. Which convention a run used is recorded in
# the header line below.

gpu_trace_start() {
    local out="${1:?usage: gpu_trace_start <csv path>}"
    mkdir -p "$(dirname "$out")"
    # Read BOTH forms: the launchers set the SINGULARITYENV_ variant, because
    # that is what actually reaches the process inside the container. Checking
    # only the bare name would label every run "unset" and quietly misreport
    # which convention produced the memory column.
    local prealloc="${XLA_PYTHON_CLIENT_PREALLOCATE:-${SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE:-}}"
    {
        echo "# host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
        echo "# XLA_PYTHON_CLIENT_PREALLOCATE=${prealloc:-<unset, JAX preallocates ~75%>}"
        echo "# PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
        echo "# sample_interval_s=${GPU_TRACE_INTERVAL:-15}"
    } > "$out"
    # 15 s is fine for a multi-hour job but far too coarse to catch a peak on a
    # two-minute one -- a batch-size sweep needs GPU_TRACE_INTERVAL=2 or less, or
    # the memory column understates the maximum.
    local iv="${GPU_TRACE_INTERVAL:-15}"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
               --format=csv,nounits -l "$iv" >> "$out" 2>/dev/null &
    GPU_TRACE_PID=$!
    export GPU_TRACE_PID GPU_TRACE_FILE="$out"
}

gpu_trace_stop() {
    [[ -n "${GPU_TRACE_PID:-}" ]] || return 0
    kill "$GPU_TRACE_PID" 2>/dev/null || true
    wait "$GPU_TRACE_PID" 2>/dev/null || true
    unset GPU_TRACE_PID

    # Collapse the trace to the three numbers a resource request is made from.
    [[ -f "${GPU_TRACE_FILE:-}" ]] || return 0
    awk -F', ' '
        !/^#/ && NF >= 6 && $3 ~ /^[0-9]+$/ {
            n++; util += $3; if ($3 > umax) umax = $3
            if ($5 + 0 > mmax) mmax = $5 + 0
        }
        END {
            if (n == 0) { print "gpu_trace: no samples"; exit }
            printf "gpu_trace: %d samples  util mean %.1f%% peak %d%%  mem peak %.1f GiB\n",
                   n, util / n, umax, mmax / 1024
        }' "$GPU_TRACE_FILE"
}
