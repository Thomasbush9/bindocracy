#!/usr/bin/env python3
"""Turn each tool's resources/ directory into one comparison table.

    summarize.py [--outputs <dir>] [--markdown]

Reads, per tool:
  resources/summary.tsv   elapsed, MaxRSS, CPUs   (written by collect_stats.sh)
  resources/gpu_trace.csv 15 s nvidia-smi samples (written by gpu_trace.sh)

and prints wall-clock, host RAM, peak GPU memory and mean/peak GPU utilisation
side by side, plus minutes per design.

READ THE GPU-MEMORY COLUMN WITH CARE. It is the peak *reservation*, not demand:
JAX preallocates ~75% of the device unless XLA_PYTHON_CLIENT_PREALLOCATE=false,
and PyTorch's caching allocator holds freed blocks. Each gpu_trace.csv records in
its header which convention that run used. A tool showing ~76 GiB on an 80 GB
H100 is almost certainly reporting the JAX allocator, not a real requirement.

Python 3.6 compatible, stdlib only.
"""

import argparse
import os
import sys

DEFAULT_OUTPUTS = ("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
                   "binder_design/outputs")

TOOLS = ["mosaic", "boltzgen", "freebindcraft", "genie3", "proteina_complexa",
         "protein_hunter", "pxdesign", "rfdiffusion", "caliby"]


def parse_gpu_trace(path):
    """-> (peak_mem_gib, mean_util, peak_util, n_samples, prealloc_note)"""
    if not os.path.exists(path):
        return (None, None, None, 0, "")
    note = ""
    util, umax, mmax, n = 0.0, 0, 0, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if "PREALLOCATE" in line:
                    note = line.split("=", 1)[1].strip()
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                u = int(parts[2])
                m = int(parts[4])
            except ValueError:
                continue
            n += 1
            util += u
            umax = max(umax, u)
            mmax = max(mmax, m)
    if n == 0:
        return (None, None, None, 0, note)
    return (mmax / 1024.0, util / n, umax, n, note)


def parse_summary_tsv(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = [l.rstrip("\n").split("\t") for l in fh if l.strip()]
    if len(rows) < 2:
        return None
    hdr, vals = rows[0], rows[1]
    return dict(zip(hdr, vals))


def hms_to_min(s):
    if not s:
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    bits = [int(x) for x in s.split(":")]
    while len(bits) < 3:
        bits.insert(0, 0)
    h, m, sec = bits
    return days * 1440 + h * 60 + m + sec / 60.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default=DEFAULT_OUTPUTS)
    ap.add_argument("--designs", type=int, default=40)
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()

    rows = []
    for t in TOOLS:
        base = os.path.join(a.outputs, t, "resources")
        s = parse_summary_tsv(os.path.join(base, "summary.tsv"))
        gmem, gutil, gpeak, nsamp, note = parse_gpu_trace(
            os.path.join(base, "gpu_trace.csv"))
        elapsed = s.get("elapsed") if s else None
        mins = hms_to_min(elapsed) if elapsed else None
        rows.append({
            "tool": t,
            "state": (s or {}).get("state", "-"),
            "elapsed": elapsed or "-",
            "min_per_design": ("%.1f" % (mins / a.designs)) if mins else "-",
            "max_rss": (s or {}).get("max_rss", "-") or "-",
            "cpus": (s or {}).get("alloc_cpus", "-"),
            "gpu_mem": ("%.1f" % gmem) if gmem else "-",
            "gpu_util": ("%.0f/%d" % (gutil, gpeak)) if gutil is not None else "-",
            "note": note,
        })

    hdr = ["tool", "state", "elapsed", "min_per_design", "max_rss", "cpus",
           "gpu_mem", "gpu_util"]
    titles = {"min_per_design": "min/design", "max_rss": "host RAM",
              "gpu_mem": "GPU GiB", "gpu_util": "GPU %mean/peak"}

    if a.markdown:
        print("| " + " | ".join(titles.get(h, h) for h in hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
        for r in rows:
            print("| " + " | ".join(str(r[h]) for h in hdr) + " |")
    else:
        widths = [max(len(titles.get(h, h)), max(len(str(r[h])) for r in rows))
                  for h in hdr]
        print("  ".join(titles.get(h, h).ljust(w) for h, w in zip(hdr, widths)))
        print("  ".join("-" * w for w in widths))
        for r in rows:
            print("  ".join(str(r[h]).ljust(w) for h, w in zip(hdr, widths)))
    # A FAILED state is not proof of failure here: a trailing `find | head` under
    # `set -o pipefail` fails the job after all real work succeeded (Genie 3 did
    # exactly this). Point at the log rather than letting the table mislead.
    failed = [r["tool"] for r in rows if r["state"].startswith("FAILED")]
    if failed:
        print()
        print("NOTE: %s report FAILED. Check the tool's own completion message in"
              % ", ".join(failed))
        print("      outputs/<tool>/logs/ before concluding the run failed --")
        print("      see docs/known-issues.md 6.1a.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
