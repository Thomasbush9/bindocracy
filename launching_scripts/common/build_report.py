#!/usr/bin/env python3
"""Assemble the shareable benchmark report from the template + measured figures.

Keeps the report reproducible: re-run make_figures.py, then this, and the page
carries the current numbers rather than hand-copied ones. The figures are
embedded as base64 data URIs because a published artifact must be self-contained
(no external hosts load).

    build_report.py <template.html> <out.html>

Python 3.6 compatible, stdlib only.
"""

import base64
import json
import os
import sys

BR = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design"
FIGDIR = os.path.join(BR, "bindocracy", "docs", "figures")

FIGS = [("__FIG1__", "01-walltime.svg"), ("__FIG1D__", "01-walltime-dark.svg"),
        ("__FIG2__", "02-host-memory.svg"), ("__FIG2D__", "02-host-memory-dark.svg"),
        ("__FIG3__", "03-gpu-memory.svg"), ("__FIG3D__", "03-gpu-memory-dark.svg"),
        ("__FIG4__", "04-gpu-utilisation.svg"), ("__FIG4D__", "04-gpu-utilisation-dark.svg"),
        ("__FIG5__", "05-batch-scaling.svg"), ("__FIG5D__", "05-batch-scaling-dark.svg")]


def data_uri(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


def main():
    tpl_path, out_path = sys.argv[1], sys.argv[2]
    html = open(tpl_path).read()

    for token, name in FIGS:
        p = os.path.join(FIGDIR, name)
        if not os.path.exists(p):
            sys.exit("FATAL: missing figure %s — run make_figures.py first" % p)
        html = html.replace(token, data_uri(p))

    # Headline RAM numbers come from the same JSON the figures were built from,
    # so the page and the plots can never disagree.
    d = json.load(open(os.path.join(FIGDIR, "figure-data.json")))
    req = sum(v["req_mem_gb"] for v in d.values()
              if v.get("req_mem_gb") and v.get("max_rss_gb"))
    use = sum(v["max_rss_gb"] for v in d.values()
              if v.get("req_mem_gb") and v.get("max_rss_gb"))
    html = html.replace("__RAM_PCT__", "%.0f" % (100.0 * use / req))
    html = html.replace("__RAM_USED__", "%.0f" % use)
    html = html.replace("__RAM_REQ__", "%.0f" % req)

    # Say plainly which tools are still running, rather than letting the page
    # imply the figures are the whole story.
    done = sorted(k for k, v in d.items() if v.get("elapsed_min"))
    pending = sorted(k for k, v in d.items() if not v.get("elapsed_min"))
    if pending:
        status = ('<p class="running">%d of %d tools complete &middot; still running: %s</p>'
                  % (len(done), len(d), ", ".join(pending)))
    else:
        status = ""
    html = html.replace("__STATUS__", status)

    leftover = [t for t, _ in FIGS if t in html] + \
               [t for t in ("__RAM_PCT__", "__RAM_USED__", "__RAM_REQ__", "__STATUS__")
                if t in html]
    if leftover:
        sys.exit("FATAL: unsubstituted tokens: %s" % ", ".join(leftover))

    with open(out_path, "w") as fh:
        fh.write(html)
    print("wrote %s  (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))
    print("host RAM: %.0f GB used of %.0f GB reserved (%.0f%%)"
          % (use, req, 100.0 * use / req))
    return 0


if __name__ == "__main__":
    sys.exit(main())
