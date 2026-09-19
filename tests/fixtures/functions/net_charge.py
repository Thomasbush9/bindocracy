#!/usr/bin/env python3
"""A worked example of a custom scoring function.

Reads the input JSONL, writes the output JSONL. Nothing here imports
bindocracy, which is the point: a scoring function is two files and any
language, not a plugin.
"""

import argparse
import json


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    p.add_argument("--ph", type=float, default=7.0)
    args = p.parse_args()

    with open(args.inputs) as source, open(args.outputs, "w") as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seq = row.get("sequence", "")
            if not seq:
                sink.write(json.dumps(
                    {"index": row["index"], "failed": "no sequence"}) + "\n")
                continue
            charge = sum(seq.count(a) for a in "KR") - sum(seq.count(a) for a in "DE")
            sink.write(json.dumps({
                "index": row["index"],
                "metrics": {"net_charge_at_ph": float(charge),
                            "fraction_charged": round(
                                sum(seq.count(a) for a in "KRDE") / len(seq), 4)},
            }) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
