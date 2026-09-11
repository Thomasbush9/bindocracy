#!/usr/bin/env python3
"""Container CLI: immutable embedded assets, no inference-time HTTP requests."""

import os
import sys
import tempfile
from pathlib import Path


def main():
    for argument in sys.argv[1:]:
        if argument.split("=", 1)[0] in {"--use-msa-server", "--use-templates-server"}:
            raise SystemExit(
                "Network services are disabled in this offline image. Supply --msa-directory with local .aligned.pqt files instead."
            )
    # A unique directory avoids user/job collisions; scratch cleanup is the job's
    # responsibility when CHAI_RUNTIME_DIR is explicitly supplied.
    runtime = Path(
        os.environ.get("CHAI_RUNTIME_DIR") or tempfile.mkdtemp(prefix="chai1-")
    )
    runtime.mkdir(parents=True, exist_ok=True)
    for variable, directory in {
        "XDG_CACHE_HOME": "xdg",
        "TORCH_HOME": "torch",
        "HF_HOME": "huggingface",
        "MPLCONFIGDIR": "matplotlib",
        "NUMBA_CACHE_DIR": "numba",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
    }.items():
        location = runtime / directory
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)
    os.environ["CHAI_DOWNLOADS_DIR"] = "/opt/chai-lab/downloads"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import requests

    from chai_lab.utils import paths

    def require_local(http_url, path):
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise FileNotFoundError(
                f"Required offline asset is absent: {path}; image must be rebuilt from complete, verified downloads"
            )

    def deny_http(self, method, url, *args, **kwargs):
        raise RuntimeError(
            f"HTTP is disabled in the Chai offline image: {method} {url}"
        )

    paths.download_if_not_exists = require_local
    requests.sessions.Session.request = deny_http
    # Import after installing the offline policy so imported function aliases
    # also use the local-only resolver (including ESM and template downloads).
    from chai_lab.main import cli

    cli()


if __name__ == "__main__":
    main()
