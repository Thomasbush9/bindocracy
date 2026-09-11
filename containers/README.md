# Container definitions

The recipes for the images this harness builds, kept here rather than only in
the upstream clone they were authored beside.

That is the whole reason this directory exists. `chai1.def` and `af3.def` sat
untracked inside `chai-lab/` and `alphafold3/` — clones of other people's
repositories — where a `git clean` or a re-clone would have destroyed the only
record of how two images the campaign depends on were built. The image is 10 GB
and 4.1 GB respectively; the recipe is 6 KB.

| file | builds | notes |
|---|---|---|
| `chai1.def` | `images/chai1.sif` (10 GB) | Chai-1 v0.6.1. Assets **embedded** — Apache-2.0 permits it. |
| `chai1-build-backend.lock` | — | Pinned setuptools/wheel for `chai1.def`'s two sdists. |
| `chai1_offline.py` | — | The image's entry point: refuses network MSA/template services. |
| `af3.def` | `images/af3.sif` (4.1 GB) | AlphaFold 3. Weights **bound, not embedded** — the terms restrict distribution. |

Not here: `mosaic.def` lives in the mosaic checkout, which is a repository in
its own right rather than a third-party clone, and `openfold3.sif` is
ProtForge's build of an official upstream image that we did not author.

## Building

Both need a CPU allocation with network access for OS and Python packages, and
neither should be built on a login node — the process arbiter kills the
`mksquashfs` step.

```bash
# Chai-1: build from the chai-lab clone root, which holds the assets %files copies.
cd .../binder_design/chai-lab && cp .../bindocracy/containers/chai1.def .
singularity build --fakeroot ../images/chai1.sif chai1.def

# AlphaFold 3: build from the alphafold3 clone root.
cd .../binder_design/alphafold3 && cp .../bindocracy/containers/af3.def .
singularity build --fakeroot ../images/af3.sif af3.def
```

The `%files` sections copy from the clone, so the build has to run there. Copy
the definition in rather than editing it in place, and edit it back here.

## Two things learned building these

**Pin the build environment, not just the runtime.** `chai1.def` pins its base
image by digest, `uv`, CPython and all 113 Python packages by hash — and then
`uv` resolved setuptools fresh from PyPI to compile the two sdists that have no
wheels, fetching 84.0.0 while the lock said 75.8.0. The one environment that
compiles C into the image was the one left unpinned.
`chai1-build-backend.lock` closes that.

**A `%test` should assert what must be absent, not only what must be present.**
`af3.def` checks that `af3.bin.zst` is *not* in the image, so a future build
that quietly embeds the restricted weights fails at build time rather than
shipping them.
