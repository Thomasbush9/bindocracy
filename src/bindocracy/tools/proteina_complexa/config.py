"""Proteina-Complexa's authored configuration.

The tool ships a large Hydra tree and reads exactly one file from outside it:
a **target registry**, a nine-line YAML whose single entry names the structure,
the crop, the epitope and the binder length. Everything else the harness needs
is a `++key=value` override on the command line, so there is no driver here and
nothing is rendered per task.

The registry is authored and archived rather than generated, for the same
reason BoltzGen's spec and PXDesign's input spec are: it is what the run
consumes, and preflight is what makes it agree with the campaign. Its contents
are folded into the stored config so the database holds the science rather than
a path to it.

Three of the image's own defaults are wrong for a campaign and are answered
here as required or explicitly-defaulted fields rather than left alone:
`filter.filter_samples_limit` is 1000, `search.best_of_n.replicas` is 2, and
`dataloader.dataset.nres.nsamples` is 4 -- so the in-image defaults would
generate 8 designs and keep all of them, whatever this config said about how
many designs the run wanted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from bindocracy.config.models import ConfigModel, ToolConfig


class ProteinaComplexaRegistryConfig(ConfigModel):
    """The target registry, bound over the one the image ships.

    Proteina-Complexa resolves its target through `target_dict_cfg[task_name]`,
    an entry carrying `target_path`, `target_input` (an `A1-201` contig that
    crops the structure), `hotspot_residues`, and `binder_length`. Nothing on
    the command line can override a field inside it, so this file is the only
    place a campaign's target reaches the tool.

    `task_name` selects the entry and is also passed as
    `++generation.task_name`; the two must be the same string or the run
    designs against whichever entry the image happened to ship.
    """

    template: Path
    task_name: str = Field(min_length=1)
    # Filled in from `template` when the config is loaded, so the stored JSON
    # holds the registry itself rather than a path to it. Authors leave it out;
    # a recovered config carries it and still validates.
    contents: dict | None = None


class ProteinaComplexaSamplingConfig(ConfigModel):
    """What one task generates, and how much of it survives the reward filter.

    Generation and output are different numbers here. The tool draws
    `samples_per_job x nrepeat_per_sample x replicas` backbones, scores every
    one of them with AF2-multimer, and keeps the top `keep_per_job` by reward.
    That is where the cost is: AF2 runs on every candidate during generation
    and again on the survivors during evaluation.
    """

    jobs: int = Field(default=1, gt=0)
    # `generation.dataloader.dataset.nres.nsamples`. The in-image default is 4.
    samples_per_job: int = Field(gt=0)
    nrepeat_per_sample: int = Field(default=1, gt=0)
    # `generation.search.best_of_n.replicas`. The in-image default is 2, so
    # leaving it alone doubles the GPU cost of a run without doubling its
    # output. 1 is best-of-n with no search; a real campaign wants 2-4, which
    # is where this tool's quality comes from.
    replicas: int = Field(default=1, gt=0)
    # `generation.filter.filter_samples_limit`. No default: the image's is
    # 1000, which keeps everything and makes the filter stage a no-op that
    # still looks like it ran.
    keep_per_job: int = Field(gt=0)
    # `generation.dataloader.batch_size`, which also caps `search.max_batch_size`.
    batch_size: int = Field(default=8, gt=0)
    # Passed as `++seed`. Each task gets seed_base + task_id: two tasks sharing
    # a seed would draw the same backbones and the run would return duplicates
    # without any of its counts changing.
    seed_base: int = Field(ge=0)
    # The AF2 reward model carries its own `seed`, hard-coded to 0 in
    # binder_generate.yaml and unaffected by `++seed`. It is not the sampling
    # seed and does not need to vary per task, but a run that does not record
    # it cannot say what scored its candidates.
    reward_seed: int = Field(default=0, ge=0)


class ProteinaComplexaRuntimeConfig(ConfigModel):
    container: Path
    # Node-local scratch root. TMPDIR must NOT be on Lustre; see
    # docs/known-issues.md section 2.1.
    node_tmp_root: Path = Path("/tmp")
    # AF2 is JAX and the generative model is PyTorch, on the same device.
    # Without this JAX preallocates ~75% of the GPU at import and starves
    # torch, which fails as an out-of-memory error from the wrong library.
    xla_preallocate: bool = False


class ProteinaComplexaConfig(ToolConfig):
    schema_version: Literal[1]
    tool: Literal["proteina_complexa"]
    registry: ProteinaComplexaRegistryConfig
    sampling: ProteinaComplexaSamplingConfig
    runtime: ProteinaComplexaRuntimeConfig
