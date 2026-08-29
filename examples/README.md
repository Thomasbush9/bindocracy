# Config templates

One template per module. **Real configs do not live here** — they live beside
the campaign data (`binder_design/configs/`), because their absolute paths
belong to one machine and one lab share, and the database, not a file, is the
durable record of what a run used.

| File | Copy it to |
|---|---|
| `general_config.example.yaml` | `configs/general_config.yaml` — the campaign target and cluster |
| `mosaic.example.yaml` | `configs/mosaic/<name>.yaml` |
| `boltzgen.example.yaml` | `configs/boltzgen/<name>.yaml` |
| `boltzgen_spec.example.yaml` | the file `spec.template` points at |
| `../workflow/campaign.example.yaml` | an execution index beside your runs |

A model config selects its own plugin through its `tool:` field, so adding a
tool means adding a template here and a package under `src/bindocracy/tools/`.
