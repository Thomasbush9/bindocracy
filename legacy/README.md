# Legacy launchers

The hand-written `sbatch` scripts from the 2026-08 alpha benchmark, one per
tool. **The harness does not use these.** They are kept because each one is a
worked example of what a tool actually needs, and because every comment in them
marks a place where a tool did something the others do not — which is the
material `docs/harness-design.md` was written from.

The Mosaic driver that started here now lives in `../drivers/mosaic/`, because
the harness runs it.

`common/` is still worth reading: `env.sh` records the cross-cutting container
fixes (node-local `TMPDIR`, the CA bundle, forwarding `CUDA_VISIBLE_DEVICES`)
that `docs/known-issues.md` explains.
