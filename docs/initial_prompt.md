# Initial prompt

Verbatim record of the brief that started the alpha-stage benchmark documented
in `launching_scripts/` and `docs/benchmark-alpha.md`.

---

This is the beginning of a library to run multiple binder design models from the same place. I am currently at the alpha stage: I have installed most libraries and dedicated containers so that they can be run cleanly on the cluster.

The next step will be to run each of them for a small run to understand the input/output of each and the resources to require for each.

Your job is to test each model/framework to generate binders against the same target 40 binders are enough. Following I will give you more information on what to do and what resources we should use.

## Target

The target to use for this test is DIO3 cut version. Here you can find the .fasta and the MSA: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup/dio3_cut

We want 40 binders (1 single job) for each tool tested as we care about: commands to launch generation + output format of each tool.

## Tools

There are several tools that you can use for this first step and to validate:

### Generation Models:
These are models to use for generate the binders:

1. Mosaic: for mosaic use the hallucination based approach (I have already tested and it works well)
	Repository: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup/mosaic
	weights: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup/weights
	container: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images/mosaic.sif

2. Boltzgen:
	Repo:/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/boltzgen
	container: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images/boltzgen.sif

3. FreeBindCraft:
	Repo:/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/FreeBindCraft
	container: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images/freebindcraft.sif

4. Genie3:
	Repo:/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/genie3
	container: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images/genie3.sif

5. Proteina-Complex:
	Repo:/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/Proteina-Complexa
	container: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images/proteina_complexa.sif

6. Protein-Hunter:
	Repo:/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/Protein-Hunter
	container: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images/protein_hunter.sif

7. pxdesign:
	Repo:/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/PXDesign
	container: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images/pxdesign.sif

8. RFDiffusion:
	Repo:/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/RFDiffusion
	container: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images/rfd.sif

### Inverse Folding Models

1. Caliby:
	Repo:/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/caliby
	container: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images/caliby.sif

2. ProteinMPNN:
	For this model i think that it's quicker to refer to Mosaic and use their own implementation

### Folding models/Scoreer:

We have installed a bunch of them here: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/test_protforge. Protforge has all the right tools to launch jobs for them and the weights are here: /n/holylfs06/LABS/bsabatini_lab/Everyone/protforge-assets

Additionally I think that we can still use Mosaic as well for scoring as they have implemented a lot of models for folding as well so if Protforge does not have a model we can fallbakc to the Mosaic launching for now.

## HPC Policy and Usage

You may require GPU using SLURM to submit jobs. You should only require h100 (kempner_h100). max 10 jobs concurrent. account: kempner_bsabatini_lab .

It is important for us also to understand how much memory/gpu each model/library consumes, we can get the report using the followign commands:

- jobstats

This is very important so that moving forward we can calibrate the right resources for laucnhing bigger campaing.

Additionally it is important to understand time wise how much they need-> rank libraries to produce 40 bindrs.

## Outputs

Outputs shuld be saved in this directory:/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs

Each model/run shuold have its own subdirecotry for the outputs so that I can decide the postprocessing later and how to unify them.

## Delivarables

Your goal is to understand how to launch correct job for all the cofolding/generation models that I have given you. You may approach the challenge as you prefere, the imprtant is to following the best practice on the kempner cluster and to not save files outside the scope of this project.

Once it is finished I'd like the following:

1. This prompt saved as "initial_prompt.md" in the docs of bindocracy
2. All the configs used for each library + the sbatch/.sh scripts used to launch thme well documented and referring to the correct library in this repo under a new dir called: launching_scripts that I will use a reference
3. The generate binders defined in outputs section.
4. Performance and resources utilization for each binder generation module
5. Any serious bug that we should think about.

If you have any other question ask me before startin working on this.

Good luck and good job.
