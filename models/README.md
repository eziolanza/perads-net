# PERADS.net model files

The local model bundle contains the five `checkpoint_final.pth` files and the
`dataset.json`/`plans.json` metadata required for inference. The checkpoints
are intentionally excluded from the Git repository because they are large;
distribute this `models/` directory separately or through a release artifact.

Expected model path:

```text
models/Dataset546_PEArteryPrior/nnUNetTrainer__nnUNetPlans__3d_fullres/
```
