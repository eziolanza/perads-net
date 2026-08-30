# PERADS.net

End-to-end PE-RADS classification pipeline for one contrast-enhanced chest
CT NIfTI, **raw and uncropped**. It segments the pulmonary arterial tree,
heart chambers and lung lobes with TotalSegmentator; crops and runs an
artery-prior-conditioned nnU-Net ensemble for embolus segmentation;
classifies PE-RADS with a named-artery branch-tree method (main pulmonary
trunk / right or left PA / lobar / segmental / subsegmental, per lung
lobe); computes the RV/LV ratio with a four-chamber method; and writes one
CSV row per case.

This is the reference implementation for the method described in the
accompanying publication.

## Method summary

1. **TotalSegmentator x3, all on the same raw CT** (`lung_vessels`,
   `heartchambers_highres`, `total -rs <5 lobe names>`). Running every task
   on the *same* uncropped input means the arteries, heart-chamber and
   lobe masks are all natively co-registered with each other and the CT —
   no resampling step between them.
2. **Crop once**, only for the nnU-Net input (CT + dilated artery prior),
   over the union of the lung airway/artery/vein masks.
3. **5-fold nnU-Net inference** on the small crop.
4. **Align** the small embolus prediction back into the full-resolution
   arteries grid (pure-translation offset recovered from the NIfTI
   affines).
5. **Named-artery classification**: a skan branch-tree over the
   pulmonary arteries, named-structure assignment per branch, hard-mask
   embolus intersection, a 30 mm³ minimum embolus volume, and
   overlap-hierarchy grading (the most proximal PE-RADS class holding
   ≥1% of the embolus volume).
6. **RV/LV ratio**, four-chamber method, on the full-resolution CT and
   heart-chamber masks.
7. Per case: `result.json`, `structure_labelmap.nii.gz`, and one row
   appended to a CSV (`case_id, perads_grade, anatomic_level,
   most_proximal_structures, embolic_voxels_hardmask, bcv_mm3,
   classification_reason, rv_lv_ratio, rv_diameter_mm, lv_diameter_mm,
   num_branches, duration_sec`).

## Installation

```bash
git clone git@github.com:eziolanza/perads-net.git
cd perads-net
pip install -r requirements.txt
```

An NVIDIA GPU with CUDA is required, together with `TotalSegmentator` and
`nnUNetv2_predict_from_modelfolder` available on the system.

Download the trained model bundle from this repository's GitHub Releases
page:

```bash
wget https://github.com/eziolanza/perads-net/releases/download/v0.2.0/PERADS.net-model-v0.2.0.tar.zst
tar --use-compress-program=unzstd -xf PERADS.net-model-v0.2.0.tar.zst
```

This extracts an `nnUNetTrainer__nnUNetPlans__3d_fullres/` folder
(`dataset.json`, `plans.json`, `fold_0` .. `fold_4`). Point the pipeline
at it with `--model /path/to/that/folder` or:

```bash
export PERADS_NET_MODEL=/path/to/nnUNetTrainer__nnUNetPlans__3d_fullres
```

## Usage

Single case:

```bash
python3 perads_net_pipeline.py \
  --input /absolute/path/to/raw_ct.nii.gz \
  --output /absolute/path/to/output_case \
  --device cuda
```

Batch (manifest CSV with a `ct_path` column, or a directory to scan):

```bash
python3 batch_perads_net_pipeline.py \
  --manifest cases.csv \
  --output-root /absolute/path/to/batch_output \
  --workers 4 --device cuda
```

Both re-use per-case TotalSegmentator/nnU-Net outputs already on disk
(idempotent), so a re-run after a crash or an `--overwrite` pass only
resumes the missing steps.

## Disclaimer

This is a research pipeline. It has not been validated for clinical use
and requires expert visual review of every case. Do not use it to guide
patient management.

## License

CC BY-NC 4.0 — see [LICENSE](LICENSE). Non-commercial use with
attribution.

## Citation

If you use this pipeline, please cite the accompanying publication
(citation to be added on acceptance).
