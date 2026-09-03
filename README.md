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

1. **TotalSegmentator x2 or x3, all on the same raw CT** (`lung_vessels`,
   `total -rs <5 lobe names>`, and — unless `--skip-rvlv` is passed —
   `heartchambers_highres`, which requires a free TotalSegmentator license;
   see [RV/LV ratio (optional)](#rvlv-ratio-optional) below). Running every
   task on the *same* uncropped input means the arteries, heart-chamber and
   lobe masks are all natively co-registered with each other and the CT —
   no resampling step between them.
2. **Crop once**, only for the nnU-Net input (CT + dilated artery prior),
   over the union of the lung airway/artery/vein masks.
3. **5-fold nnU-Net inference** on the small crop.
4. **Align** the small embolus prediction back into the full-resolution
   arteries grid (pure-translation offset recovered from the NIfTI
   affines).
5. **Named-artery classification**: a skan branch-tree over the
   pulmonary arteries, named-structure assignment per branch, a 70 mm³
   minimum embolus volume, and overlap-hierarchy grading (the most
   proximal PE-RADS class holding ≥1% of the embolus volume). The raw
   nnU-Net prediction is used directly for volume and grading; anatomic
   siting still only counts voxels that overlap the named-structure
   labelmap, which is itself nonzero only inside the artery tree. This
   step, and everything before it, needs no TotalSegmentator license.
6. **RV/LV ratio** (optional, skipped with `--skip-rvlv`), four-chamber
   method, on the full-resolution CT and heart-chamber masks.
7. Per case: `result.json`, `structure_labelmap.nii.gz`, and one row
   appended to a CSV (`case_id, perads_grade, anatomic_level,
   most_proximal_structures, embolic_voxels, bcv_mm3,
   classification_reason, rv_lv_ratio, rv_diameter_mm, lv_diameter_mm,
   num_branches, duration_sec`). With `--skip-rvlv`, the three `rv_lv_*`
   fields are empty.

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

## RV/LV ratio (optional)

The RV/LV step needs TotalSegmentator's `heartchambers_highres` task, which
is **not** covered by TotalSegmentator's default license-free tasks — it
requires requesting a free license from the TotalSegmentator authors
(see licensing instructions in the
[TotalSegmentator repository](https://github.com/wasserth/TotalSegmentator)).
Without that license, `heartchambers_highres` will fail.

**PE-RADS grading does not need it.** If you only want the PE-RADS grade
and anatomic site (no RV/LV ratio), pass `--skip-rvlv`: this skips
`heartchambers_highres` entirely, so no TotalSegmentator license is
required at all.

```bash
# PE-RADS grade only, no TotalSegmentator license needed
python3 perads_net_pipeline.py \
  --input /absolute/path/to/raw_ct.nii.gz \
  --output /absolute/path/to/output_case \
  --device cuda --skip-rvlv
```

## Usage

Single case, full pipeline (PE-RADS grade + RV/LV ratio; requires the
`heartchambers_highres` license above):

```bash
python3 perads_net_pipeline.py \
  --input /absolute/path/to/raw_ct.nii.gz \
  --output /absolute/path/to/output_case \
  --device cuda
```

Single case, PE-RADS grade only (no RV/LV, no TotalSegmentator license
needed — see [RV/LV ratio (optional)](#rvlv-ratio-optional)):

```bash
python3 perads_net_pipeline.py \
  --input /absolute/path/to/raw_ct.nii.gz \
  --output /absolute/path/to/output_case \
  --device cuda --skip-rvlv
```

Batch (manifest CSV with a `ct_path` column, or a directory to scan);
add `--skip-rvlv` here too for grade-only batch runs:

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
