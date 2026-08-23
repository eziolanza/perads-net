# PERADS.net v0.1.0 — research preview

Research pipeline for one contrast-enhanced chest CT NIfTI. It creates the
Dataset546-compatible thoracic crop and artery prior, performs the five-fold
nnU-Net ensemble, calculates RV/LV, assigns PE-RADS, and renders a preview.

## Installation

The repository is private: users must have access to GitHub `eziolanza/perads-net`.

```bash
git clone git@github.com:eziolanza/perads-net.git
cd perads-net
pip install -r requirements.txt
```

Download the private model bundle from the `v0.1.0` release:

```bash
wget https://github.com/eziolanza/perads-net/releases/download/v0.1.0/PERADS.net-model-v0.1.0.tar.zst
tar --use-compress-program=unzstd -xf PERADS.net-model-v0.1.0.tar.zst
```

The archive automatically creates the `models/` directory required by the runner.
An NVIDIA GPU with CUDA is also required, together with the
`TotalSegmentator` and `nnUNetv2_predict_from_modelfolder` commands available
on the system.

The model bundle contains only the weights required for inference; the training
dataset and Dataset 120 do not need to be downloaded.

Run on the local GPU, outside the sandbox:

```bash
python3 run_perads_case.py \
  --input /absolute/path/to/ct.nii.gz \
  --output /absolute/path/to/output_case \
  --device cuda
```

Key outputs in `04_results/`:

- `embolus_segmentation.nii.gz`
- `result.json` — PE-RADS grade, anatomical level, RV/LV and diameters
- `arterial_hierarchy.nii.gz`
- `rv_lv/rv_lv_v2_same_slice.json`
- `preview.png` — embolus and RV/LV review panels

The RV/LV v2 implementation is included locally in `rvlv_v2.py` and is the
same method used for Dataset 120. Remaining requirements are the local
`TotalSegmentator` and `nnUNetv2_predict_from_modelfolder` commands; the trained model at
`models/Dataset546_PEArteryPrior/nnUNetTrainer__nnUNetPlans__3d_fullres`.
The script accepts `--model` to override this location.

This is a research pipeline and requires visual QC; it is not for clinical use.
