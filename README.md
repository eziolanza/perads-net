# PERADS.net v0.1.0 — research preview

Research pipeline for one contrast-enhanced chest CT NIfTI. It creates the
Dataset546-compatible thoracic crop and artery prior, performs the five-fold
nnU-Net ensemble, calculates RV/LV, assigns PE-RADS, and renders a preview.

## Installazione

Il repository è privato: è necessario avere accesso a GitHub `eziolanza/perads-net`.

```bash
git clone git@github.com:eziolanza/perads-net.git
cd perads-net
pip install -r requirements.txt
```

Scaricare quindi il bundle privato del modello dalla release `v0.1.0`:

```bash
wget https://github.com/eziolanza/perads-net/releases/download/v0.1.0/PERADS.net-model-v0.1.0.tar.zst
tar --use-compress-program=unzstd -xf PERADS.net-model-v0.1.0.tar.zst
```

L’archivio crea automaticamente la cartella `models/` richiesta dal runner.
Sono necessari anche una GPU NVIDIA con CUDA e i comandi `TotalSegmentator`
e `nnUNetv2_predict_from_modelfolder` disponibili nel sistema.

Il modello contiene esclusivamente i pesi necessari all’inferenza; non è
necessario scaricare il dataset di addestramento o il Dataset 120.

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
