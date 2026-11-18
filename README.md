# YOLO-RD-T-26 v2 — Road Defect Detection with Attention on a YOLO26 Base

Road damage detection on the **Refine N-RDD2024** dataset. This project rebuilds
YOLO-RD (Wang et al., *Neurocomputing* 662, 2026, 132000) on **YOLO26s** and extends
it with a bottleneck Vision Transformer and a new **Crack Strip Attention (CSA)** block.

## Results (validation set, 3,704 images, 11,011 instances)

| Model | Params (fused) | GFLOPs | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| YOLO-RD (paper reference) | 6,455,256 | 24.0 | — | — | 51.0 | 45.5 | 26.2 |
| **YOLO-RD-T-26 v2 (this repo)** | **5,018,703** | **17.3** | **68.0** | **48.3** | **56.5** | **52.9** | **30.1** |

Trained 50 epochs with a COCO warm start. Inference: 11.5 ms/image on a Tesla T4.

<details>
<summary>Per-class AP@0.5</summary>

| Class | Description | AP@0.5 |
|---|---|---|
| D00 | Longitudinal crack | 58.0 |
| D10 | Transverse crack | 55.8 |
| D20 | Alligator crack | 64.8 |
| D30 | Repaired crack | 38.5 |
| D40 | Pothole | 51.2 |
| D50 | Pedestrian crossing blur | 64.3 |
| D60 | Lane line blur | 48.8 |
| D70 | Manhole cover | 81.3 |
| D80 | Patchy road section | 64.6 |
| D90 | Rutting (only 3 val instances) | 2.0 |

</details>

## What's in the model

| Component | Source | Purpose |
|---|---|---|
| **CSAF** | Paper | Replaces the first backbone conv; fuses a strided conv with a lossless Space-to-Depth branch so thin cracks survive downsampling |
| **LGECA** | Paper | ~10-parameter global + local channel attention before each detection head |
| **SR-WBCE** | Paper | Square-root-weighted BCE loss for class imbalance |
| **LFC** | Paper | Channel compression on the P5 branch |
| **Bottleneck Transformer** | This work | Replaces `C2PSA` (256→128→attend→256, zero-init residual) instead of stacking on it |
| **CSA (Crack Strip Attention)** | This work | Axis-wise strip pooling on the fine scales to help long, thin cracks |

Ablation switches (`USE_CSAF`, `USE_LGECA`, `USE_TRANSFORMER`, `USE_CSA`, `USE_SRWBCE`,
`USE_P2_HEAD`) are at the top of Section 6 of the notebook.

## Repository structure

```
.
├── notebooks/
│   └── yolo-rd-t-26-v2_rdd2024.ipynb   # full pipeline: data → model → train → eval → plots
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

[Refine N-RDD2024 — Road Damage and Defects](https://www.kaggle.com/datasets/zeziarehman/refine-n-rdd2024road-damage-and-defects)
(Kaggle). The notebook merges the per-country folders into one YOLO-format dataset and
makes the train/val split itself. The dataset is not included in this repo.

## How to run

### On Kaggle (recommended — this is how the notebook was built)
1. Upload the notebook to Kaggle and add the dataset above as an input.
2. Turn on a GPU accelerator (T4 is enough) and Internet.
3. **Run → Run All.** If the YOLO26 config check fails, use **Restart & Run All** once.

### Locally
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt
jupyter notebook notebooks/yolo-rd-t-26-v2_rdd2024.ipynb
```
The notebook uses Kaggle paths (`/kaggle/input`, `/kaggle/working`). Locally, change
`DATASET_HINT` (Section 2) and `WORK` (Section 1) to your own folders.

## Loading the trained weights

`best.pt` stores the custom layers (`CSAF`, `LGECA`, `CSA`, the transformer) **by reference**.
Run the Section 4 cell (or import the `restore_modules.py` produced in Section 26) before
`YOLO("best.pt")`, or PyTorch raises `Can't get attribute 'CSAF'`.

Weights are not stored in git. Download them from this repo's **Releases** page.

## Citation

```bibtex
@article{wang2026yolord,
  title   = {YOLO-RD: Road defect detection with context-aware attention and balanced loss},
  author  = {Wang and others},
  journal = {Neurocomputing},
  volume  = {662},
  pages   = {132000},
  year    = {2026}
}
```

## Acknowledgements

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
