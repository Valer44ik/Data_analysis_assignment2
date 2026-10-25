# Group 2 — Emotion recognition with classical ML (FER2013 vs KDEF)

Course 5ARE0, Assignment 2 (2026). Gabriele Salvo, Valērijs Gončarovs, Herman Chang.

## Contents

| file | what |
|---|---|
| `demo_emotion.py` | **Task 6 — real-time webcam demo** (`python demo_emotion.py`) |
| `z_load_split_fer2013.ipynb` | kit notebook: FER2013 CSV → train / validation (PublicTest) / test (PrivateTest) CSVs |
| `z_load_kdef_student.ipynb` | kit notebook: KDEF frontal images → `data/kdef_preprocessed.pkl` (subject split) |
| `00_data_preparation.ipynb` | image arrays, landmarks, KDEF variants (margins, RGB, alignment), dataset exploration |
| `01_fer_baseline.ipynb` | **Task 1** — LR / SVM / RF × HOG / landmarks / PCA / HOG+LM on FER2013, gray vs black-and-white study, error analysis |
| `02_kdef_study.ipynb` | **Task 2** — same grid on KDEF (subject-independent), gray vs RGB, ranking changes |
| `03_cross_dataset.ipynb` | **Task 3** — FER→KDEF and KDEF→FER, feature transfer, domain-shift measurement and adaptation |
| `04_fuzzy_kdef.ipynb` | **Task 4** — pyFUME Takagi–Sugeno fuzzy classifier on 6 geometric features (**fuzzy environment**) |
| `05_creative_challenge.ipynb` | **Task 5** — tuning, balancing, augmentation, KDEF geometric features & prototypes, label curation, FER+KDEF merge |
| `src/features.py` | preprocessing and feature extraction (the assignment allows feature extraction outside notebooks) |
| `GPUKernelSVC` (in notebooks 01, 02, 03, 05 and `demo_emotion.py`) | RBF-SVM whose kernel matrix is computed with PyTorch on the GPU (CUDA); same libsvm solver and identical predictions as `SVC`, ~30x faster. The demo maps the old name `src.gpu_svm` to its own copy so the saved models load |
| `models/*.pkl` | fitted models; each `.pkl` is a dict `{model, spec, labels, ...}` where `spec` describes the features |
| `results/*.csv`, `results/*.tex` | every metric of every experiment (the report tables are generated from these) |
| `figures/*.png` | figures used in the report and appendix |
| `faceAndLandmarkModels/` | Haar cascade + dlib 68-landmark predictor |
| `auxFiles/kdef_subject_split.pkl` | official (corrected) KDEF subject split |

## Environments

Two environments are needed because pyFUME requires `numpy==1.24.4`
(Python 3.11; `uv` shown, plain `python -m venv` + `pip` works the same way):

```bash
# main environment (all notebooks except 04, and the demo)
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build (GPU SVM)
.venv/Scripts/python -m ipykernel install --user --name ass2 --display-name "Python (ass2)"

# fuzzy environment (notebook 04 only)
uv venv --python 3.11 .venv-fuzzy
uv pip install --python .venv-fuzzy -r requirements_fuzzy.txt
.venv-fuzzy/Scripts/python -m ipykernel install --user --name ass2-fuzzy --display-name "Python (ass2-fuzzy)"
```

Note: pyFUME **0.3.4** is required (newer releases break first-order Takagi–Sugeno models).

## Data (not included in the zip)

As in the kit, paste the datasets under `data/`:

```
data/fer/fer2013.csv                              # Kaggle FER2013 (columns emotion, pixels, Usage)
data/KDEF_and_AKDEF/KDEF/AF01/AF01ANS.JPG ...     # KDEF release (unzipped KDEF_and_AKDEF.zip)
```

## Run order

1. `z_load_split_fer2013.ipynb`, `z_load_kdef_student.ipynb` (kit preprocessing)
2. `00_data_preparation.ipynb` (writes `data/cache/`)
3. `01` → `02` → `03` (kernel *Python (ass2)*)
4. `04_fuzzy_kdef.ipynb` (kernel *Python (ass2-fuzzy)*)
5. `05_creative_challenge.ipynb` (kernel *Python (ass2)*) → writes `models/final.pkl`
6. `python demo_emotion.py`

Each notebook reads the outputs of the previous ones (`results/task*_best.json`, `data/cache/`).
Command-line execution: `jupyter nbconvert --to notebook --execute --inplace 01_fer_baseline.ipynb`.

**Quick check mode:** with the environment variable `SMOKE=1` every notebook runs on small
subsets with reduced grids (minutes instead of hours). The delivered results are the full run
(`SMOKE` unset).

**Run time (full):** the SVM kernel matrices are computed on the GPU (CUDA, via PyTorch; CPU
fallback without a GPU); HOG, dlib, LR and RF use all CPU cores. Notebooks 01 and 05 are the longest.

## Protocol summary

* FER2013: Training for fitting, PublicTest for every model-selection decision, PrivateTest only
  for reporting. The test set is never filtered.
* KDEF: frontal pose only, official 56/14 subject split; tuning with GroupKFold over subjects.
* All KDEF images are converted to the FER format (Haar crop + margin → gray → 48×48);
  normalization happens identically for both datasets inside the feature extractors.
* Metrics: accuracy, balanced accuracy, macro-F1 and confusion matrices.

## Demo

```bash
python demo_emotion.py                 # models/final.pkl (fallback: best Task 1 model), camera 0
python demo_emotion.py --camera 1 --smooth 15 --margin 0.1
python demo_emotion.py --image face.jpg
```

## Use of AI

During the preparation of this work, we used Claude Code (Anthropic, Claude Opus 5.5) to
[REASON]. The following parts of the assignment were affected/generated by AI tool usage: [PARTS].
After using this tool/service, [NAME STUDENT(S)] evaluated the validity of the tool's outputs,
including the sources that generative AI tools have used, and edited the content as needed. As a
consequence, [NAME STUDENT(S)] take(s) full responsibility for the content of their work.
