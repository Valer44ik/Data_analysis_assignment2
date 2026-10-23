# How to run everything (step by step)

Goal: run all notebooks in full on your machine, then commit the outputs (`models/`, `results/`,
`figures/` and the executed notebooks) back to the `final` branch. All commands are for Windows
PowerShell and run from the repository root unless stated otherwise.

---

## 0. What you need

| item | where to get it |
|---|---|
| Python **3.11** | python.org, or conda (`conda create -n ...` below) |
| **FER2013** `fer2013.csv` (≈301 MB, columns `emotion,pixels,Usage`) | Kaggle FER2013 (the single-CSV version), or ask Gabriele for his copy |
| **KDEF** `KDEF_and_AKDEF.zip` (≈524 MB, the main KDEF release) | kdef.se → download (registration), or ask Gabriele for his copy |

From kdef.se you **only** need `KDEF_and_AKDEF.zip`. Skip KDEF-dyn I/II (videos),
KDEF-chimeric and KDEF-cropped.

---

## 1. Get the code

```powershell
git fetch origin
git checkout final
git pull
cd group_2_emotion_recognition
```

From here on, every command runs **inside `group_2_emotion_recognition/`**.

---

## 2. Put the datasets in `data/`

```
group_2_emotion_recognition/
└── data/
    ├── fer/
    │   └── fer2013.csv
    └── KDEF_and_AKDEF/
        └── KDEF/
            ├── AF01/AF01AFFL.JPG ...
            └── ... (140 folders, 4,900 JPGs)
```

```powershell
mkdir data\fer
copy C:\path\to\fer2013.csv data\fer\fer2013.csv
Expand-Archive C:\path\to\KDEF_and_AKDEF.zip -DestinationPath data
# check: should print 4900
(Get-ChildItem -Recurse data\KDEF_and_AKDEF\KDEF -Filter *.JPG).Count
```

`data/` is in `.gitignore`. **Never commit the datasets.**

---

## 3. Create the two Python environments

There are two environments because pyFUME (fuzzy notebook 04) needs `numpy==1.24.4`, which
would break the other notebooks. Choose **A (venv)** or **B (conda)**.

### A. Plain venv

```powershell
# main environment
py -3.11 -m venv ..\.venv
..\.venv\Scripts\python -m pip install -r requirements.txt
..\.venv\Scripts\python -m ipykernel install --user --name ass2 --display-name "Python (ass2)"

# fuzzy environment
py -3.11 -m venv ..\.venv-fuzzy
..\.venv-fuzzy\Scripts\python -m pip install -r requirements_fuzzy.txt
..\.venv-fuzzy\Scripts\python -m ipykernel install --user --name ass2-fuzzy --display-name "Python (ass2-fuzzy)"
```

### B. Conda

```powershell
conda create -n ass2 python=3.11 -y
conda run -n ass2 pip install -r requirements.txt
conda run -n ass2 python -m ipykernel install --user --name ass2 --display-name "Python (ass2)"

conda create -n ass2-fuzzy python=3.11 -y
conda run -n ass2-fuzzy pip install -r requirements_fuzzy.txt
conda run -n ass2-fuzzy python -m ipykernel install --user --name ass2-fuzzy --display-name "Python (ass2-fuzzy)"
```

**The kernel names must be exactly `ass2` and `ass2-fuzzy`**, because the notebooks are saved
with these kernels.

### GPU (CUDA) for the SVMs — main environment only

The RBF-SVMs compute their kernel matrix with PyTorch, on the GPU when CUDA is available
(`src/gpu_svm.py`: same model and identical predictions as scikit-learn's SVC, much faster).
Install the **CUDA build** of PyTorch into the main environment:

```powershell
# venv:  ..\.venv\Scripts\python -m pip install ...      conda:  conda run -n ass2 pip install ...
..\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
..\.venv\Scripts\python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

It must print `CUDA: True` and your GPU name. If it prints `False`, run `nvidia-smi`. Then install
the build for your driver from the selector at https://pytorch.org/get-started/locally/
(pip, Windows, CUDA 12.x). Each notebook prints `kernel computations on: cuda` at the start.
Without a GPU everything still works (on CPU), just slower.

RAM: an SVM on 33k images keeps a 33k × 33k kernel matrix (8.7 GB) in memory, so 16 GB of RAM or
more is needed. With less RAM, set `$env:MAX_KERNEL_N = "15000"` before running (slower, slightly worse).

**Check before the long run:** the first cells of notebooks 01, 02, 03 and 05 must print
`kernel computations on: cuda`. In the previous run notebook 01 printed nothing of the kind and
its SVM grids took 22 min each on the CPU. If you see `cpu`, the CUDA build of PyTorch is not
installed in the kernel's environment: fix that first.

Check the environments:

```powershell
..\.venv\Scripts\python -c "import dlib, cv2, sklearn, imblearn, skimage; print('main ok', sklearn.__version__)"
..\.venv-fuzzy\Scripts\python -c "import pyfume, numpy, importlib.metadata as m; print('fuzzy ok', numpy.__version__, m.version('pyfume'))"
```

The second check must print `numpy 1.24.4` and pyFUME `0.3.4`. Newer pyFUME versions crash on
first-order models.

> dlib comes from the `dlib-bin` wheel (prebuilt), so no CMake or Visual Studio is needed.

---

## 4. Run the notebooks — in this order

Either open them in VS Code / Jupyter and pick the kernel shown in the table, then *Run All*, or use
the command line. Each notebook uses files written by the previous ones, so **the order
matters**.

| # | notebook | kernel | what it does | time (approx.) |
|---|---|---|---|---|
| 1 | `z_load_split_fer2013.ipynb` | ass2 | FER CSV → `data/fer2013_{train,validation,test}.csv` | 1 min |
| 2 | `z_load_kdef_student.ipynb` | ass2 | KDEF → `data/kdef_preprocessed.pkl` | 1 min |
| 3 | `00_data_preparation.ipynb` | ass2 | image arrays, landmarks, KDEF variants, data exploration | 10–15 min |
| 4 | `01_fer_baseline.ipynb` | ass2 | **Task 1** | longest (SVM grids; roughly 30–60 min with the GPU) |
| 5 | `02_kdef_study.ipynb` | ass2 | **Task 2** | 10–20 min |
| 6 | `03_cross_dataset.ipynb` | ass2 | **Task 3** | 30–60 min |
| 7 | `04_fuzzy_kdef.ipynb` | **ass2-fuzzy** | **Task 4** | 5–15 min |
| 8 | `05_creative_challenge.ipynb` | ass2 | **Task 5** → `models/final.pkl` | longest (roughly 30–60 min with the GPU) |

Command line (with the main environment activated: `..\.venv\Scripts\activate` or `conda activate ass2`):

```powershell
$nbs = "z_load_split_fer2013","z_load_kdef_student","00_data_preparation","01_fer_baseline","02_kdef_study","03_cross_dataset"
foreach ($n in $nbs) { jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$n.ipynb" }
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=ass2-fuzzy 04_fuzzy_kdef.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 05_creative_challenge.ipynb
```

### Optional: quick check first (recommended)

With `SMOKE=1` every notebook runs on small subsets with reduced grids, so the whole chain takes
minutes. Use it to make sure everything works on your machine before the long run:

```powershell
$env:SMOKE = "1"     # quick check
# ... run the notebooks as above ...
Remove-Item Env:SMOKE  # back to the FULL run (default) before the real run
```

**The results we deliver must come from the full run (SMOKE not set).** Re-run all notebooks
from 01 on without SMOKE afterwards (00 and the kit notebooks do not depend on SMOKE).

---

## 5. Notes

* The SVM kernels run on the **GPU** (CUDA). Everything else (HOG, dlib landmarks, LR, RF)
  runs on the CPU with every core (`n_jobs=-1`). Keep the laptop plugged in and disable sleep for
  notebooks 01 and 05.
* Features are cached in `data/cache/`. If a notebook crashes halfway, just re-run it: the cached
  features are reused.
* Messages like `KeyError: '...joblib_memmapping_folder...'` in the console come from joblib on
  Windows and are **harmless**.
* If you change `src/features.py`, delete `data/cache/` and re-run from `00`.

---

## 6. Test the demo (Task 6)

```powershell
python demo_emotion.py            # uses models/final.pkl, camera 0
python demo_emotion.py --camera 1 # other camera
```

Keys: `q` quit, `s` smoothing on/off, `+`/`-` crop margin.

**Please write down** what you observe: which emotions work and which don't, FPS, flickering,
lighting problems, which margin/smoothing worked best. This goes into
`report/07_task6_realtime.md`.

---

## 7. Commit the results back (models with Git LFS)

The trained models are larger than GitHub's 100 MB file limit (e.g. `final.pkl` ≈ 125 MB), so
they are stored with **Git LFS**. The repository already has a `.gitattributes` that sends every
`group_2_emotion_recognition/models/*.pkl` to LFS; Git for Windows includes `git lfs`.

```powershell
git lfs install                     # once per machine
git pull                            # make sure you have the latest code and .gitattributes

cd group_2_emotion_recognition
git add *.ipynb results figures models
git commit -m "Full run 2: results, figures and trained models"
git push origin final

git lfs ls-files                    # every models/*.pkl must be listed here
```

The push uploads the models to LFS first (a few hundred MB: be patient). If it fails with
`HTTP 408` or a timeout, run `git config http.postBuffer 524288000` and push again.

Whoever clones or pulls afterwards runs `git lfs install` once; `git pull` then downloads the
real model files instead of small pointer files.

**Never commit:** `data/`, `.venv*`, `__pycache__`.

After the push, Gabriele fills the report numbers from `results/*.csv` / `results/*.tex`.
