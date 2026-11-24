# DNA Sequence Classification: 12 ML / DL / QML / Hybrid Models

A small, fast pipeline that goes from DNA dataset to preprocessing, k-mer and encoding features, then PCA, then 12 models, and ends with a metrics table and plots. It uses proper **train / validation / test** splits.

## Dataset
This project uses the Kaggle [DNA Sequence Dataset](https://www.kaggle.com/datasets/nageshsingh/dna-sequence-dataset) (`human.txt`, `chimpanzee.txt`, `dog.txt`, `example_dna.fa`). The first run downloads it into `data/raw/` with the `kaggle` CLI (token in `~/.kaggle/`).

* `--task family` (default): 7 gene families (GPCR, Tyr kinase, Tyr phosphatase, Synthetase, Synthase, Ion channel, Transcription factor).
* `--task species`: human vs chimpanzee vs dog.

Cleaning works like this. Each sequence is uppercased, and any non-ACGT symbol becomes `N`. Sequences shorter than 30 bases or more than 5% `N` are dropped. Exact duplicates are removed, and sequences that appear with conflicting labels are dropped, so no sequence appears in two splits.
Split: stratified **70 / 15 / 15**, seed 42, saved to `data/splits/<task>_{train,val,test}.csv`.
The validation set is used for hyper-parameters and early stopping. The test set is scored once, for the final table.

## Models
| # | Model | Notes |
|---|---|---|
| 1 | K-mer + KNN | TF-IDF of k=3..6 k-mers, cosine KNN, k chosen on val |
| 2 | K-mer + SVM | LinearSVC, C chosen on val |
| 3 | Features + XGBoost | k-mer TF-IDF + GC%, skews, CpG o/e, length |
| 4 | Encoding + 1D-CNN | one-hot (4×1024) |
| 5 | Encoding + CNN-LSTM | conv front-end, then LSTM |
| 6 | PCA + QSVM | PCA(8), 8-qubit ZZ feature map, fidelity kernel, SVC |
| 7 | PCA + QNN | 8-qubit variational circuit with data re-uploading |
| 8 | CNN + QNN | CNN features → 8 angles → QNN, trained end-to-end |
| 9 | BiLSTM + QNN | BiLSTM → 8 angles → QNN, trained end-to-end |
| 10 | QAOA + QNN | QAOA (12 qubits, p=2) solves a QUBO that picks 6 of 12 informative 3-mers, then a 6-qubit QNN |
| 11 | VQE + QNN | VQE finds the ground state of an Ising Hamiltonian built from the features. The QNN starts from that state |
| 12 | QML + CNN | quanvolution: a 4-qubit circuit on each 8-base window, then a 1D-CNN |

The quantum circuits run on a small exact state-vector simulator written in PyTorch (`dna/quantum.py`). It is differentiable and needs no PennyLane or Qiskit, which don't support Python 3.14 well. All of this is classical simulation, not quantum hardware.

## Commands
Install (once):
```bash
pip install -r requirements.txt
```

Run the unit tests (data, k-mers, simulator correctness, QAOA, VQE):
```bash
python3 -m unittest discover tests -v
```

**One command: train, validate and test all 12 models on both tasks, then write the Markdown report with graphics:**
```bash
python3 run.py
```
The report is written to `results/REPORT.md`.

**Resume after an interruption** (Ctrl+C, crash, or a frozen run). Finished models are cached in `results/cache/`, and `--resume` reuses them when the settings and data are unchanged. Only unfinished models are retrained:
```bash
python3 run.py --resume
```
Everything else is recomputed, so changing settings or the seed starts fresh automatically. `REPORT.md` is rewritten after each task, so a partial report is always available.

Only one task:
```bash
python3 run.py --task family
```

Only some models:
```bash
python3 run.py --models 1,2,6,10
```

How much it prints:
* default: timestamped progress for every step. That covers each feature-building stage; every hyper-parameter tried, with train/val accuracy; XGBoost val loss every 10 rounds; one line per epoch with train loss/acc, val loss/acc, learning rate, time and early stopping; QSVM kernel stats; QAOA/VQE energy every 25 steps; and each model's results and confusion matrix.
* `-v`: also progress inside each epoch (batch, running loss/acc, sequences/s, ETA):
  ```bash
  python3 run.py -v
  ```
* `-q`: quiet, one result line per model.

Longer training (more epochs, QSVM on the full train set):
```bash
python3 run.py --full
```

Other flags: `--seed 42`, `--device auto|cpu|mps|cuda`, `--out results`.

## Outputs
* **`results/REPORT.md`** is the full report. It opens in VS Code or GitHub preview and has:
  dataset and split sizes, class-distribution chart, results table, accuracy/F1 bar chart,
  validation-vs-test scatter, training-time chart, confusion matrices (a grid and one per model, with counts),
  per-class F1 heatmap, ROC curves, plus details for each model (chosen hyper-parameters, QAOA/VQE diagnostics).
* `results/metrics_<task>.csv` and `results/summary_<task>.md` hold the raw numbers.
* `results/plots/` holds every figure as a PNG.

If a model crashes, the run keeps going and the report lists the error.

## Notes
* On macOS, xgboost and torch each ship their own OpenMP library (libomp), and loading both in one process crashes or hangs. XGBoost therefore trains in a separate spawned process (`dna/classical.py::_xgb_worker`). Don't import xgboost anywhere else.
* Chimpanzee and human genes are close homologs. Exact duplicates are removed, but near-identical sequences can still land in different splits, which makes this an easy benchmark (k-mer KNN reaches about 98%).
