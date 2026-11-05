# ML_Course — CS308 Machine Learning lab notebooks

Each lab lives on its **own branch**. `main` only holds this overview and the shared `.gitignore`.

| Branch | Lab | Topics |
|---|---|---|
| [`svm_lab4_2324`](../../tree/svm_lab4_2324) | Lab 4 — Support Vector Machine | SGD hinge-loss SVM, kernel SVM (RBF + 2 extra kernels) on XOR, multiclass SVM |
| [`CS308_lab_23`](../../tree/CS308_lab_23) | Labs 2 & 3 — Linear models | perceptron, average perceptron, Pegasos, logistic regression, Titanic, PCA on MNIST, multiclass linear classifiers |
| [`CS308_Lab5_6`](../../tree/CS308_Lab5_6) | Labs 5 & 6 — Neural networks | NumPy NN from scratch, CuPy (GPU) version, PyTorch on GPU, hyper-parameter tuning, 6-fold CV, confusion matrix / AUC |

## What is on each branch

```
<lab folder>/
  <original>.ipynb        # PRIMARY — the original notebook, untouched
  <lab>_done.ipynb        # DONE — solved, executed top-to-bottom, outputs saved
  *.npz / *.npy / *.pth   # trained models produced by the DONE notebook
  toy_data.tsv            # (CS308_lab_23 only) data shipped with the lab
```

- Markers inside the DONE notebooks: `✅ Даалгавар …` = a task solution, `🔧 Засвар:` = a fix of a bug in the provided code (with the reason).
- Written explanations and conclusions are in Mongolian, matching the original notebooks.
- Every saved model has a "load & verify" cell in its notebook that reloads the file and reproduces the reported metric.

## Running

- Python 3.12; packages: `numpy pandas scipy scikit-learn seaborn matplotlib graphviz torch torchvision cupy nbformat nbclient ipykernel`.
- `CS308_Lab5_6` uses the GPU (CuPy + PyTorch). It was run on an NVIDIA RTX 5070 Ti (CUDA 12.8 wheels: torch `cu128`, CuPy 14). The other two labs run on CPU.
- Datasets (MNIST, Titanic) are downloaded automatically into `data/` on first run and are not committed.
- Run the DONE notebook from inside its own lab folder so relative paths resolve.
- The system `dot` binary (Graphviz) is not required; the diagram cells have a fallback.

## Verification

The DONE notebooks were completed by a developer agent and independently accepted by a separate QA reviewer (fresh re-execution with zero errors, protected-hash check that the primaries are byte-identical to the originals, independent numerical/scikit-learn cross-checks, and model reload checks).
