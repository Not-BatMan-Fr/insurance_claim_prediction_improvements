# Insurance Claim Prediction Pipeline

## 📌 Project Overview
This project implements a machine learning pipeline to predict whether an insurance policyholder will file a claim. It compares the performance of three classification algorithms: **Logistic Regression**, **Decision Tree**, and **Random Forest**.

The codebase has been refactored from a monolithic script into a modular architecture following **SOLID principles** and to prevent data leakage.

## 🚀 Key Features
- **Modular Architecture**: Separate components for data loading, preprocessing, modeling, and visualization.
- **Leakage Prevention**: Strict separation of training and testing data before oversampling or scaling to avoid information leakage.
- **Imbalance Handling**: Uses `RandomOverSampler` (SMOTE-like approach) strictly on the training set to address class imbalance.
- **Visualizations**: Built-in `MatplotlibVisualizer` produces ROC curves, confusion matrices, feature importances, class distributions, and a performance summary saved to `visualizations/`.
- **Automated Scripts & Tests**: Includes `run.sh` for quick setup and execution and `run_tests.sh` for running unit tests with coverage reporting.
- **Extensible**: Easily add new models (e.g., XGBoost, SVM) or custom visualizers without modifying core logic.

## 📂 Project Structure
```text
insurance_claim_prediction/
├── data/                  # Sample data stored here
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Centralized configuration (paths, params)
│   ├── interfaces.py      # Abstract Base Classes (Contracts)
│   ├── data_loader.py     # Data ingestion logic
│   ├── preprocessor.py    # Cleaning, encoding, and feature engineering
│   ├── models.py          # Model adapters and evaluation metrics
│   ├── pipeline.py        # Orchestrator handling the workflow
│   ├── visualizer.py      # Matplotlib-based visualization utilities
│   └── main.py            # Entry point / CLI
├── tests/                 # Unit tests (see `run_tests.sh`)
├── requirements.txt       # Python dependencies
├── run.sh                 # Setup & execution script
└── run_tests.sh           # Run tests with coverage and generate HTML report
```

**Repo:**: `insurance_claim_prediction` — a compact, modular pipeline to preprocess insurance data, train sklearn models, evaluate them, and (optionally) generate visualizations saved to `visualizations/`.

**Requirements**
- **Python:** 3.8+ recommended.
- **Dependencies:** listed in `requirements.txt`.

**Run Training & Evaluation**
- The main entrypoint runs preprocessing, trains three models, evaluates them, and prints a summary.

Recommended (Automated) invocation:
```bash
./run.sh
```

You should see console output steps (loading, preprocessing, training) and a final DataFrame with metrics for each model as shown below.
```
1. Loading Data...
2. Preprocessing...
3. Splitting Data...
4. Handling Imbalance (Training Set Only)...
5. Scaling Features...
6. Training & Evaluating: Logistic Regression...
6. Training & Evaluating: Decision Tree...
6. Training & Evaluating: Random Forest...

==================================================
FINAL MODEL PERFORMANCE
==================================================
                     Accuracy  Precision  Recall  F1_Score   ROC_AUC    FNR
Model                                                                      
Logistic Regression  0.572745   0.090594   0.628  0.158346  0.598484  0.372
Decision Tree        0.879427   0.079848   0.084  0.081871  0.508907  0.916
Random Forest        0.913986   0.078431   0.032  0.045455  0.503146  0.968
==================================================
```

**Generating Visualizations**
- By default `src/main.py` will create visualizations and save them into `visualizations/`.
- To disable visualizations: `python src/main.py --no-viz`
- To customize output directory: `python src/main.py --viz-dir my_plots/`
- Sample plots are already included in `visualizations/` (ROC curves, confusion matrices, feature importance, performance comparison, class distribution).

<img src = "visualizations/class_distribution.png">
<img src = "visualizations/confusion_matrices.png">
<img src = "visualizations/feature_importance_decision_tree.png">
<img src = "visualizations/feature_importance_random_forest.png">
<img src = "visualizations/performance_comparison.png">
<img src = "visualizations/roc_curves.png">


**Run Tests**
- Quick: `./run_tests.sh` (runs `pytest` with coverage and generates `htmlcov/` report)
- Alternatively: `pytest -q` or `coverage run -m pytest` for custom use.

**Project Structure (key files)**
- `src/main.py`: orchestrates configuration, pipeline and model registration (CLI supports `--no-viz` and `--viz-dir`).
- `src/pipeline.py`: data split, imbalance handling, scaling, training loop, and optional visualization integration.
- `src/preprocessor.py`: dataset cleaning and encoding, returns `X, y`.
- `src/models.py`: model adapters and metric evaluation helpers.
- `src/visualizer.py`: `MatplotlibVisualizer` for saving performance plots to disk.
- `data/`: include `train_data.csv` (and sample `test_data.csv`, `truncated_train_data.csv` for faster iterations).

**Troubleshooting**
- If you see `FileNotFoundError: Dataset not found at: ...`, ensure the file exists at the configured `data_path`.
- If you see `Target column 'is_claim' missing`, verify your CSV contains the `is_claim` column (case-sensitive).
- For fast iteration, use `data/truncated_train_data.csv`.
