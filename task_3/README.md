# Olist Late-Delivery Inference Pipeline

This project contains the inference pipeline for the Olist late-delivery prediction task.

The trained model, fitted imputer, and feature list are loaded from saved artifacts. Training remains in the notebooks.

## Project structure

```text
task_3/
├── app/
├── config/
│   └── params.yaml
├── data/
│   ├── artifacts/
│   ├── processed/
│   │   ├── ml_train_features.csv
│   │   ├── ml_val_features.csv
│   │   └── ml_test_features.csv
│   └── raw/
│       ├── ML_TABLE.csv
│       ├── ML_TABLE_LABELED.csv
│       ├── ml_train.csv
│       ├── ml_val.csv
│       └── ml_test.csv
├── feature_list.txt
├── models/
│   ├── trained_model.pkl
│   └── transformers/
│       └── imputer.pkl
├── notebooks/
├── requirements/
│   ├── runtime.txt
│   └── development.txt
├── src/
│   ├── config.py
│   ├── data_access.py
│   ├── features.py
│   ├── pipeline.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── validation.py
│   └── __init__.py
├── unit_tests/
│   ├── test_notebook_parity.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   └── data/
│       └── notebook_predictions.csv
└── README.md
```

## Requirements

- Python 3.11
- PostgreSQL database with Olist tables created from Task 1
- Trained model artifact
- Fitted imputer artifact
- Feature list file

Install dependencies:

```bash
pip install -r requirements/development.txt
pip install -r requirements/runtime.txt
```

## Database setup

Create the database schema using the SQL from Task 1, then import the CSV files into the PostgreSQL tables.

Update the database connection string in `config/params.yaml`.

## Model and feature artifacts

The inference pipeline expects:

- `models/trained_model.pkl`
- `models/transformers/imputer.pkl`
- `feature_list.txt`

These must be present before running predictions.

## Running the notebook parity check

From the project root:

```bash
pytest .\unit_tests\test_notebook_parity.py
```

This check compares the Python pipeline output against the saved notebook predictions for known orders.

To run all tests:

```bash
pytest unit_tests
```

## What this project does

This project does not train the model again at inference time. It loads the saved fitted objects and predicts late vs on-time delivery for a new order.

The pipeline expects a row-like dict order input and produces:

- `prediction`
- `probability`