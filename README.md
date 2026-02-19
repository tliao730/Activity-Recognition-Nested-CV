# Activity Recognition using Logistic Regression and Nested CV

## 📌 Project Overview
This repository implements a robust machine learning pipeline for **Activity Recognition**. The project focuses on ensuring model generalizability by utilizing **Nested Cross-Validation** and automated feature selection via **RFECV**.

### Key Features:
- **Feature Engineering**: Custom extraction logic for motion sensor data.
- **Nested Cross-Validation**: Implemented to provide an unbiased evaluation of the model's performance.
- **Feature Selection**: Recursive Feature Elimination with Cross-Validation (RFECV) to identify the optimal feature set (p*).
- **Scalable Structure**: Modularized code for data loading, feature extraction, and evaluation.

## 📂 Project Structure
```text
Activity_Recognition/
├── src/
│   ├── DataLoader.py           # Data ingestion logic
│   ├── feature_extraction.py    # Signal processing and feature engineering
│   ├── model_evaluation.py     # Nested CV and RFECV implementation
│   └── __init__.py
├── results/                    # CSV outputs and visualization
├── main.py                     # Entry point for running experiments
├── requirements.txt            # List of dependencies
└── README.md