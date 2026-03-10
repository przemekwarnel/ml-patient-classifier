# Patient Disease Classification

## Project Overview

This project builds a machine learning pipeline to classify patients as healthy or diseased based on clinical features.  
The goal is to develop a reliable model for medical screening while minimizing the risk of missing diseased patients (false negatives).

The project demonstrates an end-to-end applied machine learning workflow, including:

- data preprocessing with a leakage-safe sklearn pipeline
- hyperparameter tuning using cross-validation
- model comparison across multiple algorithms
- threshold tuning for medical screening scenarios
- reproducible evaluation with generated reports

The final model is selected based on cross-validated performance and evaluated under different decision thresholds to reflect real-world screening constraints.

## Dataset

The dataset contains **918 patient observations** with **11 clinical features** used to predict the presence of disease.

Features include measurements such as:

- age
- chest pain type
- resting blood pressure
- cholesterol
- maximum heart rate
- ST segment characteristics

The target variable indicates whether the patient has the disease.

The dataset is included in the repository under:

[`data/raw/heart.csv`](data/raw/heart.csv)

## Problem Formulation

This is a **binary classification problem**:

```
X → patient clinical features
y → disease presence (0 = healthy, 1 = diseased)
```

The primary evaluation metric during model selection is **ROC-AUC**, which measures the model’s ability to rank diseased patients above healthy ones.

However, in medical screening the cost of false negatives is high.  
Therefore, the project also evaluates models under **recall constraints**, ensuring that the majority of diseased patients are detected.

Threshold tuning is used to adjust the classifier for different screening scenarios.

## Project Structure

```
ml-patient-classifier
│
├── configs
│ └── base.yaml # training configuration
│
├── data
│ ├── raw
│ │ └── heart.csv # dataset
│ └── sample_patient.json # example input for inference
│
├── reports # generated evaluation artifacts
│ ├── model_comparison.*
│ ├── threshold_comparison.*
│ ├── roc_curve.png
│ ├── confusion_matrix.png
│ ├── metrics.json
│ └── eval_metrics.json
│
├── src
│ └── ml_patient_classifier
│ ├── config.py
│ ├── data.py
│ ├── preprocessing.py
│ ├── modeling.py
│ ├── predict.py
│ ├── tuning.py
│ ├── train.py
│ ├── evaluate.py
│ ├── compare_models.py
│ └── threshold_analysis.py
│
├── notebooks
│ └── analysis.ipynb # exploratory analysis
│
├── models # trained models
│
├── tests
│ └── test_data.py # basic data loading sanity test
│
├── .gitignore
├── pyproject.toml
└── README.md
```

## Installation

Clone the repository and create a virtual environment:

```zsh
git clone <repo-url>
cd ml-patient-classifier

python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```zsh
pip install -e ".[dev]"
```

## How to Run

### Train the model

```zsh
python -m ml_patient_classifier.train --config configs/base.yaml
```

This will:

- train the model using cross-validation
- save the trained pipeline to models/
- store training metrics

---

### Evaluate the model 

```zsh
python -m ml_patient_classifier.evaluate --config configs/base.yaml
```

This generates:

- ROC curve
- confusion matrix
- evaluation metrics

---

### Compare candidate models 

```zsh
python -m ml_patient_classifier.compare_models --config configs/base.yaml
```

This compares:

- Logistic Regression
- Support Vector Machine
- Random forest

and saves results to reports/model_comparison.*.

---

### Run threshold analysis 

```zsh
python -m ml_patient_classifier.threshold_analysis --config configs/base.yaml
```

This evaluates screening scenarios with different recall constraints and saves the results to reports/threshold_comparison.*.

---

### Local inference

```zsh
python -m ml_patient_classifier.predict.py --input data/sample_patient.json --model models/pipeline.joblib --threshold 0.5
```

This runs prediction for a single patient using the saved trained pipeline. 

Example output

```json
{
  "prediction": 0,
  "label": "healthy",
  "probability_positive_class": 0.0972348894346362,
  "threshold": 0.5
}
```

## Modeling Approach

The project follows a structured machine learning workflow designed to produce reproducible and leakage-safe experiments.

### Data preprocessing

All preprocessing steps are implemented inside an **scikit-learn Pipeline**, ensuring that transformations are fitted only on training data during cross-validation.

The preprocessing pipeline includes:

- missing value imputation
- numerical feature scaling
- categorical feature encoding (when applicable)

Using a pipeline prevents **data leakage** between training and evaluation stages.

---

### Model training

Model training is implemented as a CLI command:

```zsh
python -m ml_patient_classifier.train --config configs/base.yaml
```

Training uses:

- Stratified train/test split
- Stratified K-fold cross-validation
- GridSearchCV for hyperparameter tuning

The primary model selection metric is ROC-AUC, which measures the model's ability to rank diseased patients above healthy ones.

---

### Candidate models

Three candidate algorithms were evaluated:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest

Each model was trained using the same preprocessing pipeline and evaluated under identical cross-validation settings to ensure fair comparison.

---

### Reproducible configuration

All training parameters are defined in: 

[`configs/base.yaml`](configs/base.yaml)

This includes:
- model type
- cross-validation folds
- evaluation metric
- random seed

## Model Comparison

Three candidate models were evaluated using identical preprocessing and cross-validation settings.

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy | Precision | Recall | F1 |
|------|-----------|-------------|---------|----------|-------|----|
| Random Forest | 0.934 | 0.933 | 0.902 | 0.896 | 0.931 | 0.913 |
| Logistic Regression | 0.926 | 0.932 | **0.913** | **0.913** | 0.931 | **0.922** |
| SVM (RBF) | 0.924 | **0.949** | 0.891 | 0.887 | 0.922 | 0.904 |

### Model selection

Although Random Forest achieved the highest cross-validated ROC-AUC, **Logistic Regression** was selected as the final model for the following reasons:

- comparable ROC-AUC performance
- higher precision on the test set
- lower model complexity
- better interpretability of coefficients

In medical applications, simpler and more interpretable models are often preferred when predictive performance is similar.

Detailed comparison results are available in:

- [`reports/model_comparison.csv`](reports/model_comparison.csv)
- [`reports/model_comparison.md`](reports/model_comparison.md)

## Threshold Analysis for Clinical Screening

In medical applications, the cost of **false negatives** (missing a diseased patient) is often much higher than the cost of false positives.

Therefore, the project evaluates different decision thresholds under recall constraints to simulate screening scenarios.

| Setting | Threshold | Recall | Precision | False Negatives | False Positives |
|--------|-----------|--------|-----------|-----------------|----------------|
| Default | 0.50 | 0.931 | 0.913 | 7 | 9 |
| Screening (recall ≥ 0.90) | 0.528 | 0.902 | **0.920** | 10 | **8** |
| Screening (recall ≥ 0.95) | 0.258 | **0.951** | 0.822 | **5** | 21 |

### Interpretation

Lowering the decision threshold increases recall, allowing the model to detect more diseased patients.

For example:

- the default model misses **7 diseased patients**
- enforcing **recall ≥ 0.95** reduces missed cases to **5**, at the cost of additional false positives

This trade-off reflects typical medical screening workflows, where a first-stage model prioritizes sensitivity and is followed by more precise diagnostic tests.

Detailed results are available in:

- [`reports/threshold_comparison.csv`](reports/threshold_comparison.csv)  
- [`reports/threshold_comparison.md`](reports/threshold_comparison.md)

## Results

The final selected model is **Logistic Regression with L2 regularization**.

Key performance metrics on the test set:

- **ROC-AUC:** 0.93
- **Accuracy:** 0.91
- **Precision:** 0.91
- **Recall:** 0.93
- **F1-score:** 0.92

Under stricter screening conditions (recall ≥ 0.95), the model can reduce missed disease cases from **7 to 5**, at the cost of increased false positives.

This demonstrates how adjusting the decision threshold allows the model to be adapted for different clinical scenarios.

## Key Takeaways

This project demonstrates an applied machine learning workflow including:

- leakage-safe preprocessing using sklearn pipelines
- reproducible experiments via configuration files
- hyperparameter tuning with cross-validation
- model comparison across multiple algorithms
- threshold analysis for domain-specific constraints
- reproducible evaluation artifacts

The project emphasizes **interpretability, reproducibility, and practical decision-making**, which are critical in real-world ML systems.

## Future Improvements

Possible extensions of the project include:

- probability calibration for improved threshold reliability
- feature importance analysis and model interpretability
- deployment of the trained model as an API service
- additional validation using external datasets