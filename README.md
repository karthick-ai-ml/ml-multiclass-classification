# ml-multiclass-classification

Machine Learning research focused on multi-class classification, implementing multiple models with evaluation metrics and comparison, presented through an interactive Streamlit application for visualization and deployment.

** Live Demo:** [https://ml-obesity-risk-predictor.streamlit.app/](https://ml-obesity-risk-predictor.streamlit.app/)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models Used](#models-used)
  - [Model Comparison Table](#model-comparison-table)
  - [Model Observations](#model-observations)
- [Repository Structure](#repository-structure)
- [Models Details](#models-details)
- [Getting Started](#getting-started)

---

## Problem Statement

The objective of this project is to predict **obesity risk levels** in individuals using a multi-class classification approach. Given a set of 16 features capturing demographic information, eating habits, physical activity, and lifestyle factors, the task is to classify each individual into one of **7 obesity categories**: `Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, `Overweight_Level_II`, `Obesity_Type_I`, `Obesity_Type_II`, and `Obesity_Type_III`.

Six different machine learning models - **Logistic Regression**, **Decision Tree**, **K-Nearest Neighbors (kNN)**, **Naive Bayes (Gaussian)**, **Random Forest (Ensemble)**, and **XGBoost (Ensemble)** - are trained, evaluated, and compared on standard classification metrics (Accuracy, AUC, Precision, Recall, F1-Score, and MCC) to identify the best-performing model for this multi-class problem.

---

## Dataset Description

The dataset used in this project is provided in the `dataset/` directory.
License and attribution details are available in [dataset/DATASET.md](dataset/DATASET.md).

- **Dataset Name:** Multi-Class Prediction of Obesity Risk
- **Source:** [Kaggle Playground Series - Season 4, Episode 2](https://www.kaggle.com/competitions/playground-series-s4e2)
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

| Property | Value |
|---|---|
| Training Samples | 20,758 |
| Test Samples | 13,841 |
| Features | 16 input features + 1 target variable |
| Target Classes | 7 obesity categories |
| Target Column | `NObeyesdad` (Obesity Level) |

### Features

| Category | Feature | Description |
|---|---|---|
| Demographic | Gender | Male / Female |
| Demographic | Age | Age in years |
| Demographic | Height | Height in meters |
| Demographic | Weight | Weight in kg |
| Family History | family_history_with_overweight | Family history of overweight (yes/no) |
| Eating Habits | FAVC | Frequent consumption of high caloric food (yes/no) |
| Eating Habits | FCVC | Frequency of vegetable consumption (0–3) |
| Eating Habits | NCP | Number of main meals |
| Eating Habits | CAEC | Food between meals (Never/Sometimes/Frequently/Always) |
| Eating Habits | CH2O | Daily water consumption (0–3) |
| Lifestyle | SCC | Calories consumption monitoring (yes/no) |
| Lifestyle | FAF | Physical activity frequency (0–3) |
| Lifestyle | TUE | Time using technology devices (0–2) |
| Lifestyle | CALC | Alcohol consumption (Never/Sometimes/Frequently/Always) |
| Lifestyle | SMOKE | Smoking habit (yes/no) |
| Lifestyle | MTRANS | Transportation used (Walking/Bike/Public_Transportation/Automobile) |

### Preprocessing

- **Categorical Encoding:** One-Hot Encoding applied to all categorical features (Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS)
- **Target Encoding:** Label Encoding for the 7 target classes
- **Feature Scaling:** Standard Scaling applied to all numerical features
- **Final Feature Count:** 22 features after preprocessing
- **Train/Validation Split:** 15,879 training samples / 2,803 validation samples

---

## Models Used

Six machine learning models were trained and evaluated on the preprocessed dataset. The table below compares their performance on the **validation set** using standard classification metrics.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8644 | 0.9818 | 0.8638 | 0.8644 | 0.8638 | 0.8411 |
| Decision Tree | 0.8776 | 0.9749 | 0.8787 | 0.8776 | 0.8779 | 0.8566 |
| kNN | 0.7752 | 0.9501 | 0.7736 | 0.7752 | 0.7731 | 0.7366 |
| Naive Bayes | 0.6072 | 0.9185 | 0.6190 | 0.6072 | 0.5762 | 0.5485 |
| Random Forest (Ensemble) | 0.8930 | 0.9884 | 0.8937 | 0.8930 | 0.8926 | 0.8747 |
| XGBoost (Ensemble) | 0.9012 | 0.9900 | 0.9015 | 0.9012 | 0.9012 | 0.8841 |

> **Best Model:** XGBoost (Ensemble) with the highest Accuracy (0.9012), F1-Score (0.9012), and MCC (0.8841) on the validation set.

### Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | **Performance: Good** - Achieves solid validation accuracy (86.44%) and F1 (0.8638) with minimal overfitting (gap: 0.28%). Fast training and interpretable coefficients make it an excellent baseline. Limited by linear decision boundaries which may underfit complex non-linear patterns in the data. Best suited when interpretability is a priority. |
| Decision Tree | **Performance: Good** - Validation accuracy of 87.76% and F1 of 0.8779 with a small overfit gap (2.49%). Highly interpretable and handles non-linear relationships well. However, prone to instability with small data changes. Good choice when model visualization and interpretability are needed. |
| kNN | **Performance: Moderate** - Validation accuracy of 77.52% and F1 of 0.7731 with **high overfitting** (gap: 22.48%, training accuracy 100%). The model memorizes training data rather than learning generalizable patterns. Sensitive to feature scaling and noise. Only suitable for small datasets where similar samples naturally cluster together. |
| Naive Bayes | **Performance: Needs Improvement** - Lowest validation accuracy (60.72%) and F1 (0.5762) among all models, but well generalized (gap: 0.87%). The strong feature independence assumption fails to capture correlations present in the obesity dataset. Very fast and useful for quick baselines or real-time predictions, but not suitable as the primary model for this problem. |
| Random Forest (Ensemble) | **Performance: Good** - Strong validation accuracy (89.30%) and F1 (0.8926) with mild overfitting (gap: 7.43%). Effectively reduces variance compared to a single Decision Tree through bagging. Provides useful feature importance rankings. Slightly slower than single models but delivers better generalization. Ideal for general-purpose use when accuracy matters more than speed. |
| XGBoost (Ensemble) | **Performance: Excellent** - Best overall model with validation accuracy of 90.12%, F1 of 0.9012, and highest MCC (0.8841). Mild overfitting (gap: 7.23%) is well-controlled through built-in regularization. Handles missing values naturally and delivers state-of-the-art performance. Requires more complex tuning and longer training time. Best choice when maximum predictive accuracy is the primary goal. |

---

## Repository Structure

```
ml-multiclass-classification/
├── README.md                          # Project documentation (this file)
├── LICENSE                            # License file
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
├── install_prerequisites.sh           # Script to install prerequisites
│
├── dataset/                           # Dataset files
│   ├── DATASET.md                     # Dataset license and attribution
│   ├── dataset.json                   # Dataset metadata configuration
│   ├── train.csv                      # Training data (20,758 samples)
│   ├── test.csv                       # Test data (13,841 samples)
│   └── sample_submission.csv          # Kaggle submission format
│
├── experiments/                       # Jupyter Notebook experiments
│   ├── stage1-data_exploration_and_preprocessing.ipynb   # EDA & preprocessing
│   └── stage2-model_training_and_evaluation.ipynb        # Model training & evaluation
│
├── models/                            # Trained model artifacts (.joblib)
│   ├── logistic_regression.joblib
│   ├── decision_tree_classifier.joblib
│   ├── k-nearest_neighbor_classifier.joblib
│   ├── naive_bayes_classifier_gaussian.joblib
│   ├── random_forest_ensemble.joblib
│   ├── xgboost_ensemble.joblib
│   └── kaggle_obesity_prediction_model_trained.json      # Training metadata
│
└── artifacts/                         # Generated outputs
    ├── data/                          # Processed data & metrics
    │   ├── kaggle_obesity_prediction_model_evaluation_results.json
    │   ├── kaggle_obesity_prediction_model_observations.csv
    │   ├── kaggle_obesity_prediction_preprocessing_analysis.json
    │   ├── kaggle_obesity_prediction_training_metrics_comparison.csv
    │   ├── kaggle_obesity_prediction_validation_metrics_comparison.csv
    │   ├── kaggle_obesity_prediction_training_eda_analysis.json
    │   ├── kaggle_obesity_prediction_visualization_analysis.json
    │   ├── kaggle_obesity_prediction_sample_head_10.csv
    │   ├── kaggle_obesity_prediction_sample_tail_10.csv
    │   ├── kaggle_obesity_prediction_train_preprocessed.csv
    │   ├── kaggle_obesity_prediction_validation_preprocessed.csv
    │   └── kaggle_obesity_prediction_test.csv
    ├── images/                        # UI images
    │   ├── Doctor.png
    │   └── Robotic.png
    └── reports/                       # Interactive HTML visualizations
        ├── kaggle_obesity_prediction_dataset_overview.html
        ├── kaggle_obesity_prediction_feature_summary.html
        ├── kaggle_obesity_prediction_target_distribution.html
        ├── kaggle_obesity_prediction_numerical_distributions.html
        ├── kaggle_obesity_prediction_categorical_distributions.html
        ├── kaggle_obesity_prediction_correlation_matrix.html
        ├── kaggle_obesity_prediction_training_metrics_heatmap.html
        ├── kaggle_obesity_prediction_validation_metrics_heatmap.html
        ├── kaggle_obesity_prediction_model_comparison_validation_metrics.html
        ├── kaggle_obesity_prediction_confusion_matrices.html
        ├── kaggle_obesity_prediction_auc_roc_curves.html
        └── kaggle_obesity_prediction_model_observations.html
```

---

## Models Details

### 1. Logistic Regression
- **Type:** Linear model (One-vs-Rest for multi-class)
- **File:** `models/logistic_regression.joblib`
- **Key Characteristics:** Fast training, interpretable coefficients, linear decision boundaries
- **Validation Accuracy:** 86.44%

### 2. Decision Tree Classifier
- **Type:** Tree-based model
- **File:** `models/decision_tree_classifier.joblib`
- **Key Characteristics:** Highly interpretable, captures non-linear relationships, single tree structure
- **Validation Accuracy:** 87.76%

### 3. K-Nearest Neighbor (kNN) Classifier
- **Type:** Instance-based / Lazy learner
- **File:** `models/k-nearest_neighbor_classifier.joblib`
- **Key Characteristics:** No explicit training phase, distance-based classification, sensitive to feature scaling
- **Validation Accuracy:** 77.52%

### 4. Naive Bayes Classifier (Gaussian)
- **Type:** Probabilistic model
- **File:** `models/naive_bayes_classifier_gaussian.joblib`
- **Key Characteristics:** Very fast, assumes feature independence, probabilistic output
- **Validation Accuracy:** 60.72%

### 5. Random Forest (Ensemble)
- **Type:** Bagging ensemble of Decision Trees
- **File:** `models/random_forest_ensemble.joblib`
- **Key Characteristics:** Reduces variance through bagging, provides feature importance, robust to overfitting
- **Validation Accuracy:** 89.30%

### 6. XGBoost (Ensemble)
- **Type:** Gradient Boosting ensemble
- **File:** `models/xgboost_ensemble.joblib`
- **Key Characteristics:** State-of-the-art accuracy, built-in regularization, handles missing values
- **Validation Accuracy:** 90.12%

---

## Getting Started

### Prerequisites
```bash
# Install prerequisites
bash install_prerequisites.sh

# Or install Python dependencies directly
pip install -r requirements.txt
```

### Experiments
The project workflow is organized into two Jupyter Notebook stages:

1. **Stage 1 — Data Exploration & Preprocessing:** `experiments/stage1-data_exploration_and_preprocessing.ipynb`
2. **Stage 2 — Model Training & Evaluation:** `experiments/stage2-model_training_and_evaluation.ipynb`

---

## License

See [LICENSE](LICENSE) for details.
