# Early Detection of Amyloidosis Using Machine Learning

This repository contains the complete source code, datasets, and results for the Master's Thesis project on the early detection of Amyloidosis using machine learning. The project explores and compares different modeling strategies on two distinct datasets to develop a scientifically valid and effective predictive tool.

## Abstract

Amyloidosis is a rare, fatal disease caused by the deposition of abnormal amyloid proteins, leading to organ failure. The cardiac variant, Transthyretin Amyloid Cardiomyopathy (ATTR-CM), is particularly aggressive, and since treatments can only slow its progression, early diagnosis is critical. However, current diagnostic methods are often late and invasive. This project addresses this challenge by developing and evaluating supervised machine learning models, specifically a Multi-Layer Perceptron (MLP) and a TabTransformer, on a large-scale patient dataset from IQVIA. The research successfully demonstrates that these models can accurately identify patients at high risk for ATTR-CM using only their pre-diagnostic medical history. A key finding was the success of the TabTransformer, which confirmed that analyzing the temporal sequence of medical events is a highly effective strategy for true early detection, even when all direct diagnostic codes for amyloidosis are removed. The significance of this work is twofold: it provides a blueprint for a practical clinical screening tool and advances the methodology for predictive health analytics.

## Repository Structure

The project is organized into two main folders, Approach 1 and Approach 2, each representing a distinct experimental track.
```
.
└── 00_Final_Project_MENG/
    ├── Approach 1/
    │   ├── Dataset_1/                # Contains the pre-processed binary feature data and the split notebook.
    │   ├── MLP/                      # Saved artifacts (model, scaler, predictions) for the MLP on Dataset 1.
    │   ├── MLP-Notebooks/            # Jupyter notebooks for the MLP workflow (Train, Val, Test).
    │   ├── TabTransformer/           # Saved artifacts for the TabTransformer on Dataset 1.
    │   └── TabTransformer-Notebooks/ # Jupyter notebooks for the TabTransformer workflow.
    │
    └── Approach 2/
        ├── Dataset_2/                # Contains the raw patient code data and ICD map.
        ├── MLP/                      # Saved artifacts for the MLP on Dataset 2 (TF-IDF).
        ├── MLP_Notebooks/            # Jupyter notebooks for the MLP (TF-IDF) workflow.
        ├── TabTransformer/           # Saved artifacts for the final TabTransformer model.
        └── TabTransformer_Notebooks/ # Notebooks for the final TabTransformer (leakage-corrected).

```

## Methodology Overview

This project implements and evaluates four distinct models:

1.  **MLP on Dataset 1:** A baseline Multi-Layer Perceptron trained on the pre-processed binary features to establish a performance benchmark. Implemented in TensorFlow/Keras.
2.  **TabTransformer on Dataset 1:** An attention-based model trained on the binary features to determine if a more complex architecture could find a stronger signal. Implemented using `pytorch-tabular`.
3.  **MLP on Dataset 2 (TF-IDF):** An MLP trained on features engineered from raw medical codes using TF-IDF vectorization. **This model was invalidated due to the discovery of severe data leakage**, a critical methodological finding of the project.
4.  **TabTransformer-MLP Hybrid on Dataset 2 (Final Model):** The project's flagship model. A custom Transformer architecture trained on sanitized, sequential patient data where all direct Amyloidosis codes were removed to prevent data leakage. This represents a scientifically valid early detection system.

## Getting Started

### Prerequisites

To run this project, you will need Python (3.9+ recommended) and the libraries listed below. It is highly recommended to use a virtual environment.

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repository.git]
    cd your-repository
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required libraries:
  libraries like: tensorflow, torch, pytorch-tabular, scikit-learn, pandas, numpy, seaborn, matplotlib, shap, joblib

### Execution Order

To reproduce the results, the Jupyter notebooks for each model should be run in a specific order: **Training -> Validation -> Testing**. The training notebook generates the model artifacts that the other two notebooks depend on.

**Approach 1: Baseline Models**
*   **MLP:**
    1.  `Approach 1/MLP-Notebooks/MLP_Model_Training.ipynb`
    2.  `Approach 1/MLP-Notebooks/MLP_Model_Validation.ipynb`
    3.  `Approach 1/MLP-Notebooks/MLP_Model_Testing.ipynb`
*   **TabTransformer:**
    1.  `Approach 1/TabTransformer-Notebooks/Transformer_Model_Training.ipynb`
    2.  `Approach 1/TabTransformer-Notebooks/Transformer_Model_Validation.ipynb`
    3.  `Approach 1/TabTransformer-Notebooks/Transformer_Model_Testing.ipynb`

**Approach 2: Advanced Models**
*   **MLP (TF-IDF):**
    1.  `Approach 2/MLP_Notebooks/MLP_Model_Training.ipynb`
    2.  `Approach 2/MLP_Notebooks/MLP_Model_Validation.ipynb`
    3.  `Approach 2/MLP_Notebooks/MLP_Model_Testing.ipynb`
*   **TabTransformer (Final Model):**
    1.  `Approach 2/TabTransformer_Notebooks/training_tabtransformer.ipynb`
    2.  `Approach 2/TabTransformer_Notebooks/validation_tabtransformer.ipynb`
    3.  `Approach 2/TabTransformer_Notebooks/testing_TabTransformer.ipynb`

## Key Results Summary

The following table summarizes the final, unbiased performance of the valid models on their respective test sets. The MLP on Dataset 2 is excluded as it was invalidated by data leakage.

| Model | Dataset & Feature Type | ROC-AUC | F1-Score (Positive Class) | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP (Baseline)** | Dataset 1 (Binary) | 0.874 | 0.785 | 0.776 | 0.795 |
| **TabTransformer (Final Model)** | Dataset 2 (Sequential, Corrected) | **0.892** | **0.807** | **0.880** | **0.745** |

The final leakage-corrected TabTransformer model successfully surpassed the strong baseline, demonstrating the effectiveness of analyzing the temporal sequence of patient data for true early detection.

## Author

*   **Sreelakshmi Krishnakumar** 
