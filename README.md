# AI-Powered Support Ticket Classification and Prioritization

An NLP and Machine Learning project that automatically categorizes IT support tickets and predicts their priority using text classification.

## Project Overview

Support teams receive large numbers of tickets every day. Manually categorizing and prioritizing them can be time-consuming.

This project uses Natural Language Processing (NLP) and Machine Learning to:

- Categorize support tickets into IT service categories.
- Predict ticket priority as Low or Medium.
- Convert ticket text into numerical features using TF-IDF.
- Train and evaluate machine learning classifiers.

## Dataset

- Total tickets: 300
- Multilingual IT support ticket descriptions
- Text preprocessing and TF-IDF feature extraction
- Unclassified and insufficiently represented examples excluded from relevant model training.

### Priority distribution

| Priority | Tickets |
|---|---:|
| Low | 172 |
| Medium | 101 |
| High | 2 |
| Unclassified | 25 |

The priority classifier is trained on Low and Medium tickets only. High-priority examples were insufficient for reliable supervised learning.

## Machine Learning Models

### 1. Category Classification

Models evaluated:
- Logistic Regression with word-level TF-IDF.
- Logistic Regression with combined word and character TF-IDF.
- Linear Support Vector Machine (Linear SVM).

Selected model: Linear SVM.

Five-fold cross-validation:
- Mean accuracy: 68.24%
- Mean macro F1-score: 65.62%

Holdout accuracy: 66.67%.

### 2. Priority Classification

Selected model: Linear SVM with combined word and character TF-IDF features.

Holdout evaluation:

| Metric | Score |
|---|---:|
| Accuracy | 83.64% |
| Macro Precision | 0.83 |
| Macro Recall | 0.85 |
| Macro F1-score | 0.83 |

Five-fold cross-validation accuracy: 80.61%.

The model currently distinguishes Low and Medium priorities. High-priority classification requires more verified examples.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Linear SVM
- Matplotlib
- Joblib

## Project Structure

```text
FUTURE_ML_02/
│
├── data/
│   ├── priority_training.csv
│   └── priority_review.csv
│
├── models/
│   ├── category_classifier_svm.joblib
│   └── priority_classifier.joblib
│
├── results/
│   └── priority_confusion_matrix.png
│
├── train_category_svm.py
├── train_priority_model.py
├── predict_category.py
├── prepare_priority_dataset.py
├── review_priority_batch.py
├── save_priority_batches5_6_7.py
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib joblib
```

Train the priority classifier:

```bash
python train_priority_model.py
```

The trained model is saved in:

```text
models/priority_classifier.joblib
```

## Evaluation

Models were evaluated using:
- Train-test split.
- Accuracy, precision, recall and F1-score.
- Five-fold cross-validation.
- Confusion matrix analysis.

## Limitations

- The dataset is relatively small and contains imbalanced categories.
- The priority model currently supports Low and Medium classes only.
- More verified High-priority examples are required.
- Predictions depend on the quality and consistency of ticket descriptions and labels.

## Future Improvements

- Collect additional verified support tickets.
- Improve multilingual text preprocessing.
- Introduce a reliable High-priority classification class.
- Experiment with additional NLP models.
- Deploy the classifier through a simple web application.

## Project Objective

To demonstrate how NLP and supervised machine learning can help automate IT support ticket categorization and prioritization.
