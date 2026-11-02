# TicketIQ — Support Ticket Classification & Prioritization

An NLP-powered machine learning system that automatically classifies IT support tickets and predicts their priority levels. Built with Python, Scikit-learn, and Streamlit.

## Overview

TicketIQ helps streamline support ticket management by using machine learning to:

- Categorize incoming support tickets.
- Predict ticket priority as **Low** or **Medium**.
- Process ticket descriptions using TF-IDF feature extraction.
- Compare classification models and evaluate their performance.
- Explore results through an interactive Streamlit dashboard.

## Interactive Dashboard

TicketIQ includes a Streamlit dashboard with four sections:

- **Overview:** Ticket statistics and category/priority distributions.
- **Ticket Classifier:** Enter a support ticket and get a predicted category.
- **Priority Predictor:** Enter a ticket and predict whether its priority is Low or Medium.
- **Model Insights:** Review model evaluation metrics and the priority confusion matrix.

## Machine Learning Approach

### Ticket Category Classification

Models were compared using TF-IDF features and classification algorithms, including Logistic Regression and Linear SVM.

The selected model is a **Linear Support Vector Machine (LinearSVC)** using word-level and character-level TF-IDF features.

| Evaluation metric | Result |
|---|---:|
| 5-fold cross-validation accuracy | 68.24% |
| 5-fold cross-validation macro F1 | 65.62% |
| Holdout accuracy | 66.67% |

### Priority Classification

The priority model uses word-level and character-level TF-IDF features with a LinearSVC classifier and