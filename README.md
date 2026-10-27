# Toxic Comment Detection Using Machine Learning and NLP

## Project Overview

This project explores automated toxic comment classification using Natural Language Processing (NLP) and machine learning. It compares Logistic Regression, Linear Support Vector Machine (SVM), and Long Short-Term Memory (LSTM) models to identify potentially toxic online comments.

The project demonstrates text preprocessing, feature extraction, model development, hyperparameter tuning, and performance evaluation.

## Objectives

- Explore and prepare the toxic comment dataset.
- Analyse the distribution and characteristics of comments.
- Preprocess text for machine learning.
- Apply TF-IDF feature extraction for classical machine learning models.
- Develop and compare Logistic Regression, Linear SVM, and LSTM models.
- Tune model hyperparameters using GridSearchCV.
- Evaluate classification performance using appropriate metrics.
- Discuss practical applications and limitations of automated content moderation.

## Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow / Keras
- Natural Language Processing (NLP)
- TF-IDF
- LSTM neural networks
- Jupyter Notebook / Google Colab

## Methodology

1. **Data preparation:** Inspect and prepare the dataset for analysis.
2. **Exploratory data analysis:** Examine class distribution and comment characteristics.
3. **Text preprocessing:** Prepare text for feature extraction and modelling.
4. **Feature extraction:** Use TF-IDF for classical models and token sequences for the LSTM model.
5. **Model development:** Train Logistic Regression, Linear SVM, and LSTM classifiers.
6. **Hyperparameter tuning:** Use GridSearchCV to tune the Linear SVM model.
7. **Model evaluation:** Compare performance using precision, recall, F1-score, and ROC-AUC.
8. **Error analysis:** Consider misclassifications and the limitations of automated toxicity detection.

## Model Comparison

The notebook compares three approaches:

| Model | Approach |
|---|---|
| Logistic Regression | TF-IDF-based classification |
| Linear SVM | TF-IDF-based classification with hyperparameter tuning |
| LSTM | Deep learning for text classification |

The notebook's validation results identify the LSTM as the strongest of the three models by F1-score. The detailed results and evaluation metrics are available in the notebook.

## Practical Applications

Toxic comment detection can help online platforms screen user-generated content and prioritise potentially harmful comments for human moderation. Such systems can support moderation workflows but should not replace human judgement.

## Limitations

- Toxicity can depend on context, sarcasm, language, and cultural differences.
- Class imbalance may affect model performance.
- False positives can incorrectly flag acceptable comments, while false negatives can miss harmful content.
- Performance may vary across communities and unseen data.
- Human review and further validation are important before real-world deployment.

## How to Run

1. Open `toxic_comment_detection_analysis.ipynb` in Google Colab or Jupyter Notebook.
2. Obtain the dataset required by the notebook.
3. Follow the notebook's instructions for data loading and dependency installation.
4. Run the cells in order.
5. Review the exploratory analysis, model comparison, evaluation metrics, and error analysis.

## Project Purpose

This project demonstrates practical skills in Natural Language Processing, text classification, machine learning, deep learning, hyperparameter tuning, and model evaluation.
