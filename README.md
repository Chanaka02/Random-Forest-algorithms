# Random-Forest-algorithms

Intelligent Systems & Machine Learning

This repository contains the implementation of **Random Forest** algorithms (both Classification and Regression)

## 📋 Project Overview
The lab focuses on implementing ensemble learning techniques using Random Forests to tackle distinct classification and continuous value prediction tasks:

### 1: Random Forest Classification
* **Dataset:** `heart.csv`
* **Objective:** Classify observations into correct target outputs (e.g., predicting heart disease risk).
* **Methodology:** 
  * Analyzing output class distribution.
  * Splitting features (\(X\)) and target (\(y\)) into training and testing sets.
  * Building a base `RandomForestClassifier` with 100 decision trees.
  * Evaluating model behavior via Decimal/Percentage Accuracy, Confusion Matrices, and full Classification Reports.
  * Tuning and benchmarking performance over varied tree counts (10, 50, 100, and 200 trees).

### 2: Random Forest Regression
* **Dataset:** `Student_Performance.csv`
* **Objective:** Predict continuous student performance indicators based on study and extracurricular metrics.
* **Methodology:** 
  * Categorical encoding of variables (such as Extracurricular Activities) using `LabelEncoder`.
  * Training a `RandomForestRegressor` ensemble model.
  * Evaluating continuous predictions using standard regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and \(R^2\) Score.
  * Analyzing visual and numeric actual-vs-predicted comparisons across different tree architectures (10 to 200 estimators).

## 🛠️ Tech Stack & Dependencies
The practical is implemented using Python inside Google Colab, leveraging:
* **Pandas** & **NumPy** — Data preparation, structural transformation, and mathematical evaluation.
* **Matplotlib** — Matrix mapping and visualization plots.
* **Scikit-Learn** — Preprocessing (`LabelEncoder`), core models (`RandomForestClassifier`, `RandomForestRegressor`), and robust metric functions (`confusion_matrix`, `classification_report`).

## 🚀 How to Run the Notebook
1. Clone this repository to your environment:
   ```bash
   git clone https://github.com
   ```
2. Upload the completed notebook to [Google Colab](https://google.com).
3. Make sure the dataset assets (`heart.csv` and `Student_Performance.csv`) are placed within the local working execution directory.
4. Step through the program sequentially to check performance metrics across the estimator tables.
