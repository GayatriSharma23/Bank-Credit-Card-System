# Credit Card Issuance Prediction Model

## Overview

This project implements a **Supervised Machine Learning Model** to predict the optimal conditions under which a bank can maximize profit by issuing credit cards to individuals. The project uses various predictive models, including **Random Forest**, **Decision Tree**, and **Linear Regression**, to determine whether issuing a credit card to a specific customer is a profitable decision or not.

The dataset for this project is sourced from the **UCI Machine Learning Repository** and includes data related to credit card fraud. The model predicts whether the bank should issue a credit card to a customer by analyzing historical data.

## Features

- **Supervised Learning**: Utilizes labeled data to train the model.
- **Predictive Models**: Implements Random Forest, Decision Tree, and Linear Regression.
- **Visualization**: Includes decision tree visualization to provide insights into the model's decision-making process.
- **Fraud Detection**: Incorporates data analysis to assess credit card fraud likelihood.

## Dataset

The dataset used in this project is sourced from the **UCI Machine Learning Repository**. It includes customer information, transaction history, and fraud indicators. The dataset is preprocessed to handle missing values and normalize features for optimal model performance.

## Models Used

1. **Random Forest**: Provides robust and accurate predictions by creating an ensemble of decision trees.
2. **Decision Tree**: Trained to visualize decision-making and understand key factors influencing predictions.
3. **Linear Regression**: Explores relationships between features and the target variable for baseline predictions.

## Project Workflow

1. **Data Preprocessing**:
   - Cleaned the dataset.
   - Handled missing values and performed feature scaling.
   - Split data into training and testing sets.

2. **Model Training**:
   - Trained models (Decision Tree, Random Forest, and Linear Regression) using the training dataset.
   - Evaluated performance using metrics like accuracy, precision, recall, and F1-score.

3. **Visualization**:
   - Visualized the Decision Tree to interpret the model's predictions.

4. **Prediction**:
   - Predicted whether the bank should issue a credit card to a specific customer.
   - Assessed the risk of fraud and potential profit for each case.

## Results

- **Model Accuracy**: Achieved high accuracy in predicting profitable credit card issuance decisions.
- **Decision Tree Visualization**: Offers insights into the factors influencing model decisions.

