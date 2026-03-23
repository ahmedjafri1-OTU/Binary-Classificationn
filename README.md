# Binary-Classification
Binary Classification App using Streamlit - Confusion matrix, ROC Curve, Precision Recall Curve
### Project Overview
This project builds a machine learning application to classify mushrooms as edible or poisonous based on their physical characteristics. The model is trained using the Mushroom dataset from the UCI Machine Learning Repository.
Multiple classification algorithms are implemented and compared to evaluate their predictive performance. An interactive web interface built with Streamlit allows users to train models and visualize evaluation metrics.
### Dataset Description
The dataset contains descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota family.
Each species is labeled as:
* Edible
* Poisonous
* Unknown edibility
  
The unknown category is grouped with the poisonous class for safety purposes.
The dataset includes multiple categorical attributes describing mushroom characteristics such as:

* Cap shape
* Cap color
* Odor
* Gill size
* Gill color
* Stalk shape
* Habitat
  
The objective is to predict whether a mushroom is edible or poisonous based on these features.
### Machine Learning Models
The following classification algorithms are implemented and compared:
* Random Forest Classifier
* Support Vector Machine (SVM)
* Logistic Regression
  
These models are used to analyze the dataset and predict the edibility of mushrooms.
### Model Evaluation Metrics
The models are evaluated using standard classification metrics:
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
  
These metrics provide insights into model performance and allow comparison between the different algorithms.
### Streamlit Application
Streamlit is an open-source Python framework used to build interactive web applications for data science and machine learning.
In this project, Streamlit provides a simple interface that allows users to:
Select different machine learning algorithms
* Train models on the dataset
* Compare model performance
* Visualize evaluation metrics such as confusion matrix, ROC curve, and precision-recall curve
* This interactive dashboard makes it easier to explore how different algorithms perform on the mushroom classification problem.
### Results
The application enables comparison of multiple machine learning models and their classification performance. Visualization tools help analyze prediction accuracy and understand how well each model distinguishes between edible and poisonous mushrooms.
