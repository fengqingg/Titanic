# Titanic Classification with Python

This project is a machine learning competition hosted by Kaggle, in which participants are tasked with predicting which passengers survived the Titanic shipwreck. This repository contains a Jupyter Notebook that explores the dataset, preprocesses the data, and trains a machine learning model to predict passenger survival.

## Data Exploration
Only 38.4 of the passengers survived in the dataset. 
<p align="center">
  <img width="409" alt="image" src="https://user-images.githubusercontent.com/85885666/233941404-357db14d-75c5-4a3c-b904-4892b8b85de5.png">
</p>

In the EDA, it can be seen that 'Sex','Pclass','Age','Fare','Embarked' are useful in predicting the survival rate of the passengers. The following plots shows the relationship between the features and the surival rate.
<p align="center">
<img width="404" alt="image" src="https://user-images.githubusercontent.com/85885666/233942281-678a4694-c33f-4b3e-beee-6dde1d83ba8c.png">
</p>
<p align="center">
<img width="404" alt="image" src="https://user-images.githubusercontent.com/85885666/233942336-c7f2b15b-5b0b-4741-8e20-867af4c40504.png">
</p>
<p align="center">
<img width="535" alt="image" src="https://user-images.githubusercontent.com/85885666/233942456-e1e9c5e7-2358-4c76-8d64-ec73f144e744.png">
</p>
<p align="center">
<img width="392" alt="image" src="https://user-images.githubusercontent.com/85885666/233942529-98872f4c-c140-409f-be7b-4914f9c3bfeb.png">
</p>
<p align="center">
<img width="392" alt="image" src="https://user-images.githubusercontent.com/85885666/233942572-98235d94-00be-4be9-b80f-ef7192c57f7c.png">
</p>

The features 'PassengerID','Name','Ticket','Cabin' were dropped as they were not correlated with the survival rate of the passengers. The columns 'Sex', 'Embarked' were OneHotEncoded as they were categorical datas. 

## Model Performance
The models used in this dataset is Logistic regression, kNN Classification, Decision Tree, Random Forest classifier, RBF SVMand XGBoost.
Cross validation was done to each of the model to provide a more robust and reliable way to assess model performance by splitting the data into multiple folds and training the model on different subsets of the data.
GridSearchCV was also used to search for the best parameter in each of the models and the accuracy for the training and testing data are as follows.

|                   | Logistic Regression | kNN Classification | Decision Tree | Random Forest Classifier | RBF SVM | XGBoost |
|:-----------------:|:-------------------:|:------------------:|:-------------:|:------------------------:|:-------:|:-------:|
| Training Accuracy |        79.6%        |        82.7%       |     82.4%     |           83.4%          |  83.3%  |  83.3%  |
|  Testing Accuracy |        79.9%        |        80.4%       |     77.7%     |           80.4%          |  80.4%  |  82.1%  |
  
The final model selected is the XGBoost due to the high test accuracy compared to the other models. This shows that the model is not overfitted to the training data.

## Submission
This model is then use to evaluate the competition test data which obtained a public score of 76.8%.

## Requirements
To run the notebook, you will need to have Python 3.6 or higher and the following libraries installed:

<ol>
  <li>Pandas</li>
  <li>Numpy</li>
  <li>Matplotlib</li>
  <li>Scikit-learn</li>
  <li>Seaborn</li>
</ol>

You can install these packages by running the following command using pip:
<code>
pip install pandas numpy matplotlib scikit-learn seaborn
</code>

## Usage
To use the notebook, you can simply open it in Jupyter and run each cell sequentially. The notebook is divided into several sections:

Introduction
Data Exploration
Data Cleaning and Preparation
Feature Selection
Model Building and Training
Model Evaluation
The notebook contains detailed explanations of each step, as well as code snippets and visualizations to help you understand the process.

## Dataset
The dataset used in this notebook is a collection of data on passengers aboard the Titanic ship. The dataset consists of a total of 891 instances, each representing a passenger, and includes 12 predictor variables and 1 binary target variable representing whether or not the passenger survived. The target variable takes on the value of 1 if the passenger survived and 0 if they did not survive. The predictor variables provide various attributes of the passengers, such as age, gender, class, fare, and more.

You can find the dataset at the following link: https://www.kaggle.com/c/titanic/data

## License
The code in this repository is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for more information.
