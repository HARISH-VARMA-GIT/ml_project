# **Credit Risk Predictor using South German Bank Data**

This is an end-to-end ML project, which aims at developing a classification model for the problem of classifying a customer profile into 2 types based on the risk factor, using a given labeled dataset.

## **Table of Content**
1. Introduction
2. Approach
3. User Interface
4. Deployment Link
5. Installation
6. Technology Used
7. Document

## **Introduction**
Normally, most of the bank's wealth is obtained from providing credit loans so that a marketing bank must be able to reduce the risk of non-performing credit loans. The risk of providing loans can be minimized by studying patterns from existing lending data. One technique that you can use to solve this problem is to use data mining techniques. Data mining makes it possible to find hidden information from large data sets by way of classification. The goal of this project, is to build a model to predict whether the person, described by the attributes of the dataset, is a good (1) or a bad (0) credit risk..

**Input variables are:**
1. laufkont = status
2. laufzeit = duration
3. moral = credit_history
4. verw = purpose
5. hoehe = amount
6. sparkont = savings
7. beszeit = employment_duration
8. rate = installment_rate
9. famges = personal_status_sex
10. buerge = other_debtors
11. wohnzeit = present_residence 
12. verm = property
13. alter = age
14. weitkred = other_installment_plans
15. wohn = housing
16. bishkred = number_credits
17. beruf = job
18. pers = people_liable
19. telef = telephone
20. gastarb = foreign_worker

**Output variables:**

kredit = credit_risk

## **Approach**

Data Exploration : I started exploring dataset using pandas,numpy,matplotlib and seaborn.

Data visualization : Ploted graphs to get insights about dependend and independed variables.

Feature Engineering : All The Value Are Arrange In One Range.

Model Selection I : Tested all base models to check the base accuracy.

Model Selection II : Performed Hyperparameter tuning using gridsearchCV.

Pickle File : Selected model as per best accuracy and created pickle file using Pickle .

Webpage & deployment : 
1. Created a web form that takes all the necessary inputs from user and shows output. 
2. Deploy 

## **User Interface**

The Prediction of Credit Risk Final Model Run in Local Enviornment

**Home Page :**

**Prediction Page :**

**Output Page :**

## **Deployment Link**

AWS - 

## **Installation**

The Code is written in Python 3.8.11.If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

1. Create a Virtual Env with conda create "Your Env name"
2. pip install -r requirements.txt
3. Run app.py file

## **Technology Used**

1. Python
2. Sklearn
3. Pandas
4. Numpy
5. Seaborn
6. Matplotlib
7. Flask
8. HTML
9. AWS
10. Docker

## **Document**

Below providing the link of all the document that are required for creating the project.

Link: 

