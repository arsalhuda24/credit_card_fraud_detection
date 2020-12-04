# Fraud Detection
This repository contains end-end machine learning based solution for detecting fraudulent transactions

## Machine Learning Project Life Cycle.

![Model](https://github.com/arsalhuda24/credit_card_fraud_detection/blob/main/machine_learning_project_lifecycle.bmp)

### Business Problem

It is imperative for credit card companies and banks to detect fradulent transaction as early as possible. In this work we use the credit card fraud detection data provided by kaggle competion and build a robust model that can capture these transactions. At the end we will deploy this model for realtime fraud detection.  

### Challenges in fraud detection 
Fraud detection systems are prune to many difficulties. An efficient system shouild be able to address following challenges to achieve best performance
1) Highly imbalanced datasets which means that there are far more normal trasactions compared to fraudulent ones.
2) Misclassification Importance: 



### Data Preprocessing
1) First we observe that this data is highly imbalanced where 99.83% of the transactions are non-fradulent and only 0.17% are fradulent transactions. 
2) We also observe that there are some features which are highly skewed (eg Transaction_Amount). This needs to be avoided if we want our models to be interpretable. We will explore different techqniques (eg log-transformation) to take care of this problem.  
