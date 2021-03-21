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
First we observe that this data is highly imbalanced where 99.83% of the transactions are non-fradulent and only 0.17% are fradulent transactions. Lets see how the t-sne visualization looks for the entire dataset. 
#### 1) T-SNE Visualization
t-SNE takes a high-dimensional dataset and reduces it to a low-dimensional graph that retains a lot of the original information. t-SNE measures the euclidean distance between two points and then plots that distance on a normal curve that is centered on the point of interest. Lastly, it takes the distance between point 2 and where it is on the normal curve.

![Model](https://github.com/arsalhuda24/credit_card_fraud_detection/blob/main/t_sne.png)

#### 2) Feature Importance
We also observe that there are some features which are highly skewed (eg Transaction_Amount). This needs to be avoided if we want our models to be interpretable. We will explore different techqniques (eg log-transformation) to take care of this problem. 

![Model](https://github.com/arsalhuda24/credit_card_fraud_detection/blob/main/feature_importance.png)

#### 3) Outlier Removal
Since we found that V10, V14 and V12 appear to be most important features. Lets visualize their distributions and box plots to identify any outliers. 

![Model](https://github.com/arsalhuda24/credit_card_fraud_detection/blob/main/outliers.png)

after removing the outliers we can visualize the box plots again. 

![Model](https://github.com/arsalhuda24/credit_card_fraud_detection/blob/main/outliers_removal.png)


### Model Building
The problem of fraud detection falls into the domain of anomaly identification where the legal transactions are termed as normal while fradulent ones as anomalies. There are number of ML techniques that we will test/evaluate and then carry out a thorough comparision to pick the best model.

1) One-Class Support Vector Machines (SVM) 
2) Isolation Forest 
3) Minimum Covariance Determinant 
4) Local Outlier Factor
5) Local Correlation Integral (LOCI)

#### One-Class SVM
A One-Class Support Vector Machine is an unsupervised learning algorithm that is trained only on the ‘normal’ data, in our case the positive examples (Non-Fradulent Transactions). It learns the boundaries of these points and is therefore able to classify any points that lie outside the boundary as, you guessed it, outliers.

![Model](https://github.com/arsalhuda24/credit_card_fraud_detection/blob/main/predictions.png)

