# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

STEP 1: Strat program.

STEP 2: Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

STEP 3: Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

STEP 4: Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

STEP 5: Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

STEP 6: Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

STEP 7: Evaluate the clustering results: Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

STEP 8: Select the best clustering solution: If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements.

STEP 9: End program.

## Program:



Program to implement the K Means Clustering for Customer Segmentation.
Developed by: ANUVIND KRISHNA.K
RegisterNumber: 212223080004

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")


## Output:
data.head() 

![image](https://github.com/Preetha-Senthamilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119390282/dce1b89d-03a1-4798-8c5e-1833e77785b5)

data.info()

![image](https://github.com/Preetha-Senthamilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119390282/1e2edcde-62c9-4d33-8185-40b6474d1ccc)

Null Values 

![image](https://github.com/Preetha-Senthamilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119390282/ecaff80e-a0c5-4b99-80ba-8a8785e2948e)

Elbow Graph 

![image](https://github.com/Preetha-Senthamilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119390282/782e8b5b-4945-40eb-9d67-aac0852fc4a6)

K-Means Cluster Formation

![image](https://github.com/Preetha-Senthamilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119390282/33cf9558-4554-4ac1-8e69-a683e810021c)


Predicted Value

![image](https://github.com/Preetha-Senthamilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119390282/cc326c5f-f1b1-4465-8938-16c573818cac)


Final Graph

![image](https://github.com/Preetha-Senthamilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119390282/6b0162c3-a8e8-4b3d-8a09-0cfc72342fc0)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
