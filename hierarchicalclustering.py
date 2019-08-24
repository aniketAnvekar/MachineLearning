# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:29:20 2019

@author: anike
"""
import pandas as pd
import matplotlib.pyplot as plt
#Reading the data 
data = pd.read_csv('C:\\Users\\anike\\Documents\\python\Customers.csv')
#taking input as AnnualIncome and PendingScore into X
X = data.iloc[:,[3,4]].values
#Creating dendogram for the dataset
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian distance')
plt.show()
#Using AgglomerativeClustering to cluster the data wkere no.of clusters k=5
from sklearn.cluster import AgglomerativeClustering
ahc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_predict = ahc.fit_predict(X)
#mapping of clusters 
plt.scatter(X[y_predict==0,0],X[y_predict==0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_predict==1,0],X[y_predict==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_predict==2,0],X[y_predict==2,1],s=100,c='green',label='Cluster3')
plt.scatter(X[y_predict==3,0],X[y_predict==3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(X[y_predict==4,0],X[y_predict==4,1],s=100,c='magenta',label='Cluster5')
plt.title('Clusters of the Customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(0-100)')
plt.legend()
plt.show()