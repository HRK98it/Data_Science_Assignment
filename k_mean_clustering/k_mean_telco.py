import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df=pd.read_excel('Telco_customer_churn.xlsx')

print("Shape of dataset:", df.shape)
print(df.head())

#------------------------Drop Customer ID (unique identifier, not useful for clustering)------------------
df = df.drop(columns=["Customer ID"], errors="ignore")

#---------------Handle categorical variables (One-Hot Encoding)--------------

df_encoded = pd.get_dummies(df, drop_first=True)

#--------------------Standardize the data-------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

#----------------- Elbow Method for optimal k----------------------
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

plt.plot(K, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Telco Customer Churn")
plt.show()

#-------------------- Train final KMeans (let's say k=4 after elbow curve)-----------------
k_optimal = 4
km_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df_encoded["Cluster"] = km_final.fit_predict(scaled_data)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df_encoded.groupby("Cluster").mean())
'''
Cluster 0 → Customers with low monthly charges, low revenue → budget/low-usage customers.

Cluster 1 → Customers with moderate charges, stable revenue, longer tenure → loyal mid-tier customers.

Cluster 2 → Customers with high monthly charges but low tenure → at high churn risk.

Cluster 3 → Customers with very high charges & revenue → premium customers (must retain).
'''
