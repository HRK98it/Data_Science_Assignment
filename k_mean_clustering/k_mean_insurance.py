import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('Insurance Dataset.csv')

print("Shape of dataset:", df.shape)
print(df.head())

#------Select only numeric columns-----------------------------
num_df = df.select_dtypes(include=["int64", "float64"])


#-------------------Standardize the data--------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)

#----------finding the best value of k_-----------------

TWSS = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_data)
    TWSS.append(km.inertia_)
    
plt.plot(k_range, TWSS, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Insurance Dataset")
plt.show()

# the best value is 3

km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(scaled_data)


print(df.groupby("Cluster").mean())

'''
Cluster 0 → Young customers, low premium, low claims.

Cluster 1 → Middle-aged, high income, high premiums, higher claims.

Cluster 2 → Older customers, moderate income, moderate claims.
'''