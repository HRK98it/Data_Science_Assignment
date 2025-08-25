import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df=pd.read_excel('EastWestAirlines.xlsx')

print("Shape of dataset:", df.shape)
print(df.head())


#--------------Drop ID column (not useful for clustering)--------------------
df_num = df.drop(columns=["ID#"], errors="ignore")

#---------------standardize numeric data---------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_num)

# -----------Elbow Method to find optimal k-------------------------
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

plt.plot(K, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method - EastWestAirlines")
plt.show()

# ----------------Train final KMeans (let's say k=4 after elbow curve)-----------------------
k_optimal = 4
km_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(scaled_data)


df.groupby("Cluster").mean()


'''
Cluster 0 → Low balance, few miles, not frequent flyers.

Cluster 1 → Moderate balance, average bonus miles, casual flyers.

Cluster 2 → Very high balance, huge bonus miles, premium frequent flyers.

Cluster 3 → Medium balance, many transactions, mid-level loyalty members.
'''