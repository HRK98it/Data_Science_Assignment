import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('AutoInsurance.csv')

df.describe()

#----Lets drop th non rilivent feature-----------#
df2 = df.drop(columns=["Customer", "Effective To Date", "State", "Response", "Coverage",
                                "Education", "EmploymentStatus", "Gender", "Location Code",
                                "Marital Status", "Policy Type", "Policy", "Renew Offer Type",
                                "Sales Channel", "Vehicle Class", "Vehicle Size"], errors="ignore")

sns.pairplot(df2)
# here features a numeric and it is more so we directly apply clusterig

cpr=df2.select_dtypes(include="number").corr()
sns.heatmap(cpr,annot=True)

# -------------now lets do scaling----------------#

scaler = StandardScaler()
a=df.columns
df_std=pd.DataFrame(scaler.fit_transform(df2),columns=df2.columns)

# finding the optimal value of k
TWSS=[]
k_range=list(range(2,10))
for k in k_range:
    kmeans=KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_std)  
    TWSS.append(kmeans.inertia_)

plt.plot(k_range,TWSS, 'ro-')
plt.xlabel('Number of Clusters')
plt.ylabel('Total within sum of squares (TWSS)')
plt.title('Elbow Curve to determine Optimal K')
plt.show()
# here at 4 it slope is small 

model = KMeans(n_clusters=4,random_state=42)
model.fit(df_std)


df2['Cluster']= model.labels_

df= df2[['Cluster'] + list(df2.columns[:-1])]

df2.groupby(df2.Cluster).mean()

'''
Cluster 0
Customers with moderate lifetime value (~6676) and average claim amounts (~361).
→ These are medium-value, stable customers.

Cluster 1
Customers with slightly higher lifetime value (~6916) and claims (~382) than Cluster 0.
→ Another mid-range group, not very different from Cluster 0.

Cluster 2
Customers with very high lifetime value (~18,414) and high claim amounts (~927).
→ These are high-value premium customers, contributing the most revenue but also making bigger claims.

Cluster 3
Customers with lower lifetime value (~6559) but similar claim levels (~386).
→ These are low-value customers who might not be very profitable in the long run.
'''
df.groupby('Cluster').mean()
 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    