import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('crime_data.csv')

df.describe()

df=df.rename(columns={'Unnamed: 0':'cuntry'})
sns.pairplot(df)

cpr=df.select_dtypes(include="number").corr()
sns.heatmap(cpr,annot=True)

df.drop(['cuntry'],axis=1,inplace=True)

scaler = StandardScaler()

df_std=pd.DataFrame(scaler.fit_transform(df.iloc[:,1:]),columns=df.columns[1:])

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

model = KMeans(n_clusters=3,random_state=42)
model.fit(df_std)


df['Cluster']= model.labels_

df= df[['Cluster'] + list(df.columns[:-1])]

df.iloc[:,2:].groupby(df.Cluster).mean()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    