import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('Company_data.csv')

df.shape

df.duplicated().sum()

#no duplicate values

df.dtypes

df['Income']=df['Income'].astype(float)
df['Price']=df['Price'].astype(float)
df.dtypes

#-----------outliers----------------#

from scipy.stats import zscore

outlier_cols = df.select_dtypes(include='number').columns[(abs(zscore(df.select_dtypes(include='number'))) > 3).any(axis=0)].tolist()

print("Columns with outliers:", outlier_cols)

sns.boxplot(df['Price'])
sns.boxplot(df['Advertising'])
sns.boxplot(df['CompPrice'])
sns.boxplot(df['Sales'])

#----------------removing outlier------------------#

from feature_engine.outliers import Winsorizer

a=['Sales', 'CompPrice', 'Advertising', 'Price']

win=Winsorizer(capping_method='iqr',fold=1.5,tail='both',variables=a)
df1 = win.fit_transform(df)
 
plt.figure(figsize=(12,12))
for i,val in enumerate(a):
    plt.subplot(2,2,i+1)
    sns.boxplot(df1[val])
    
# outlier has been removed

df.isnull().sum()  
# there is no null value so no need of imputation

#----------droping 0 varience features-------------#

varience=df.select_dtypes(include='number')    

var=varience.var()    
    
zero_var=var[var<1].index.to_list()

# there is no featues having zero varience
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    