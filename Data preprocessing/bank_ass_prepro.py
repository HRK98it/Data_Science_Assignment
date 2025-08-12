import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv('bank_data.csv')

df.dtypes

df=df.rename(columns={'poutfailure':'fail','poutsuccess':'success','con_telephone':'telephone'})
df.dtypes

df['fail']=df['fail'].astype(float)
df['success']=df['success'].astype(float)

df.dtypes

#---------duplicates--------------#

df.duplicated().sum()
#there is one duplicate remove it

df=df.drop_duplicates()

df.duplicated().sum()

#---------------outliers---------------#

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(df['balance'])
sns.boxplot(df['duration'])    

#----------removing outliers---------------#

from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['balance'])
df1=winsor.fit_transform(df[['balance']])
sns.boxplot(df['balance'])
sns.boxplot(df1['balance'])

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['duration'])
df1=winsor.fit_transform(df[['duration']])
sns.boxplot(df['duration'])
sns.boxplot(df1['duration'])
    
#outliers has been removed

#checking null values

df.isnull().sum()
#htere is no null values 


















