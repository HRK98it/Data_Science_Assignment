import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv("D:/assignment/heart disease.csv")

df.dtypes

df['sex']=df['sex'].map({1:'M',0:'F'})

df.dtypes

df.duplicated().sum()

df=df.drop_duplicates()
df.duplicated().sum().sum()


df.isnull().sum()

#---------checking outlier------------W

from scipy.stats import zscore

outlier_cols = df.select_dtypes(include='number').columns[(abs(zscore(df.select_dtypes(include='number'))) > 3).any(axis=0)].tolist()

sns.boxplot(df[outlier_cols])

#---checking varience--------#

var=df[outlier_cols].var()

z_var=var[var<9].index

#-----droping low var -----------#

df=df.drop(columns=z_var)

#----------removing outlier---------------#

from feature_engine.outliers import Winsorizer

win=Winsorizer(capping_method='iqr',fold=1.5,tail='both',variables=outlier_cols)

df1=win.fit_transform(df)

sns.boxplot(df1[outlier_cols])

#--------outliers has been removed








