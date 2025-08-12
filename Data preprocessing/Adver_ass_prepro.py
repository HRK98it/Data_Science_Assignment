import pandas as pd
import numpy as np

df=pd.read_csv('advertising.csv')

#--checking dtypes-----#

df.dtypes

#---lets convert datypes of some features---#

df=df.rename(columns={'Daily_Time_ Spent _on_Site':'Time','Daily Internet Usage':'internet','Ad_Topic_Line':'ads'})

df.dtypes

df['Time'] = pd.to_datetime(df['Time'])
df['Area_Income']=df['Area_Income'].astype(int)

df.dtypes
#-------data types has been converted--------#

#-----------checking duplicates---------#

df.duplicated().sum()
# there is no duplicates is the data set
# but if there are duplicated then we need to remove it using (df.drop_duplicates())

#-------Checking outliers-------------#

import seaborn as sns

sns.boxplot(df.select_dtypes(include='number'))
#as we can seee Area income has outlier we need to remove  it

sns.boxplot(df['Area_Income'])

'''there is various method to treat outliers but the usefull methot is Winsoriser so we used is'''

from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Area_Income'])
df1=winsor.fit_transform(df[['Area_Income']])
sns.boxplot(df['Area_Income'])
sns.boxplot(df1['Area_Income'])

'''as we can see the outlier in Arean_Incoome is removed'''

'''Now we need to remove 0 varience feature because it is not contributing in our model'''

#---------1. select only numeric column----------------#

numeric_df=df.select_dtypes(include='number')

#--------------2. find varience of each numeric column-------------#
variences=numeric_df.var()

#here male and Clicked on add these feature having 0 varience so we need to remove it 

#--------------3. identify columns with zero variene-----------#
zero_var_cols=variences[variences<1].index.tolist()
zero_var_cols

#-----now droping this feature form original dataframe

df_cleaned = df.drop(columns=zero_var_cols)
zero_var_cols

'''the features having low varience  is being removed'''

#--------------VALUE IMPUTATION----------------#

df=pd.read_csv('advertising.csv')

df.isnull().sum()
''' there is no null value in dataset so we create null value for imputation other wise there is no need of this 
'''
df.isnull().sum()
#now ther is null values in the columns

df=df.rename(columns={'Daily_Time_ Spent _on_Site':'Time','Daily Internet Usage':'internet','Ad_Topic_Line':'ads'})

'''Area_Income,internet,ads,country'''
#these having null value

#--------------mean imputation------------#

import numpy as np
from sklearn.impute import SimpleImputer

mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

df['Area_Income']= pd.DataFrame(mean_imputer.fit_transform(df[['Area_Income']]))
df['Area_Income'].isna().sum()
#the null values has been filled with mean


#---------------Medain imputation---------------#

median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')

df['internet']= pd.DataFrame(mean_imputer.fit_transform(df[['internet']]))
df['internet'].isna().sum()
#the null values has been imputed with median


#--------------mode imputation----------------#

'''Now mode is for categorical data in our case catogrical data is (country,ads)
so we need to do mode imputation'''

mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')

df['Country']= pd.DataFrame(mode_imputer.fit_transform(df[['Country']]))
df['Country'].isna().sum()

df['ads']= pd.DataFrame(mode_imputer.fit_transform(df[['ads']]))
df['ads'].isna().sum()

df.isnull().sum().sum()


'''Now ther is no nul value mean our data is finallu preprocesed'''





