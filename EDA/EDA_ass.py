import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
data frame is advertising'''

df=pd.read_csv('advertising.csv')

df.info()
'''
there are 1000 entries in dataframe with total 10 columns'''

df=df.rename(columns={'Daily_Time_ Spent _on_Site':'Dailytime','Daily Internet Usage':'Int_Usage','Ad_Topic_Line':'AdTopic'})
df.dtypes
df.shape
df.describe()
'''
here dailytime is left skewed 
Age : slightly right skewed 
Male : right skewed
Clicked add : is perfectly symmetrical
'''
#--FIRST MOVEMENR BUSSINESS MOVEMENT-----#

from scipy import stats
print(df.mean(numeric_only=True))
print(df.median(numeric_only=True))

#--------SECOND MOVEMENT BUSSINESS MOVEMENT-----#

from scipy.stats import skew,kurtosis

print(df.var(numeric_only=True))
print(df.std(numeric_only=True))

#----------THIRD MOVEMENT BUSSINESS DECISION-----#

print(df.skew(numeric_only=True))
'''
Dailytime       -0.371760
Age              0.479142
Area_Income     -0.650373
Int_Usage       -0.033537
Male             0.076169
Clicked_on_Ad    0.000000

here [daily time ,area income, internet usage ] are newgataively skewed
'''
a=df.select_dtypes(include='number').columns.tolist()
plt.figure(figsize=(8,8))
for i, val in enumerate(a):
    plt.subplot(3, 2, i+1)
    sns.histplot(df[val], kde=True)
    plt.title(f'Distribution of {val}')
    plt.tight_layout()
    plt.show()


#---------FOURTH MOVEMENT BUSSINESS DECISION-----#

print(df.kurtosis(numeric_only=True))
'''
Dailytime       -1.095534
Age             -0.400524
Area_Income     -0.099810
Int_Usage       -1.272659
Male            -1.998199
Clicked_on_Ad   -2.004012
'''

#------------DATA CLEANING------------#

df.isnull().sum()
'''
As we can see there is no null value'''

df.duplicated().sum()
'''there is no duplicates value in the dataset'''

#------------VISUALIZATION---------------#

sns.pairplot(df)
cor=df.select_dtypes(include='number').corr()
sns.heatmap(cor,annot=True)
'''
as heatmap shows the most corelated features are daily time with feature 
click on ad also feature Internet usage
'''





