import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= BANK DATASET EDA =================
df = pd.read_csv("bank_data.csv")

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescription:")
print(df.describe(include="all"))


#--FIRST MOVEMENR BUSSINESS MOVEMENT-----#

from scipy import stats
print(df.mean(numeric_only=True))
print(df.median(numeric_only=True))
'''
here every features has mean and median 0 but some features (balance,duration,age)
have high mean and median'''

#--------SECOND MOVEMENT BUSSINESS MOVEMENT-----#

from scipy.stats import skew,kurtosis

print(df.var(numeric_only=True))
print(df.std(numeric_only=True))
'''
here each feature has var nearly 1 but some features like (pdays,duration,balance)
have high standard deviation\
    '''
    
#----------THIRD MOVEMENT BUSSINESS DECISION-----#

print(df.skew(numeric_only=True))


a=df.select_dtypes(include='number').columns.tolist()
plt.figure(figsize=(8,8))
for i, val in enumerate(a):
    plt.subplot(3, 2, i+1)
    sns.histplot(df[val], kde=True)
    plt.title(f'Distribution of {val}')
    plt.tight_layout()
    plt.show()
'''
here most of the features are right skewed wiht value in between 1 to 4 but some feature(previous)
are heavily right skewed'''

#---------FOURTH MOVEMENT BUSSINESS DECISION-----#

print(df.kurtosis(numeric_only=True))
'''
the features (balance ,previous ,jounknown,)
has high pickedness
'''

# missing values


print("\nMissing values:\n", df.isnull().sum())

# Target distribution
sns.countplot(x="y", data=df)
plt.title("Target Distribution (y)")
plt.show()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(df["age"], kde=True, bins=20)
plt.title("Age Distribution")
plt.show()

# Balance distribution
plt.figure(figsize=(6,4))
sns.histplot(df["balance"], kde=True, bins=30)
plt.title("Balance Distribution")
plt.show()

# Boxplots for outliers
sns.boxplot(x=df["balance"])
plt.title("Balance Outliers")
plt.show()

# Correlation for numeric
plt.figure(figsize=(6,4))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

'''
Inference:
- Target variable 'y' is imbalanced (more 'no' than 'yes').
- Age is slightly right-skewed, with many customers between 25-40.
- Balance column has strong outliers.
- Numeric correlations are weak.
'''
