import pandas as pd
import numpy as np
import seaborn as sns
import maatplotlib.pyplot as plt


# ================= COMPANY DATASET EDA =================
df = pd.read_csv("Company_Data.csv")

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescription:")
print(df.describe())


#--FIRST MOVEMENR BUSSINESS MOVEMENT-----#

from scipy import stats
print(df.mean(numeric_only=True))
print(df.median(numeric_only=True))
'''
here the features (CompPrice,Population,Price)
has high median value means there is more  outliwr in the featurs
'''
#--------SECOND MOVEMENT BUSSINESS MOVEMENT-----#

from scipy.stats import skew,kurtosis

print(df.var(numeric_only=True))
print(df.std(numeric_only=True))
'''
here the features (Population,Price,Income )
have high var and standard deviation it mean ther is more outlier in the column
it is usefull for training our ml model
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
here (CompPrice,Population,Price,Age)
are highly negatively skewed 
'''


#---------FOURTH MOVEMENT BUSSINESS DECISION-----#

print(df.kurtosis(numeric_only=True))

# missing values checking

print("\nMissing values:\n", df.isnull().sum())

# Distribution of Sales
sns.histplot(df["Sales"], kde=True)
plt.title("Sales Distribution")
plt.show()

# Boxplot for outliers
sns.boxplot(x=df["Sales"])
plt.title("Outliers in Sales")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="viridis")
plt.show()

'''
Inference:
- Sales distribution is slightly right-skewed.
- Stronger relationship between Sales and variables like Price & Income.
- Outliers exist in Sales.
- Dataset is complete, no missing values.
'''


