import pandas as pd
import numpy as np
import seabron as sns
import matplotlib.pyplot as plt

#================= CRIME DATASET EDA =================
df = pd.read_csv("crime_data.csv")

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
here (Assualt) feature have high mean and median means it has outliers'''

#--------SECOND MOVEMENT BUSSINESS MOVEMENT-----#

from scipy.stats import skew,kurtosis

print(df.var(numeric_only=True))
print(df.std(numeric_only=True))
'''
same here Assult have high var and std as compared to other mean it is helpfull to out ml model
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
here only UrbonPop is negatively skewed other  are 
near to zero
'''


#---------FOURTH MOVEMENT BUSSINESS DECISION-----#

print(df.kurtosis(numeric_only=True))


print("\nMissing values:\n", df.isnull().sum())

# Pairplot
sns.pairplot(df)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplots for outlier check
for col in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Outlier check for {col}")
    plt.show()

'''
Inference:
- UrbanPop is strongly correlated with Assault.
- Data looks suitable for clustering / PCA.
- Several variables show outliers.
- No missing values.
'''
