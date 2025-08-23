import pandas as pd
import numpy as np
import seabron as sns
import matplotlib.pyplot as plt
import pandas as pd

# ================= EAST WEST AIRLINES DATASET EDA =================
df = pd.read_excel("EastWestAirlines.xlsx")

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescription:")
print(df.describe())


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


print("\nMissing values:\n", df.isnull().sum())

# Histograms for numeric variables
for col in df.select_dtypes("number").columns[:6]:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplots
sns.boxplot(x=df["Balance"])
plt.title("Outliers in Balance")
plt.show()

'''
Inference:
- Balance and Bonus_miles show extreme skewness.
- Strong correlation exists between related variables like Bonus_miles and Bonus_trans.
- Several high outliers are present.
- No missing values in dataset.
'''
