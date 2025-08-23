import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ================= MTCARS DATASET EDA =================
df = pd.read_csv("mtcars.csv")

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescription:")
print(df.describe())

#--FIRST MOVEMENR BUSSINESS MOVEMENT-----#

from scipy import stats

print(df.mean(numeric_only=True))
'''
the features (dusp,hp,17,) have high mean 
'''
print(df.median(numeric_only=True))
'''
the features (dusp,hp,17,) have high median
'''

#--------SECOND MOVEMENT BUSSINESS MOVEMENT-----#

from scipy.stats import skew,kurtosis

print(df.var(numeric_only=True))
print(df.std(numeric_only=True))
'''
the features (disp,hp,mpg') have high varience and std means this features is helpfull for our model'''
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
here only Feature cyl is having negatice skewness most of the cylinder is of less power'''
#---------FOURTH MOVEMENT BUSSINESS DECISION-----#

print(df.kurtosis(numeric_only=True))

print("\nMissing values:\n", df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="Greens")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Outlier check
sns.boxplot(x=df["mpg"])
plt.title("Outliers in MPG")
plt.show()

'''
Inference:
- mpg is negatively correlated with wt and cyl.
- Higher cylinder cars → lower fuel efficiency.
- Strong relationships between hp, wt, and qsec.
- Dataset has no missing values.
'''
