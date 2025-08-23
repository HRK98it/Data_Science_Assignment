import numpy as np
import seabron as sns
import matplotlib.pyplot as plt
import pandas as pd

# ================= HEART DISEASE DATASET EDA =================
df = pd.read_csv("heart disease.csv")

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescription:")
print(df.describe())

#--FIRST MOVEMENR BUSSINESS MOVEMENT-----#

from scipy import stats

print(df.mean(numeric_only=True))
'''
age          54.366337
sex           0.683168
cp            0.966997
trestbps    131.623762
chol        246.264026
fbs           0.148515
restecg       0.528053
thalach     149.646865
exang         0.326733
oldpeak       1.039604
slope         1.399340
ca            0.729373
thal          2.313531
target        0.544554

here we can see the average
'''
print(df.median(numeric_only=True))
'''
here the feature (age,trestbps,chol,thalach) this features have high median means it has high outliers
'''

#--------SECOND MOVEMENT BUSSINESS MOVEMENT-----#

from scipy.stats import skew,kurtosis

print(df.var(numeric_only=True))
print(df.std(numeric_only=True))
'''
here (trestbps,chol,thalach) this features have high var and std so it is usefull to train our model very well
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
here (age,sex, thalah,slope,thal,target) are highly negatively skewed
'''
#---------FOURTH MOVEMENT BUSSINESS DECISION-----#

print(df.kurtosis(numeric_only=True))
'''
here (age,sex,cp,restecg,thalach,exang,slope) they
have negative value meant they contain leess outlier
'''

print("\nMissing values:\n", df.isnull().sum())

# Target distribution
sns.countplot(x="target", data=df)
plt.title("Target Distribution")
plt.show()

# Age distribution
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="RdBu")
plt.title("Correlation Heatmap")
plt.show()

# Boxplot check
sns.boxplot(x=df["chol"])
plt.title("Outliers in Cholesterol")
plt.show()

'''
Inference:
- Target variable indicates presence/absence of heart disease.
- Cholesterol and age distributions show right-skewness.
- Chest pain type (cp) and maximum heart rate (thalach) are correlated with target.
- Some outliers exist in cholesterol.
'''
