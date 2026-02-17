'''
Now in EDA we have seen that the most imp and relevant feantures are the age and smoker by iusing this features we can train our model
'''
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df=pd.read_csv('insurance.csv')

X=df.drop("charges", axis=1)
y=df["charges"]


numerical_features=["age", "bmi", "children"]
categorical_features=["sex", "smoker", "region"]

# now we will create the pipeline for numeric feature processing

num_pipeline=Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# now for catgorical data

cat_pipeline=Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first"))
])

preprocessor=ColumnTransformer(transformers=[
    ("num", num_pipeline, numerical_features),
    ("cat", cat_pipeline, categorical_features)
])

# create full pipeline with linear regression 

from sklearn.linear_model import LinearRegression

model=Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
                            X,y,test_size=0.2,random_state=42
                            )


model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred=model.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2:", r2)

'''
RMSE: 5796.284659276275
R2: 0.7835929767120722
'''