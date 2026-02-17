from preprocessing import X_train,X_test,y_train,y_test
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Random forest
rf = RandomForestRegressor(
                            n_estimators=100,
                            random_state=42
                            )
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Results")
print("RMSE:", rmse_rf)
print("R2:", r2_rf)

'''RMSE: 5796.284659276275
R2: 0.7835929767120722'''