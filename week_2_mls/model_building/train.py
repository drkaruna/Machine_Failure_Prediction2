import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Load local split files
Xtrain = pd.read_csv("week_2_mls/artifacts/Xtrain.csv")
Xtest = pd.read_csv("week_2_mls/artifacts/Xtest.csv")
ytrain = pd.read_csv("week_2_mls/artifacts/ytrain.csv")
ytest = pd.read_csv("week_2_mls/artifacts/ytest.csv")

numeric_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
categorical_features = ['Type']

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)
param_grid = {
    'xgbclassifier__n_estimators': [50],
    'xgbclassifier__max_depth': [3],
    'xgbclassifier__learning_rate': [0.1],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain.values.ravel())

# Save directly to the deployment directory so it gets bundled with app.py
os.makedirs("week_2_mls/deployment", exist_ok=True)
joblib.dump(grid_search.best_estimator_, "week_2_mls/deployment/best_machine_failure_model_v1.joblib")
print("Model trained and saved to deployment directory.")
