import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Point to your local CSV file
DATASET_PATH = "week_2_mls/data/machine-failure-prediction.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Preprocessing
df.drop(columns=['UDI'], inplace=True)
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

X = df.drop(columns=['Failure'])
y = df['Failure']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save artifacts directly inside the workspace for the next stage
os.makedirs("week_2_mls/artifacts", exist_ok=True)
Xtrain.to_csv("week_2_mls/artifacts/Xtrain.csv", index=False)
Xtest.to_csv("week_2_mls/artifacts/Xtest.csv", index=False)
ytrain.to_csv("week_2_mls/artifacts/ytrain.csv", index=False)
ytest.to_csv("week_2_mls/artifacts/ytest.csv", index=False)
print("Data preparation complete. Artifacts saved locally.")
