import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv("Fraud.csv")


df = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

print("Dataset shape:", df.shape)
print("Fraud cases:", df["isFraud"].sum())

df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["receiver_is_new"] = (df["oldbalanceDest"] == 0).astype(int)
df["large_transaction"] = (df["amount"] > 100000).astype(int)

df = pd.get_dummies(df, columns=["type"], prefix="type")

df = df.drop(["newbalanceOrig", "newbalanceDest"], axis=1)

print("Features:", df.columns.tolist())


fraud = df[df["isFraud"] == 1]
non_fraud = df[df["isFraud"] == 0]


non_fraud_sample = non_fraud.sample(n=min(500000, len(non_fraud)), random_state=42)
df_balanced = pd.concat([fraud, non_fraud_sample], axis=0)

print("Balanced dataset shape:", df_balanced.shape)

X = df_balanced.drop("isFraud", axis=1)
y = df_balanced["isFraud"]


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
)


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, "fraud_model.sav")


joblib.dump(list(X.columns), "model_features.sav")

print("\nModel and Feature List Saved Successfully ")