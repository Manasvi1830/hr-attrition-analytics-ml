import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def train_model():
    df = pd.read_csv("dataset/employee_data.csv")

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

if __name__ == "__main__":
    train_model()
