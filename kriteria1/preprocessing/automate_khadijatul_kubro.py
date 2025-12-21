import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle outliers (BMI)
    Q1 = df["bmi"].quantile(0.25)
    Q3 = df["bmi"].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df["bmi"] >= Q1 - 1.5 * IQR) &
        (df["bmi"] <= Q3 + 1.5 * IQR)
    ]

    #Pisahkan fitur & target
    X = df.drop("charges", axis=1)
    y = df["charges"]

    #Tentukan kolom numerik & kategorikal
    num_cols = ["age", "bmi", "children"]
    cat_cols = ["sex", "smoker", "region"]

    # Encoding
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_processed = preprocess.fit_transform(X_train)
    X_test_processed = preprocess.transform(X_test)

    joblib.dump(preprocess, f"{output_path}/preprocess.pkl")
    pd.DataFrame(X_train_processed).to_csv(
        f"{output_path}/X_train.csv", index=False
    )
    pd.DataFrame(X_test_processed).to_csv(
        f"{output_path}/X_test.csv", index=False
    )
    y_train.to_csv(f"{output_path}/y_train.csv", index=False)
    y_test.to_csv(f"{output_path}/y_test.csv", index=False)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(
        BASE_DIR,
        "..",
        "healthcareinsurance_raw.csv"
    )

    output_path = os.path.join(
        BASE_DIR,
        "healthcareinsurance_preprocessing"
    )

    preprocess_data(input_path, output_path)