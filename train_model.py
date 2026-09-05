import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load dataset
df = pd.read_csv('data/train.csv')
df = df.drop(columns=["Id"])
df.info()

# Kolom kategorikal yang NaN artinya "fitur ini memang tidak ada" (bukan data hilang)
none_means_no_features = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
]
for col in none_means_no_features:
    df[col] = df[col].fillna("None")

# Kolom numerik yang NaN-nya juga berarti "tidak ada fitur itu" -> isi 0
# (MasVnrArea NaN = tidak ada veneer, GarageYrBlt NaN = tidak ada garasi)
zero_means_no_feature = ["MasVnrArea", "GarageYrBlt"]
for col in zero_means_no_feature:
    df[col] = df[col].fillna(0)

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Numeric_transformer tetap pakai median untuk sisa NaN "random"
# (contoh: LotFrontage — NaN di sini memang missing beneran, bukan "tidak ada")
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# categorical_transformer tetap pakai most_frequent, tapi ini sekarang
# HANYA berlaku untuk kolom yang NaN-nya memang missing acak (contoh: Electrical,
# MSZoning) — karena kolom "None"-meaning sudah di-fillna manual di atas duluan.
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Training beberapa model
model_candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

models = {}
metrics = {}

for name, model in model_candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X_train, y_train, cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    models[name] = pipe
    metrics[name] = {
        "rmse": rmse, "mae": mae, "r2": r2,
        "y_test": y_test.values, "y_pred": y_pred,
        "train_sizes": train_sizes,
        "train_scores_mean": train_scores.mean(axis=1),
        "test_scores_mean": test_scores.mean(axis=1),
    }
    print(f"{name} -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")

    # Simpan Pickle + sample data
with open("models/models.pkl", "wb") as f:
    pickle.dump(models, f)
with open("models/metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

df.to_csv("data/house_prices_sample.csv", index=False)
print("Selesai! models.pkl dan metrics.pkl tersimpan di folder models/")