import os
import sys
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# --- Cargar configuración ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Umbrales ---
MAX_RMSE = config["thresholds"]["max_rmse"]
MIN_R2 = config["thresholds"]["min_r2"]

# --- Cargar dataset ---
print("📂 Cargando dataset para validación...")
df = pd.read_csv(config["data"]["path"])

# --- Mismo preprocesamiento que train.py ---
threshold = 0.4 * len(df)
df = df.dropna(thresh=threshold, axis=1)

target = "SalePrice"
X = df.drop(columns=[target, "Id"], errors="ignore")
y = df[target]

num_cols = X.select_dtypes(include=np.number).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

cat_cols = X.select_dtypes(include="object").columns
X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"]
)

# --- Cargar modelo ---
model_path = os.path.abspath("model.pkl")
print(f"📦 Cargando modelo desde: {model_path}")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"❌ ERROR: No se encontró model.pkl en {model_path}")
    sys.exit(1)

# --- Validación ---
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"🔍 RMSE: {rmse:.2f} (máximo permitido: {MAX_RMSE})")
print(f"🔍 R2: {r2:.4f} (mínimo permitido: {MIN_R2})")

if rmse <= MAX_RMSE and r2 >= MIN_R2:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo NO cumple el umbral. Deteniendo pipeline.")
    sys.exit(1)
