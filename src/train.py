import os
import sys
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib

# --- Cargar configuración ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Cargar dataset ---
print("📂 Cargando dataset...")
df = pd.read_csv(config["data"]["path"])
print(f"Shape original: {df.shape}")

# --- Preprocesamiento ---
print("🧹 Limpiando datos...")

# Eliminar columnas con más del 40% de nulos
threshold = 0.4 * len(df)
df = df.dropna(thresh=threshold, axis=1)

# Separar target
target = "SalePrice"
X = df.drop(columns=[target, "Id"], errors="ignore")
y = df[target]

# Manejar nulos numéricos con mediana
num_cols = X.select_dtypes(include=np.number).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Manejar nulos categóricos con moda
cat_cols = X.select_dtypes(include="object").columns
X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

# Codificación de variables categóricas
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

print(f"Shape preprocesado: {X.shape}")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"]
)

# --- Configurar MLflow ---
tracking_uri = "file://" + os.path.abspath("mlruns")
mlflow.set_tracking_uri(tracking_uri)

experiment_name = config["mlflow"]["experiment_name"]
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location="file://" + os.path.abspath("mlruns")
    )
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# --- Entrenamiento y tracking ---
print("🚀 Entrenando modelo...")
params = {
    "n_estimators": config["model"]["n_estimators"],
    "max_depth": config["model"]["max_depth"],
    "random_state": config["model"]["random_state"]
}

with mlflow.start_run(experiment_id=experiment_id) as run:
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📊 R2: {r2:.4f}")

    # Log params y métricas
    mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Log modelo
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

    # Guardar model.pkl para validación
    joblib.dump(model, "model.pkl")
    print(f"✅ Modelo guardado. Run ID: {run.info.run_id}")
