# MLOps Pipeline — House Prices

Pipeline reproducible de Machine Learning con tracking MLflow y CI/CD con GitHub Actions.

## Dataset
House Prices (Kaggle) — Regresión para predecir precios de vivienda en dólares.
Fuente: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Estructura
- `src/train.py` — Preprocesamiento, entrenamiento y registro con MLflow
- `src/validate.py` — Validación de métricas contra umbrales definidos
- `config.yaml` — Hiperparámetros y rutas centralizadas
- `Makefile` — Comandos automatizados
- `.github/workflows/ml.yml` — Pipeline CI/CD completo

## Comandos
make install   # Instala dependencias
make train     # Entrena el modelo
make test      # Valida el modelo
make lint      # Revisa calidad del código

## Métricas objetivo
- RMSE < 35,000 (error promedio en dólares)
- R2 > 0.80 (el modelo explica al menos el 80% de la variación en precios)

## CI/CD
Cada push a main ejecuta automáticamente: instalación → lint → entrenamiento → validación → artefactos
