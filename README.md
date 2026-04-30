# MLOps Pipeline — House Prices

Pipeline reproducible de Machine Learning con tracking MLflow y CI/CD con GitHub Actions.

## Dataset
House Prices (Kaggle) — Regresión para predecir precios de vivienda.

## Estructura
- `src/train.py` — Entrenamiento y registro con MLflow
- `src/validate.py` — Validación de métricas
- `config.yaml` — Hiperparámetros y rutas
- `Makefile` — Comandos automatizados
- `.github/workflows/ml.yml` — Pipeline CI/CD

## Comandos
```bash
make install   # Instala dependencias
make train     # Entrena el modelo
make test      # Valida el modelo
make lint      # Revisa calidad del código
```

## Métricas objetivo
- RMSE < 50,000
- R2 > 0.70
