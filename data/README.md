# Datos

## Series de monitoreo del arroyo Mburicaó

`Mburicao_2021.csv` y `Mburicao_2025.csv` contienen las dos campañas de monitoreo del nivel de agua
y la precipitación del arroyo Mburicaó (Asunción, Paraguay), obtenidas de sensores instalados en el
arroyo.

**Columnas:**

| Columna | Descripción | Unidad |
|---|---|---|
| `Time` | Marca temporal de la observación | fecha-hora |
| `Nivel` | Nivel de agua (variable objetivo) | metros |
| `Precipitacion` | Precipitación acumulada del intervalo | milímetros |

**Notas de resolución.**
- `Mburicao_2021.csv`: resolución nativa de **10 minutos**.
- `Mburicao_2025.csv`: adquirido cada **5 minutos**, con el acumulado de lluvia de 10 minutos
  repetido en cada par de marcas. El script `src/01_eda.py` lo **regulariza a 10 minutos** (nivel =
  último valor del intervalo con interpolación temporal de huecos; precipitación = promedio del
  intervalo, equivalente al acumulado de 10 min), reproduciendo las 29 055 muestras usadas en el
  modelado.

## Predicciones de test

`predicciones_test.csv` contiene las predicciones de los **7 modelos** sobre los conjuntos de test de
ambos segmentos, **ya generadas** (el repositorio no reentrena modelos). Es la base de todas las
métricas y figuras del análisis.

| Columna | Descripción |
|---|---|
| `segmento` | Segmento de monitoreo (2021 o 2025) |
| `modelo` / `modelo_label` | Identificador y nombre legible del modelo |
| `origin_time` | Tiempo de origen del pronóstico |
| `pred_time` | Tiempo objetivo de la predicción |
| `lead_step` / `lead_minutes` | Horizonte de pronóstico (1/2/3 pasos = 10/20/30 min) |
| `actual` | Nivel observado (m) |
| `pred` | Nivel predicho (m) |
