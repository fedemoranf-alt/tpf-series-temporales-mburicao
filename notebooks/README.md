# Notebooks

Notebooks originales del proyecto **IBERAMIA** correspondientes al análisis exploratorio y al
entrenamiento de los modelos usados en este trabajo. Se incluyen para que el proyecto sea
**autocontenido y trazable**: documentan cómo se prepararon los datos y cómo se entrenó cada modelo
cuyas predicciones se analizan en `../data/predicciones_test.csv`.

| Notebook | Rol | Modelos / salida |
|---|---|---|
| `01_eda_datos_27_04.ipynb` | Análisis exploratorio de datos | Calidad de datos, series, distribuciones |
| `02_feature_ing.ipynb` | Ingeniería de features | Features autorregresivos y exógenos (ARX) |
| `15_nbeats_vs_baselines_final.ipynb` | Entrenamiento | **Ridge-ARX** y **N-BEATS** (versiones finales) |
| `17_arima_sarimax_baselines.ipynb` | Entrenamiento | **ARIMA** y **SARIMAX** (selección de órdenes por segmento) |
| `22_lstm.ipynb` | Entrenamiento | **LSTM** (grid search + entrenamiento) |
| `23_gradient_boosting_arx.ipynb` | Entrenamiento | **Gradient Boosting-ARX** |
| `24_comparaciones_Iberamia.ipynb` | Consolidación | Reúne las predicciones de los 7 modelos y calcula métricas/figuras comparativas |

> El modelo **Persistencia** no requiere entrenamiento (repite el último nivel observado).

## Cómo interpretar estos notebooks (importante)

Estos notebooks son un **registro documentado del EDA y del entrenamiento**, con sus **salidas ya
ejecutadas y guardadas**. **No están pensados para re-ejecutarse tal cual** dentro de este repositorio.
Provienen del pipeline original de la tesis/IBERAMIA, que se corrió en **varios entornos distintos**
(dos equipos Windows y un cluster Linux con GPU). Concretamente, no son ejecutables aquí porque:

1. **Rutas absolutas de otros entornos.** Contienen rutas hardcodeadas como
   `G:\...\Tesis\Proyectos\Mburicao_Iberamia`, `D:\Google Drive\...` y `/data/students/federico.moran`
   (cluster). Su función `resolve_project_dir()` apunta al proyecto **IBERAMIA original**, no a este repo.
2. **Archivos de entrada no incluidos.** Requieren datos crudos, CSVs procesados intermedios y archivos
   de *features* que no se incluyen aquí por tamaño; además, `24_comparaciones_Iberamia.ipynb` consume las
   predicciones por modelo generadas por `15/17/22/23` en el árbol `reports/` del proyecto original.
3. **Dependencias pesadas y GPU.** Necesitan `torch`, `darts`, `statsmodels`, `scikit-learn`, `scipy` y
   `seaborn` (no incluidos en `../requirements.txt`); N-BEATS y LSTM realistamente requieren **GPU**.

## Dónde está la reproducibilidad real de este TPF

La reproducibilidad **de los resultados** (todas las métricas, tablas y figuras del reporte) **sí está
garantizada y es autocontenida**: se calcula en la carpeta [`../src/`](../src) a partir de las
predicciones ya generadas (`../data/predicciones_test.csv`), sin reentrenar ningún modelo. Ver la sección
*Reproducibilidad* del [README principal](../README.md). Los notebooks de esta carpeta cumplen un rol
**complementario y de trazabilidad**: muestran *cómo* se obtuvieron esas predicciones.
