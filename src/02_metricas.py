"""Calculo reproducible de las metricas de comparacion de modelos.

Lee las predicciones de test ya generadas (`data/predicciones_test.csv`) y calcula
las metricas puntuales (MAE, RMSE, MAPE, sMAPE, NSE, Bias) en tres vistas:
global (todos los horizontes), a 30 minutos, y por horizonte de pronostico.
Los resultados se guardan en `results/metricas/`.

Uso:
    python src/02_metricas.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from metricas import tabla_por_grupo

RAIZ = Path(__file__).resolve().parents[1]
PRED = RAIZ / "data" / "predicciones_test.csv"
METRICAS = RAIZ / "results" / "metricas"

ORDEN_MODELOS = ["LSTM", "Ridge-ARX", "SARIMAX", "N-BEATS", "ARIMA",
                 "Gradient Boosting-ARX", "Persistencia"]


def ordenar(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena las filas por RMSE ascendente (mejor modelo primero)."""
    return df.sort_values("RMSE_cm").reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(PRED)
    METRICAS.mkdir(parents=True, exist_ok=True)

    # Vista global: se agrupan todas las predicciones de ambos segmentos y horizontes.
    global_ = ordenar(tabla_por_grupo(df, ["modelo_label"]))
    global_.to_csv(METRICAS / "verif_metricas_globales.csv", index=False)

    # Vista a 30 minutos (horizonte operativo principal).
    t30 = ordenar(tabla_por_grupo(df[df["lead_minutes"] == 30], ["modelo_label"]))
    t30.to_csv(METRICAS / "verif_metricas_t30.csv", index=False)

    # Vista por horizonte de pronostico.
    por_lead = tabla_por_grupo(df, ["modelo_label", "lead_minutes"])
    por_lead = por_lead.sort_values(["lead_minutes", "RMSE_cm"]).reset_index(drop=True)
    por_lead.to_csv(METRICAS / "verif_metricas_por_horizonte.csv", index=False)

    cols = ["modelo_label", "MAE_cm", "RMSE_cm", "MAPE_pct", "sMAPE_pct", "NSE", "Bias_cm"]
    print("=== Metricas globales (todos los horizontes) ===")
    print(global_[cols].to_string(index=False))
    print("\n=== Metricas a 30 minutos ===")
    print(t30[cols].to_string(index=False))


if __name__ == "__main__":
    main()
