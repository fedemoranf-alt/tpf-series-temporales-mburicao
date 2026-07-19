"""Analisis de residuales de los modelos de pronostico.

Genera el diagnostico de residuales exigido por la rubrica a partir de las
predicciones ya guardadas (`data/predicciones_test.csv`), sin reentrenar ningun
modelo. Produce dos figuras y una tabla resumen:

- `results/figures/residuales_lstm_t30.png`: diagnostico del mejor modelo (LSTM)
  a 30 minutos (histograma, residual vs prediccion, autocorrelacion y serie temporal).
- `results/figures/residuales_boxplot_modelos_t30.png`: comparacion de la
  distribucion de residuales entre los 7 modelos.
- `results/metricas/residuales_resumen_t30.csv`: media, desvio y asimetria por modelo.

Convencion: residual = nivel observado - nivel predicho (positivo = subestimacion).

Uso:
    python src/03_analisis_residuales.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PRED = RAIZ / "data" / "predicciones_test.csv"
FIGURAS = RAIZ / "results" / "figures"
METRICAS = RAIZ / "results" / "metricas"

ORDEN_MODELOS = ["LSTM", "Ridge-ARX", "SARIMAX", "N-BEATS", "ARIMA",
                 "Gradient Boosting-ARX", "Persistencia"]


def acf(x: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Autocorrelacion muestral de una serie hasta `nlags` rezagos."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.sum(x ** 2)
    n = len(x)
    return np.array([np.sum(x[: n - k] * x[k:]) / var for k in range(nlags + 1)])


def diagnostico_lstm(df_t30: pd.DataFrame) -> None:
    """Figura 2x2 de diagnostico de residuales para LSTM a 30 minutos."""
    lstm = df_t30[df_t30["modelo_label"] == "LSTM"].copy()
    lstm["residual_cm"] = (lstm["actual"] - lstm["pred"]) * 100

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Histograma
    ax[0, 0].hist(lstm["residual_cm"], bins=80, color="#1f4e79", alpha=0.85)
    ax[0, 0].axvline(0, color="crimson", linestyle="--", linewidth=1)
    ax[0, 0].set_title("(a) Histograma de residuales")
    ax[0, 0].set_xlabel("Residual (cm)")
    ax[0, 0].set_ylabel("Frecuencia")

    # (b) Residual vs nivel predicho
    hb = ax[0, 1].hexbin(lstm["pred"], lstm["residual_cm"], gridsize=45,
                         cmap="Blues", mincnt=1)
    ax[0, 1].axhline(0, color="crimson", linestyle="--", linewidth=1)
    ax[0, 1].set_title("(b) Residual vs nivel predicho")
    ax[0, 1].set_xlabel("Nivel predicho (m)")
    ax[0, 1].set_ylabel("Residual (cm)")
    fig.colorbar(hb, ax=ax[0, 1], label="conteo")

    # (c) Autocorrelacion de residuales (segmento 2025 ordenado en el tiempo)
    s2025 = (lstm[lstm["segmento"] == 2025]
             .sort_values("pred_time"))
    valores_acf = acf(s2025["residual_cm"].to_numpy(), nlags=40)
    ax[1, 0].bar(range(len(valores_acf)), valores_acf, color="#2e86c1", width=0.8)
    ax[1, 0].axhline(0, color="black", linewidth=0.8)
    ax[1, 0].set_title("(c) Autocorrelacion de residuales (segmento 2025)")
    ax[1, 0].set_xlabel("Rezago (pasos de 10 min)")
    ax[1, 0].set_ylabel("ACF")

    # (d) Serie temporal de residuales (segmento 2025)
    ax[1, 1].plot(pd.to_datetime(s2025["pred_time"]), s2025["residual_cm"],
                  color="#1f4e79", linewidth=0.6)
    ax[1, 1].axhline(0, color="crimson", linestyle="--", linewidth=1)
    ax[1, 1].set_title("(d) Residuales en el tiempo (segmento 2025)")
    ax[1, 1].set_xlabel("Fecha")
    ax[1, 1].set_ylabel("Residual (cm)")
    ax[1, 1].tick_params(axis="x", rotation=30)

    fig.suptitle("Analisis de residuales - LSTM, horizonte 30 min", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURAS / "residuales_lstm_t30.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def boxplot_modelos(df_t30: pd.DataFrame) -> None:
    """Comparacion de la distribucion de residuales entre modelos a 30 min."""
    datos, etiquetas = [], []
    for modelo in ORDEN_MODELOS:
        sub = df_t30[df_t30["modelo_label"] == modelo]
        datos.append(((sub["actual"] - sub["pred"]) * 100).to_numpy())
        etiquetas.append(modelo)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(datos, labels=etiquetas, showfliers=False)
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1)
    ax.set_title("Distribucion de residuales por modelo (horizonte 30 min)")
    ax.set_ylabel("Residual = observado - predicho (cm)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURAS / "residuales_boxplot_modelos_t30.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def resumen_residuales(df_t30: pd.DataFrame) -> None:
    """Guarda media, desvio, min, max y asimetria de residuales por modelo."""
    filas = []
    for modelo in ORDEN_MODELOS:
        r = (df_t30[df_t30["modelo_label"] == modelo]
             .eval("(actual - pred) * 100"))
        filas.append({
            "modelo": modelo,
            "media_cm": round(r.mean(), 3),
            "desvio_cm": round(r.std(), 3),
            "min_cm": round(r.min(), 2),
            "max_cm": round(r.max(), 2),
            "asimetria": round(r.skew(), 3),
        })
    pd.DataFrame(filas).to_csv(METRICAS / "residuales_resumen_t30.csv", index=False)


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    METRICAS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED)
    df_t30 = df[df["lead_minutes"] == 30].copy()

    diagnostico_lstm(df_t30)
    boxplot_modelos(df_t30)
    resumen_residuales(df_t30)
    print("Figuras de residuales y resumen generados en results/.")


if __name__ == "__main__":
    main()
