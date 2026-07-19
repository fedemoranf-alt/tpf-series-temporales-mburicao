"""Analisis exploratorio y resumen del dataset del arroyo Mburicao.

Lee los dos segmentos de monitoreo (2021-2022 y 2025-2026), reporta un resumen
descriptivo y genera una figura de la serie de nivel con la precipitacion de 10
minutos para cada segmento.

Uso:
    python src/01_eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
FIGURAS = RAIZ / "results" / "figures"
METRICAS = RAIZ / "results" / "metricas"


def cargar_segmento(nombre: str, etiqueta: str) -> pd.DataFrame:
    """Carga un segmento y lo regulariza a una grilla temporal de 10 minutos.

    El segmento 2025 se adquiere cada 5 minutos con la precipitacion acumulada
    de 10 minutos repetida en cada par de marcas; por eso el nivel se toma como
    ultimo valor del intervalo (interpolando huecos) y la precipitacion como
    promedio del intervalo (que equivale al acumulado de 10 minutos). Asi ambos
    segmentos quedan en la misma resolucion usada para el modelado.
    """
    df = pd.read_csv(DATA / nombre, parse_dates=["Time"]).set_index("Time")
    reg = df.resample("10min").agg({"Nivel": "last", "Precipitacion": "mean"})
    reg["Nivel"] = reg["Nivel"].interpolate("time")
    reg["Precipitacion"] = reg["Precipitacion"].fillna(0.0)
    reg = reg.reset_index()
    reg["segmento"] = etiqueta
    return reg


def resumen(df: pd.DataFrame) -> dict:
    """Resumen descriptivo de un segmento."""
    return {
        "segmento": df["segmento"].iloc[0],
        "inicio": df["Time"].min(),
        "fin": df["Time"].max(),
        "muestras": len(df),
        "nivel_min_m": round(df["Nivel"].min(), 3),
        "nivel_max_m": round(df["Nivel"].max(), 3),
        "nivel_medio_m": round(df["Nivel"].mean(), 3),
        "lluvia_total_mm": round(df["Precipitacion"].sum(), 1),
        "faltantes_nivel": int(df["Nivel"].isna().sum()),
    }


def graficar_segmento(df: pd.DataFrame, ax_nivel, ax_lluvia, titulo: str) -> None:
    """Dibuja nivel (arriba) y precipitacion (abajo) para un segmento."""
    ax_nivel.plot(df["Time"], df["Nivel"], color="#1f4e79", linewidth=0.6)
    ax_nivel.set_ylabel("Nivel (m)")
    ax_nivel.set_title(titulo)
    ax_lluvia.bar(df["Time"], df["Precipitacion"], color="#2e86c1", width=0.01)
    ax_lluvia.set_ylabel("Lluvia (mm/10min)")
    ax_lluvia.invert_yaxis()


def main() -> None:
    seg2021 = cargar_segmento("Mburicao_2021.csv", "2021-2022")
    seg2025 = cargar_segmento("Mburicao_2025.csv", "2025-2026")

    tabla = pd.DataFrame([resumen(seg2021), resumen(seg2025)])
    METRICAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(METRICAS / "resumen_dataset.csv", index=False)
    print("Resumen del dataset:")
    print(tabla.to_string(index=False))

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), height_ratios=[3, 1],
                             gridspec_kw={"hspace": 0.05, "wspace": 0.2})
    graficar_segmento(seg2021, axes[0, 0], axes[1, 0], "Segmento 2021-2022")
    graficar_segmento(seg2025, axes[0, 1], axes[1, 1], "Segmento 2025-2026")
    fig.suptitle("Arroyo Mburicao: nivel de agua y precipitacion (10 min)", fontsize=13)
    fig.savefig(FIGURAS / "eda_series_segmentos.png", dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada en {FIGURAS / 'eda_series_segmentos.png'}")


if __name__ == "__main__":
    main()
