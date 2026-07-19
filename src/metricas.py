"""Funciones de metricas de evaluacion para el pronostico de nivel de agua.

Todas las metricas trabajan con arreglos de valores observados (`y`) y predichos
(`yhat`) expresados en metros. Los errores porcentuales (MAPE/sMAPE) se calculan
ignorando los pocos casos con nivel observado igual a cero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Error absoluto medio."""
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Raiz del error cuadratico medio."""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Sesgo (error medio con signo). Negativo = subestimacion."""
    return float(np.mean(yhat - y))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Error porcentual absoluto medio (%), ignorando y == 0."""
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Error porcentual absoluto medio simetrico (%)."""
    denom = np.abs(y) + np.abs(yhat)
    mask = denom != 0
    return float(np.mean(2 * np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100)


def nse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (identica a R2 en este contexto)."""
    denom = np.sum((y - np.mean(y)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((y - yhat) ** 2) / denom)


def calcular_metricas(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """Devuelve el conjunto completo de metricas puntuales para un par (y, yhat)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return {
        "n": int(len(y)),
        "MAE_m": mae(y, yhat),
        "RMSE_m": rmse(y, yhat),
        "MAE_cm": mae(y, yhat) * 100,
        "RMSE_cm": rmse(y, yhat) * 100,
        "MAPE_pct": mape(y, yhat),
        "sMAPE_pct": smape(y, yhat),
        "NSE": nse(y, yhat),
        "Bias_cm": bias(y, yhat) * 100,
    }


def tabla_por_grupo(df: pd.DataFrame, grupo: list[str]) -> pd.DataFrame:
    """Calcula las metricas puntuales agrupando por las columnas indicadas.

    Se espera que `df` tenga las columnas `actual` y `pred`.
    """
    filas = []
    for clave, sub in df.groupby(grupo):
        fila = dict(zip(grupo, clave if isinstance(clave, tuple) else (clave,)))
        fila.update(calcular_metricas(sub["actual"].to_numpy(), sub["pred"].to_numpy()))
        filas.append(fila)
    return pd.DataFrame(filas)
