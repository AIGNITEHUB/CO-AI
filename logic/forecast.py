
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
import warnings
import statsmodels.api as sm  # required by pmdarima
from pmdarima import auto_arima

@dataclass
class ForecastResult:
    future_years: np.ndarray
    y_pred: np.ndarray
    model_name: str
    diagnostics: Dict[str, float] | None = None

def fit_poly(df: pd.DataFrame, degree: int = 2):
    df = df.dropna().sort_values('year')
    X = df[['year']].values
    y = df['emissions_gtco2'].values
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, y)
    return model, poly

def predict_poly(model, poly, future_years: np.ndarray):
    Xp = poly.transform(future_years.reshape(-1,1))
    return model.predict(Xp)

def fit_auto_arima(y: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stepwise = auto_arima(y, seasonal=False, error_action="ignore", suppress_warnings=True, trace=False, max_p=5, max_q=5, max_order=None)
    return stepwise

def predict_arima(model, steps: int):
    fc = model.predict(n_periods=steps, return_conf_int=False)
    return np.asarray(fc)

def backtest_last_n(df: pd.DataFrame, n_test: int = 5, degree: int = 2) -> dict:
    df = df.dropna().sort_values('year').copy()
    if len(df) < n_test + 5:
        return {"poly_mape": float('nan'), "arima_mape": float('nan')}
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    poly_model, poly = fit_poly(train, degree=degree)
    y_hat_poly = predict_poly(poly_model, poly, test['year'].values)
    mape_poly = float(mape(test['emissions_gtco2'].values, y_hat_poly))
    y_train = pd.Series(train['emissions_gtco2'].values, index=train['year'].values)
    arima_model = fit_auto_arima(y_train)
    y_hat_arima = predict_arima(arima_model, n_test)
    mape_arima = float(mape(test['emissions_gtco2'].values, y_hat_arima))
    return {"poly_mape": mape_poly, "arima_mape": mape_arima}

def forecast_best(df: pd.DataFrame, horizon: int = 20, degree: int = 2) -> 'ForecastResult':
    df = df.dropna().sort_values('year')
    last_year = int(df['year'].max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon)
    scores = backtest_last_n(df, n_test=min(8, max(3, len(df)//6)), degree=degree)
    poly_model, poly = fit_poly(df, degree=degree)
    y_pred_poly = predict_poly(poly_model, poly, future_years)
    y_full = pd.Series(df['emissions_gtco2'].values, index=df['year'].values)
    arima_model = fit_auto_arima(y_full)
    y_pred_arima = predict_arima(arima_model, len(future_years))
    poly_mape = scores.get("poly_mape", float('inf'))
    arima_mape = scores.get("arima_mape", float('inf'))
    if np.isnan(poly_mape) or np.isnan(arima_mape):
        chosen = "ARIMA"
    else:
        chosen = "ARIMA" if arima_mape <= poly_mape else f"Poly(deg={degree})"
    y_pred = y_pred_arima if chosen.startswith("ARIMA") else y_pred_poly
    return ForecastResult(future_years=future_years, y_pred=y_pred, model_name=chosen, diagnostics={"poly_mape": poly_mape, "arima_mape": arima_mape})

def filter_years(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df[(df['year'] >= start) & (df['year'] <= end)].copy()
