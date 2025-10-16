
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
import warnings
import statsmodels.api as sm  # required by pmdarima
from pmdarima import auto_arima

try:
    from logic.policy_commitments import CountryCommitment
except ImportError:
    CountryCommitment = None

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

@dataclass
class CommitmentForecast:
    future_years: np.ndarray
    bau_forecast: np.ndarray
    commitment_forecast: np.ndarray
    emissions_gap: np.ndarray
    bau_model_name: str
    pathway_type: str
    target_year: int
    target_emissions: float
    current_emissions: float

def _linear_pathway(start_year: int, start_emissions: float, target_year: int, target_emissions: float, future_years: np.ndarray) -> np.ndarray:
    """Linear reduction pathway from start to target."""
    slope = (target_emissions - start_emissions) / (target_year - start_year)
    return start_emissions + slope * (future_years - start_year)

def _exponential_pathway(start_year: int, start_emissions: float, target_year: int, target_emissions: float, future_years: np.ndarray) -> np.ndarray:
    """Exponential decay pathway to target."""
    years_to_target = target_year - start_year
    if target_emissions <= 0:
        target_emissions = 0.01  # Avoid log(0)
    k = -np.log(target_emissions / start_emissions) / years_to_target
    return start_emissions * np.exp(-k * (future_years - start_year))

def _scurve_pathway(start_year: int, start_emissions: float, target_year: int, target_emissions: float, future_years: np.ndarray) -> np.ndarray:
    """S-curve (logistic) pathway - slow start, rapid middle, slow end."""
    years_to_target = target_year - start_year
    midpoint = start_year + years_to_target / 2
    k = 8 / years_to_target  # Steepness parameter

    L = start_emissions - target_emissions  # Range
    x = future_years - midpoint
    sigmoid = L / (1 + np.exp(k * x))
    return target_emissions + sigmoid

def _polynomial_constrained(df: pd.DataFrame, milestones: Dict[int, float], future_years: np.ndarray, degree: int = 3) -> np.ndarray:
    """Fit polynomial through historical data and milestone constraints."""
    years = np.concatenate([df['year'].values, list(milestones.keys())])
    emissions = np.concatenate([df['emissions_gtco2'].values, list(milestones.values())])

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(years.reshape(-1, 1))
    y = emissions

    model = LinearRegression().fit(X, y)
    X_future = poly.transform(future_years.reshape(-1, 1))
    return np.maximum(0, model.predict(X_future))  # Ensure non-negative

def _policy_driven_pathway(
    df: pd.DataFrame,
    country_commitment: 'CountryCommitment',
    future_years: np.ndarray,
    bau_forecast: np.ndarray,
    target_year: int,
    target_emissions: float
) -> np.ndarray:
    """
    Policy-driven pathway based on specific country commitment actions.

    Uses actual policy reduction data to create a distinct stepped/staged
    reduction trajectory that differs from exponential decay.

    Approach:
    - Directly applies policy reduction fractions (weighted by sector share)
    - Creates staged reduction following policy milestones
    - Smooth interpolation between policy data points
    - Ensures reaching target by target_year

    Args:
        df: Historical emissions data
        country_commitment: CountryCommitment object with policy actions
        future_years: Array of future years to forecast
        bau_forecast: Business-as-usual forecast for future years
        target_year: Final year to reach target
        target_emissions: Target emissions at target_year

    Returns:
        Array of emissions following policy-driven pathway with distinct shape
    """
    start_year = int(df['year'].max())

    # Use baseline emissions as reference (not current)
    baseline_emissions = country_commitment.baseline_emissions_gtco2

    emissions_projection = []

    for i, year in enumerate(future_years):
        # Get weighted reduction fraction from all policy actions for this year
        reduction_fraction = country_commitment.calculate_annual_reduction_fraction(int(year))

        # Calculate emissions based on policy reduction from baseline
        # This creates a distinct shape following policy milestones
        policy_emissions = baseline_emissions * (1 - reduction_fraction)

        # Ensure we don't go below target
        policy_emissions = max(policy_emissions, target_emissions)

        # Ensure we don't exceed BAU (policies only reduce, not increase)
        policy_emissions = min(policy_emissions, bau_forecast[i])

        emissions_projection.append(policy_emissions)

    pathway = np.array(emissions_projection)

    # Final smoothing pass to ensure monotonic decrease
    # This preserves the staged shape but removes any slight increases
    for i in range(1, len(pathway)):
        if pathway[i] > pathway[i-1]:
            # Gentle decrease instead of increase
            pathway[i] = pathway[i-1] * 0.995

    # Ensure final value exactly hits target
    pathway[-1] = target_emissions

    return pathway

def forecast_with_commitment(
    df: pd.DataFrame,
    target_year: int,
    target_emissions: float,
    pathway_type: str = 'exponential',
    milestones: Dict[int, float] | None = None,
    bau_degree: int = 2,
    country_commitment: Optional['CountryCommitment'] = None,
) -> CommitmentForecast:
    """
    Forecast with commitment scenario.

    Args:
        df: Historical emissions data
        target_year: Year to reach target
        target_emissions: Target emissions (GtCO2)
        pathway_type: 'linear', 'exponential', 'scurve', 'milestones', or 'policy_driven'
        milestones: Optional dict of {year: emissions} intermediate targets
        bau_degree: Polynomial degree for BAU forecast
        country_commitment: Optional CountryCommitment for policy-driven pathway

    Returns:
        CommitmentForecast with BAU and commitment pathways
    """
    df = df.dropna().sort_values('year')
    last_year = int(df['year'].max())
    current_emissions = float(df.iloc[-1]['emissions_gtco2'])

    if target_year <= last_year:
        raise ValueError(f"Target year must be > {last_year}")

    horizon = target_year - last_year
    future_years = np.arange(last_year + 1, target_year + 1)

    # BAU forecast
    bau_result = forecast_best(df, horizon=horizon, degree=bau_degree)
    bau_forecast = bau_result.y_pred

    # Commitment pathway
    if pathway_type == 'policy_driven' and country_commitment is not None:
        commitment_forecast = _policy_driven_pathway(
            df, country_commitment, future_years, bau_forecast, target_year, target_emissions
        )
    elif pathway_type == 'milestones' and milestones:
        # Add target to milestones
        all_milestones = {**milestones, target_year: target_emissions}
        commitment_forecast = _polynomial_constrained(df, all_milestones, future_years, degree=3)
    elif pathway_type == 'linear':
        commitment_forecast = _linear_pathway(last_year, current_emissions, target_year, target_emissions, future_years)
    elif pathway_type == 'scurve':
        commitment_forecast = _scurve_pathway(last_year, current_emissions, target_year, target_emissions, future_years)
    else:  # exponential (default)
        commitment_forecast = _exponential_pathway(last_year, current_emissions, target_year, target_emissions, future_years)

    # Ensure non-negative
    commitment_forecast = np.maximum(0, commitment_forecast)

    # Calculate emissions gap
    emissions_gap = bau_forecast - commitment_forecast

    return CommitmentForecast(
        future_years=future_years,
        bau_forecast=bau_forecast,
        commitment_forecast=commitment_forecast,
        emissions_gap=emissions_gap,
        bau_model_name=bau_result.model_name,
        pathway_type=pathway_type,
        target_year=target_year,
        target_emissions=target_emissions,
        current_emissions=current_emissions,
    )
