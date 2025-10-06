
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

PRIMARY_PRODUCT = {
    'Sabatier (methanation)': 'CH4 (methane)',
    'RWGS + Fischer-Tropsch': 'C5+ hydrocarbons (diesel/jet range)',
    'Methanol synthesis': 'CH3OH (methanol)',
    'Electroreduction (formate route)': 'HCOO− / HCOOH (formate/formic acid)',
    'Electroreduction (CO route)': 'CO (syngas component)',
}

BASE_YIELD = {
    'Sabatier (methanation)': 0.72,
    'RWGS + Fischer-Tropsch': 0.55,
    'Methanol synthesis': 0.60,
    'Electroreduction (formate route)': 0.45,
    'Electroreduction (CO route)': 0.50,
}

CATALYST_GAIN = {
    'Ni-based': 1.00,
    'Cu/ZnO/Al2O3': 1.05,
    'Fe/Co (FT)': 1.10,
    'Ag/Au (electro)': 0.95,
    'Cu (electro)': 1.08,
    'Other': 1.00,
}

H2_PER_CO2 = {
    'Sabatier (methanation)': 4.0,
    'Methanol synthesis': 3.0,
    'Electroreduction (formate route)': 2.0,
    'Electroreduction (CO route)': 1.0,
    'RWGS + Fischer-Tropsch': 2.2,
}

OPT_WINDOWS = {
    'Sabatier (methanation)': (320, 60, 10, 8, 4.0, 0.8),
    'RWGS + Fischer-Tropsch': (380, 80, 20, 12, 2.0, 0.9),
    'Methanol synthesis': (250, 40, 50, 20, 3.0, 0.8),
    'Electroreduction (formate route)': (25, 15, 1, 0.8, 2.0, 0.8),
    'Electroreduction (CO route)': (25, 15, 1, 0.8, 1.5, 0.6),
}

def _gauss(x: float, mu: float, sigma: float) -> float:
    return float(np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)))

def _bounded(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _mix_selectivity(pathway: str, temp_score: float, h2_score: float) -> Dict[str, float]:
    """Return a simple multi-product slate as selectivities summing to 1.0 (demo)."""
    if pathway == 'Sabatier (methanation)':
        ch4 = 0.75 * temp_score + 0.10 * h2_score + 0.10
        co = 0.10 * (1 - temp_score)
        ch3oh = 0.03 * h2_score
        c5p = 0.02
        others = 1.0 - (ch4 + co + ch3oh + c5p)
        return {'CH4': _bounded(ch4,0,1), 'CO': _bounded(co,0,1), 'CH3OH': _bounded(ch3oh,0,1), 'C5+': _bounded(c5p,0,1), 'others': _bounded(others,0,1)}
    if pathway == 'RWGS + Fischer-Tropsch':
        c5p = 0.55 * temp_score + 0.10 * h2_score + 0.10
        co = 0.08 * (1 - temp_score)
        ch4 = 0.06 * h2_score
        ch3oh = 0.04
        others = 1.0 - (c5p + co + ch4 + ch3oh)
        return {'C5+': _bounded(c5p,0,1), 'CO': _bounded(co,0,1), 'CH4': _bounded(ch4,0,1), 'CH3OH': _bounded(ch3oh,0,1), 'others': _bounded(others,0,1)}
    if pathway == 'Methanol synthesis':
        ch3oh = 0.70 * temp_score + 0.10 * h2_score + 0.10
        co = 0.06 * (1 - temp_score)
        ch4 = 0.04 * h2_score
        c5p = 0.02
        others = 1.0 - (ch3oh + co + ch4 + c5p)
        return {'CH3OH': _bounded(ch3oh,0,1), 'CO': _bounded(co,0,1), 'CH4': _bounded(ch4,0,1), 'C5+': _bounded(c5p,0,1), 'others': _bounded(others,0,1)}
    if pathway == 'Electroreduction (formate route)':
        hcoo = 0.65 * temp_score + 0.15 * h2_score + 0.05
        co = 0.08 * (1 - temp_score)
        ch3oh = 0.05
        ch4 = 0.02
        others = 1.0 - (hcoo + co + ch3oh + ch4)
        return {'HCOO/HCOOH': _bounded(hcoo,0,1), 'CO': _bounded(co,0,1), 'CH3OH': _bounded(ch3oh,0,1), 'CH4': _bounded(ch4,0,1), 'others': _bounded(others,0,1)}
    if pathway == 'Electroreduction (CO route)':
        co = 0.70 * temp_score + 0.10 * h2_score + 0.05
        hcoo = 0.06 * (1 - temp_score)
        ch3oh = 0.05
        ch4 = 0.02
        others = 1.0 - (co + hcoo + ch3oh + ch4)
        return {'CO': _bounded(co,0,1), 'HCOO/HCOOH': _bounded(hcoo,0,1), 'CH3OH': _bounded(ch3oh,0,1), 'CH4': _bounded(ch4,0,1), 'others': _bounded(others,0,1)}
    return {'others': 1.0}

def predict_conversion_advanced(
    pathway: str,
    catalyst_family: str,
    temperature_c: float,
    pressure_bar: float,
    h2_to_co2: float,
    reactor_type: str,
    grid_ci_gco2_per_kwh: float,
) -> Dict[str, object]:
    base = BASE_YIELD.get(pathway, 0.4)
    gain = CATALYST_GAIN.get(catalyst_family, 1.0)
    t0, tw, p0, pw, h0, hw = OPT_WINDOWS.get(pathway, (300,60,10,8,2.0,1.0))

    temp_score = _gauss(temperature_c, t0, max(10, tw/2))
    press_score = _gauss(pressure_bar, p0, max(2, pw/2))
    h2_score = _gauss(h2_to_co2, h0, max(0.4, hw/2))

    reactor_bias = {
        'Fixed-bed': 1.00,
        'Slurry': 0.98,
        'Electrochemical (flow cell)': 1.05,
        'Other': 1.00,
    }.get(reactor_type, 1.0)

    raw = base * gain * reactor_bias * (0.5 + 0.5 * temp_score) * (0.6 + 0.4 * press_score) * (0.5 + 0.5 * h2_score)
    yield_est = float(np.clip(raw, 0.05, 0.95))

    notes: List[str] = []
    if 'Electro' in pathway:
        if grid_ci_gco2_per_kwh > 600:
            yield_est *= 0.9
            notes.append("High grid carbon intensity penalizes effective (net) efficiency for electro routes.")
        elif grid_ci_gco2_per_kwh < 100:
            yield_est *= 1.03
            notes.append("Low grid carbon intensity slightly improves effective (net) efficiency for electro routes.")

    slate = _mix_selectivity(pathway, temp_score, h2_score)
    ssum = sum(slate.values())
    if ssum > 0:
        slate = {k: v/ssum for k,v in slate.items()}

    h2_stoich = H2_PER_CO2.get(pathway, 2.0)
    h2_limit_factor = float(min(1.0, (h2_to_co2 + 1e-9) / h2_stoich))
    yield_est *= (0.8 + 0.2 * h2_limit_factor)
    yield_est = float(np.clip(yield_est, 0.03, 0.95))

    c_frac = 1.0 - slate.get('others', 0.0)
    carbon_util = float(np.clip(yield_est * (0.8 + 0.2 * c_frac), 0.02, 0.98))

    pct = round(100 * yield_est, 1)
    low = max(3.0, round(pct - 7.0, 1))
    high = min(95.0, round(pct + 7.0, 1))

    suggestions: List[str] = []
    if temp_score < 0.5:
        suggestions.append(f"Increase temperature toward ~{t0}°C (±{int(tw/2)}°C).")
    if press_score < 0.5 and 'Electro' not in pathway:
        suggestions.append(f"Increase pressure toward ~{p0} bar (±{int(pw/2)} bar).")
    if h2_score < 0.5:
        suggestions.append(f"Adjust H₂:CO₂ toward ~{h0}:1 (±{hw/2:.1f}).")
    if 'Electro' in pathway and grid_ci_gco2_per_kwh > 400:
        suggestions.append("Use cleaner electricity (lower gCO₂/kWh) to improve net efficiency.")

    return {
        'primary_product': PRIMARY_PRODUCT.get(pathway, 'Unknown'),
        'yield_percent_estimate': pct,
        'yield_percent_range': f"{low}–{high}%",
        'product_slate_selectivity': {k: round(100*v, 1) for k,v in slate.items()},
        'estimated_H2_per_CO2_mol': round(h2_stoich, 2),
        'carbon_utilization_proxy_percent': round(100 * carbon_util, 1),
        'operating_scores': {
            'temperature_score': round(float(temp_score), 3),
            'pressure_score': round(float(press_score), 3),
            'h2_ratio_score': round(float(h2_score), 3),
        },
        'suggestions': suggestions or ["Operating point sits near nominal window."],
        'notes': notes or ["Heuristic demo output — replace with real kinetics/thermo + trained model for production."],
    }

# ============================================================================
# Catalyst-Pathway Comparison & Optimization Functions
# ============================================================================

def is_valid_combination(pathway: str, catalyst: str) -> bool:
    """
    Check if pathway-catalyst combination is valid.

    Rules:
    - Thermal catalysts (Ni-based, Cu/ZnO/Al2O3, Fe/Co) → Only thermal pathways
    - Electro catalysts (Cu electro, Ag/Au electro) → Only electro pathways
    """
    thermal_catalysts = {'Ni-based', 'Cu/ZnO/Al2O3', 'Fe/Co (FT)', 'Other'}
    electro_catalysts = {'Cu (electro)', 'Ag/Au (electro)'}

    is_electro_pathway = 'Electro' in pathway
    is_thermal_pathway = not is_electro_pathway

    is_thermal_catalyst = catalyst in thermal_catalysts
    is_electro_catalyst = catalyst in electro_catalysts

    # Valid if both thermal OR both electro
    if is_thermal_pathway and is_thermal_catalyst:
        return True
    if is_electro_pathway and is_electro_catalyst:
        return True

    # Handle "Other" catalyst - works with all
    if catalyst == 'Other':
        return True

    return False

def generate_catalyst_pathway_matrix() -> pd.DataFrame:
    """
    Generate a comprehensive matrix of yield estimates for all pathway × catalyst combinations.

    Returns:
        DataFrame with pathways as rows, catalysts as columns, yields as values
    """
    pathways = list(PRIMARY_PRODUCT.keys())
    catalysts = list(CATALYST_GAIN.keys())

    # Initialize matrix
    matrix_data = []

    for pathway in pathways:
        row = {'Pathway': pathway}

        # Get optimal conditions for this pathway
        t0, tw, p0, pw, h0, hw = OPT_WINDOWS.get(pathway, (300, 60, 10, 8, 2.0, 1.0))

        for catalyst in catalysts:
            if not is_valid_combination(pathway, catalyst):
                row[catalyst] = None  # N/A
            else:
                # Calculate yield at optimal conditions
                try:
                    result = predict_conversion_advanced(
                        pathway=pathway,
                        catalyst_family=catalyst,
                        temperature_c=float(t0),
                        pressure_bar=float(p0),
                        h2_to_co2=float(h0),
                        reactor_type='Electrochemical (flow cell)' if 'Electro' in pathway else 'Fixed-bed',
                        grid_ci_gco2_per_kwh=50.0 if 'Electro' in pathway else 0.0,  # Low CI for electro
                    )
                    row[catalyst] = result['yield_percent_estimate']
                except:
                    row[catalyst] = None

        matrix_data.append(row)

    df = pd.DataFrame(matrix_data)

    # Reorder columns: Pathway first, then catalysts
    cols = ['Pathway'] + catalysts
    df = df[cols]

    return df

def get_optimal_conditions_table() -> pd.DataFrame:
    """
    Generate a reference table of optimal operating conditions for each pathway.

    Returns:
        DataFrame with pathway, optimal T/P/H2:CO2, best catalyst, max yield, and product
    """
    pathways = list(PRIMARY_PRODUCT.keys())

    table_data = []

    for pathway in pathways:
        # Get optimal operating window
        t0, tw, p0, pw, h0, hw = OPT_WINDOWS.get(pathway, (300, 60, 10, 8, 2.0, 1.0))

        # Find best catalyst for this pathway
        best_catalyst = None
        max_yield = 0.0

        for catalyst, gain in CATALYST_GAIN.items():
            if not is_valid_combination(pathway, catalyst):
                continue

            try:
                result = predict_conversion_advanced(
                    pathway=pathway,
                    catalyst_family=catalyst,
                    temperature_c=float(t0),
                    pressure_bar=float(p0),
                    h2_to_co2=float(h0),
                    reactor_type='Electrochemical (flow cell)' if 'Electro' in pathway else 'Fixed-bed',
                    grid_ci_gco2_per_kwh=50.0 if 'Electro' in pathway else 0.0,
                )

                if result['yield_percent_estimate'] > max_yield:
                    max_yield = result['yield_percent_estimate']
                    best_catalyst = catalyst
            except:
                continue

        # Format temperature range
        t_range = f"{int(t0)}±{int(tw/2)}"

        # Format pressure range
        p_range = f"{int(p0)}±{int(pw/2)}"

        # Format H2:CO2 range
        h_range = f"{h0:.1f}±{hw/2:.1f}"

        table_data.append({
            'Pathway': pathway,
            'Temperature (°C)': t_range,
            'Pressure (bar)': p_range,
            'H₂:CO₂ ratio': h_range,
            'Best Catalyst': best_catalyst or 'N/A',
            'Max Yield (%)': round(max_yield, 1),
            'Primary Product': PRIMARY_PRODUCT.get(pathway, 'Unknown')
        })

    df = pd.DataFrame(table_data)

    return df
