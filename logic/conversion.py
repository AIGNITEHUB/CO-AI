from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

# Simple rule base for primary product by pathway
PRIMARY_PRODUCT = {
    'Sabatier (methanation)': 'CH4 (methane)',
    'RWGS + Fischer-Tropsch': 'C5+ hydrocarbons (synthetic diesel/jet range)',
    'Methanol synthesis': 'CH3OH (methanol)',
    'Electroreduction (formate route)': 'HCOO− / HCOOH (formate/formic acid)',
    'Electroreduction (CO route)': 'CO (syngas component)',
}

# Nominal base yields (arbitrary demo values: 0..1)
BASE_YIELD = {
    'Sabatier (methanation)': 0.72,
    'RWGS + Fischer-Tropsch': 0.55,
    'Methanol synthesis': 0.60,
    'Electroreduction (formate route)': 0.45,
    'Electroreduction (CO route)': 0.50,
}

# Catalyst family performance modifiers (arbitrary demo multipliers)
CATALYST_GAIN = {
    'Ni-based': 1.00,
    'Cu/ZnO/Al2O3': 1.05,
    'Fe/Co (FT)': 1.10,
    'Ag/Au (electro)': 0.95,
    'Cu (electro)': 1.08,
    'Other': 1.00,
}

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def predict_product_and_yield(
    pathway: str,
    catalyst_family: str,
    temperature_c: float,
    pressure_bar: float,
    h2_to_co2: float,
    reactor_type: str,
    grid_ci_gco2_per_kwh: float,
) -> Dict[str, object]:
    """Heuristic demo predictor. Returns product, yield_est (0..1), and notes.
    This is NOT scientific — for UI demo only.
    """
    product = PRIMARY_PRODUCT.get(pathway, 'Unknown')
    base = BASE_YIELD.get(pathway, 0.4)
    gain = CATALYST_GAIN.get(catalyst_family, 1.0)

    # Rough operating window “sweet spot” per pathway (center, slope)
    temp_center = {
        'Sabatier (methanation)': 320,
        'RWGS + Fischer-Tropsch': 380,
        'Methanol synthesis': 250,
        'Electroreduction (formate route)': 25,
        'Electroreduction (CO route)': 25,
    }.get(pathway, 300)

    temp_score = _sigmoid((temperature_c - temp_center) / 40.0)

    # Pressure helps thermal routes more than electro
    if 'Electro' in pathway:
        p_score = 0.5 + 0.1 * _sigmoid((pressure_bar - 1) / 5.0)
    else:
        p_score = 0.5 + 0.4 * _sigmoid((pressure_bar - 10) / 10.0)

    # Stoichiometry proximity (e.g., Sabatier ~ 4:1 H2:CO2, methanol ~ 3:1)
    h2_opt = {
        'Sabatier (methanation)': 4.0,
        'RWGS + Fischer-Tropsch': 2.0,  # RWGS ~1:1 then FT uses H2/CO
        'Methanol synthesis': 3.0,
        'Electroreduction (formate route)': 2.0,
        'Electroreduction (CO route)': 1.5,
    }.get(pathway, 2.0)
    h2_score = np.exp(-((h2_to_co2 - h2_opt) ** 2) / (2 * 0.8 ** 2))  # Gaussian

    # Reactor-type bias
    reactor_bias = {
        'Fixed-bed': 1.00,
        'Slurry': 0.98,
        'Electrochemical (flow cell)': 1.05,
        'Other': 1.00,
    }.get(reactor_type, 1.0)

    # Combine
    raw = base * gain * reactor_bias * (0.5 + 0.5 * temp_score) * (0.6 + 0.4 * p_score) * (0.5 + 0.5 * h2_score)
    # Clamp 0..0.95
    yield_est = float(np.clip(raw, 0.05, 0.95))

    # Toy net carbon intensity adjustment (electro routes penalized by dirty grid, rewarded by clean)
    net_ci_note = ""
    if 'Electro' in pathway:
        if grid_ci_gco2_per_kwh > 600:
            yield_est *= 0.9
            net_ci_note = "High grid CI reduces effective (net) efficiency in electro routes."
        elif grid_ci_gco2_per_kwh < 100:
            yield_est *= 1.03
            net_ci_note = "Low grid CI slightly improves effective (net) efficiency in electro routes."

    # Convert to % and make a band ±5%
    pct = round(100 * yield_est, 1)
    low = max(5.0, round(pct - 5.0, 1))
    high = min(95.0, round(pct + 5.0, 1))

    return {
        'predicted_product': product,
        'yield_percent_estimate': pct,
        'yield_percent_range': f"{low}–{high}%",
        'notes': net_ci_note or 'Heuristic demo output — replace with real model/data for production.',
    }
