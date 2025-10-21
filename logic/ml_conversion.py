"""
ML-based CO2 Conversion Predictor
Wrapper for XGBoost models to use in Streamlit app

Add this file as: logic/ml_conversion.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
import os
from typing import Dict, Optional

class MLConversionPredictor:
    """ML-based predictor for CO2 conversion using XGBoost"""

    def __init__(self, models_dir: str = "models"):
        """
        Initialize ML predictor

        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_cols = []
        self.is_loaded = False

    def load_models(self) -> bool:
        """
        Load all models and preprocessing objects

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load XGBoost models
            for target in ['X_CO2', 'Y_MeOH']:
                model_path = os.path.join(self.models_dir, f'xgboost_{target}_conversion.json')
                if os.path.exists(model_path):
                    model = xgb.XGBRegressor()
                    model.load_model(model_path)
                    self.models[target] = model

            # Load scalers
            scaler_path = os.path.join(self.models_dir, 'conversion_scalers.pkl')
            if os.path.exists(scaler_path):
                self.scalers = joblib.load(scaler_path)

            # Load encoders
            encoder_path = os.path.join(self.models_dir, 'conversion_encoders.pkl')
            if os.path.exists(encoder_path):
                self.encoders = joblib.load(encoder_path)

            # Load feature columns
            features_path = os.path.join(self.models_dir, 'feature_columns.json')
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_cols = json.load(f)['features']

            self.is_loaded = len(self.models) > 0 and len(self.encoders) > 0

            return self.is_loaded

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def can_predict(self, pathway: str, catalyst: str) -> bool:
        """
        Check if ML prediction is available for given pathway/catalyst

        Args:
            pathway: Conversion pathway
            catalyst: Catalyst family

        Returns:
            True if ML prediction available
        """
        if not self.is_loaded:
            return False

        # ML models work best for methanol synthesis with real catalysts
        if pathway == 'Methanol synthesis':
            return True

        # Can handle other pathways but with lower confidence
        return True

    def prepare_features(
        self,
        pathway: str,
        catalyst_family: str,
        temperature_c: float,
        pressure_bar: float,
        h2_to_co2: float,
        reactor_type: str
    ) -> pd.DataFrame:
        """
        Prepare features for prediction

        Returns:
            DataFrame with engineered features
        """
        # Create input dataframe
        data = pd.DataFrame({
            'pathway': [pathway],
            'catalyst_family': [catalyst_family],
            'reactor_type': [reactor_type],
            'temperature_c': [temperature_c],
            'pressure_bar': [pressure_bar],
            'h2_co2_ratio': [h2_to_co2]
        })

        # Calculate derived features
        data['1000_T_K'] = 1000 / (data['temperature_c'] + 273.15)
        data['temp_pressure'] = data['temperature_c'] * data['pressure_bar']
        data['ln_pressure'] = np.log(data['pressure_bar'] + 1)

        # Pathway flags
        data['is_thermal'] = 0 if 'Electro' in pathway else 1
        data['is_methanation'] = 1 if 'methanation' in pathway else 0
        data['is_methanol'] = 1 if 'Methanol' in pathway else 0

        # Encode categorical features
        try:
            data['pathway_encoded'] = self.encoders['pathway'].transform(data['pathway'])
            data['catalyst_encoded'] = self.encoders['catalyst'].transform(data['catalyst_family'])
            data['reactor_encoded'] = self.encoders['reactor'].transform(data['reactor_type'])
        except ValueError as e:
            # Unknown category - use default encoding
            print(f"Warning: Unknown category - {e}")
            data['pathway_encoded'] = 0
            data['catalyst_encoded'] = 0
            data['reactor_encoded'] = 0

        # Select feature columns
        X = data[self.feature_cols]

        return X

    def predict(
        self,
        pathway: str,
        catalyst_family: str,
        temperature_c: float,
        pressure_bar: float,
        h2_to_co2: float,
        reactor_type: str
    ) -> Dict[str, float]:
        """
        Predict conversion and yield using ML models

        Returns:
            Dict with predictions and confidence scores
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Prepare features
        X = self.prepare_features(
            pathway, catalyst_family, temperature_c,
            pressure_bar, h2_to_co2, reactor_type
        )

        predictions = {}

        # Predict X_CO2
        if 'X_CO2' in self.models:
            X_scaled = self.scalers['X_CO2'].transform(X)
            x_co2_pred = self.models['X_CO2'].predict(X_scaled)[0]
            predictions['X_CO2'] = np.clip(x_co2_pred, 0.0, 100.0)

        # Predict Y_MeOH
        if 'Y_MeOH' in self.models:
            X_scaled = self.scalers['Y_MeOH'].transform(X)
            y_meoh_pred = self.models['Y_MeOH'].predict(X_scaled)[0]
            predictions['Y_MeOH'] = np.clip(y_meoh_pred, 0.0, 100.0)

        # Calculate selectivity (if both available)
        if 'X_CO2' in predictions and 'Y_MeOH' in predictions:
            if predictions['X_CO2'] > 0.01:
                predictions['S_MeOH'] = (predictions['Y_MeOH'] / predictions['X_CO2']) * 100
                predictions['S_MeOH'] = np.clip(predictions['S_MeOH'], 0.0, 100.0)
            else:
                predictions['S_MeOH'] = 0.0

        # Confidence score (based on distance from training range)
        confidence = self._calculate_confidence(temperature_c, pressure_bar, h2_to_co2)
        predictions['confidence'] = confidence

        return predictions

    def _calculate_confidence(
        self,
        temperature_c: float,
        pressure_bar: float,
        h2_to_co2: float
    ) -> float:
        """
        Calculate confidence score based on distance from training data

        Returns:
            Confidence score (0-1)
        """
        # Training ranges (from real data)
        temp_range = (170, 240)
        pressure_range = (1, 50)
        h2_range = (1.0, 4.0)

        # Calculate normalized distances
        temp_dist = 0.0
        if temperature_c < temp_range[0]:
            temp_dist = (temp_range[0] - temperature_c) / temp_range[0]
        elif temperature_c > temp_range[1]:
            temp_dist = (temperature_c - temp_range[1]) / temp_range[1]

        press_dist = 0.0
        if pressure_bar < pressure_range[0]:
            press_dist = 0.2  # Low pressure
        elif pressure_bar > pressure_range[1]:
            press_dist = (pressure_bar - pressure_range[1]) / pressure_range[1]

        h2_dist = 0.0
        if h2_to_co2 < h2_range[0] or h2_to_co2 > h2_range[1]:
            h2_dist = min(abs(h2_to_co2 - h2_range[0]), abs(h2_to_co2 - h2_range[1])) / h2_range[1]

        # Combine distances
        total_dist = (temp_dist + press_dist + h2_dist) / 3

        # Convert to confidence (exponential decay)
        confidence = np.exp(-3 * total_dist)

        return float(np.clip(confidence, 0.0, 1.0))


def predict_conversion_ml(
    pathway: str,
    catalyst_family: str,
    temperature_c: float,
    pressure_bar: float,
    h2_to_co2: float,
    reactor_type: str,
    grid_ci_gco2_per_kwh: float = 0.0,
    predictor: Optional[MLConversionPredictor] = None
) -> Dict:
    """
    Main prediction function using ML models
    Compatible with existing app interface

    Args:
        pathway: CO2 conversion pathway
        catalyst_family: Catalyst type
        temperature_c: Temperature (°C)
        pressure_bar: Pressure (bar)
        h2_to_co2: H2:CO2 ratio
        reactor_type: Reactor configuration
        grid_ci_gco2_per_kwh: Grid carbon intensity (for electro routes)
        predictor: Optional pre-loaded predictor instance

    Returns:
        Dict with predictions in same format as heuristic model
    """
    # Initialize predictor if not provided
    if predictor is None:
        predictor = MLConversionPredictor()
        if not predictor.load_models():
            raise RuntimeError("Failed to load ML models")

    # Get ML predictions
    ml_preds = predictor.predict(
        pathway, catalyst_family, temperature_c,
        pressure_bar, h2_to_co2, reactor_type
    )

    # Extract predictions
    x_co2 = ml_preds.get('X_CO2', 0.0)
    y_meoh = ml_preds.get('Y_MeOH', 0.0)
    s_meoh = ml_preds.get('S_MeOH', 0.0)
    confidence = ml_preds.get('confidence', 0.5)

    # Calculate yield estimate (for compatibility)
    if pathway == 'Methanol synthesis':
        yield_est = y_meoh
        primary_product = 'CH3OH (methanol)'
    else:
        yield_est = x_co2 * 0.6  # Rough estimate for other pathways
        primary_product = {
            'Sabatier (methanation)': 'CH4 (methane)',
            'RWGS + Fischer-Tropsch': 'C5+ hydrocarbons (diesel/jet range)',
            'Electroreduction (formate route)': 'HCOO− / HCOOH (formate/formic acid)',
            'Electroreduction (CO route)': 'CO (syngas component)',
        }.get(pathway, 'Unknown')

    # Product slate (simplified for ML model)
    if pathway == 'Methanol synthesis':
        slate = {
            'CH3OH': s_meoh,
            'CO': 100 - s_meoh - 5,  # Rough estimate
            'CH4': 3,
            'others': 2
        }
    else:
        # Use heuristic for non-methanol pathways
        slate = {'others': 100.0}

    # Suggestions based on confidence and conditions
    suggestions = []
    if confidence < 0.7:
        suggestions.append(f"⚠️ Prediction confidence: {confidence*100:.0f}% - conditions outside training range")

    if pathway == 'Methanol synthesis':
        if temperature_c < 200:
            suggestions.append("Consider increasing temperature for higher conversion")
        elif temperature_c > 280:
            suggestions.append("High temperature may decrease selectivity")

        if pressure_bar < 20:
            suggestions.append("Higher pressure typically improves methanol synthesis")

    # Notes
    notes = [
        f"ML prediction with {confidence*100:.0f}% confidence",
        "Based on experimental data from Pt-MOF catalysts"
    ]

    if 'Electro' in pathway and grid_ci_gco2_per_kwh > 400:
        notes.append("High grid CI impacts net efficiency for electrochemical routes")

    # Return in compatible format
    return {
        'primary_product': primary_product,
        'yield_percent_estimate': round(yield_est, 1),
        'yield_percent_range': f"{max(3, round(yield_est-5, 1))}–{min(95, round(yield_est+5, 1))}%",
        'product_slate_selectivity': {k: round(v, 1) for k, v in slate.items()},
        'estimated_H2_per_CO2_mol': h2_to_co2,
        'carbon_utilization_proxy_percent': round(min(95, x_co2 * 0.9), 1),
        'operating_scores': {
            'temperature_score': round(confidence, 3),
            'pressure_score': round(confidence, 3),
            'h2_ratio_score': round(confidence, 3),
        },
        'ml_predictions': {
            'X_CO2': round(x_co2, 2),
            'Y_MeOH': round(y_meoh, 2),
            'S_MeOH': round(s_meoh, 2),
            'confidence': round(confidence, 3)
        },
        'suggestions': suggestions or ["Operating conditions within acceptable range"],
        'notes': notes,
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize predictor
    predictor = MLConversionPredictor(models_dir="models")

    if not predictor.load_models():
        print("❌ Failed to load models")
        exit(1)

    print("✅ Models loaded successfully")

    # Test prediction
    result = predict_conversion_ml(
        pathway="Methanol synthesis",
        catalyst_family="Cu/ZnO/Al2O3",
        temperature_c=250,
        pressure_bar=40,
        h2_to_co2=3.0,
        reactor_type="Fixed-bed",
        predictor=predictor
    )

    print("\n📊 Prediction Results:")
    print(f"  Primary product: {result['primary_product']}")
    print(f"  Yield estimate: {result['yield_percent_estimate']}%")
    print(f"  Yield range: {result['yield_percent_range']}")
    print(f"\n  ML Predictions:")
    print(f"    X_CO2: {result['ml_predictions']['X_CO2']}%")
    print(f"    Y_MeOH: {result['ml_predictions']['Y_MeOH']}%")
    print(f"    S_MeOH: {result['ml_predictions']['S_MeOH']}%")
    print(f"    Confidence: {result['ml_predictions']['confidence']*100:.1f}%")
    print(f"\n  Suggestions:")
    for s in result['suggestions']:
        print(f"    - {s}")
