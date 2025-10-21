"""
Train XGBoost models for CO2 conversion prediction
For integration with Streamlit app
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import json

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("="*80)
print("TRAINING XGBOOST FOR CO2 CONVERSION PREDICTION")
print("="*80)

# Load complete dataset
df = pd.read_csv('co2_conversion_complete.csv')
print(f"\n✅ Loaded {len(df)} records")
print(f"   Pathways: {df['pathway'].nunique()}")
print(f"   Catalyst families: {df['catalyst_family'].nunique()}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Encode categorical features
encoders = {}

# Pathway
le_pathway = LabelEncoder()
df['pathway_encoded'] = le_pathway.fit_transform(df['pathway'])
encoders['pathway'] = le_pathway

# Catalyst family
le_catalyst = LabelEncoder()
df['catalyst_encoded'] = le_catalyst.fit_transform(df['catalyst_family'])
encoders['catalyst'] = le_catalyst

# Reactor type
le_reactor = LabelEncoder()
df['reactor_encoded'] = le_reactor.fit_transform(df['reactor_type'])
encoders['reactor'] = le_reactor

print(f"\nEncodings created:")
print(f"  Pathways: {dict(zip(le_pathway.classes_, le_pathway.transform(le_pathway.classes_)))}")
print(f"  Catalysts: {dict(zip(le_catalyst.classes_, le_catalyst.transform(le_catalyst.classes_)))}")
print(f"  Reactors: {dict(zip(le_reactor.classes_, le_reactor.transform(le_reactor.classes_)))}")

# Define feature columns
feature_cols = [
    'pathway_encoded',
    'catalyst_encoded',
    'reactor_encoded',
    'temperature_c',
    'pressure_bar',
    'h2_co2_ratio',
    '1000_T_K',
    'temp_pressure',
    'ln_pressure',
    'is_thermal',
    'is_methanation',
    'is_methanol'
]

print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

# ============================================================================
# TRAIN MODELS FOR DIFFERENT TARGETS
# ============================================================================

# Define targets
targets = {
    'X_CO2': 'CO2 Conversion (%)',
    'Y_MeOH': 'Methanol Yield (%)'
}

models = {}
scalers = {}
results = {}

for target_name, target_desc in targets.items():
    print(f"\n{'='*60}")
    print(f"TRAINING: {target_desc}")
    print(f"{'='*60}")

    # Prepare data (filter out rows with missing targets)
    df_target = df[df[target_name].notna()].copy()
    X = df_target[feature_cols]
    y = df_target[target_name]

    print(f"\nTraining samples: {len(X)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )

    # Train
    print("\nTraining...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='rmse',
        verbose=False
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_scaled, y, cv=5, scoring='r2', n_jobs=-1
    )

    print(f"\n{'RESULTS':^60}")
    print(f"{'-'*60}")
    print(f"{'Metric':20s} {'Train':>15s} {'Test':>15s}")
    print(f"{'-'*60}")
    print(f"{'R² Score':20s} {train_r2:15.4f} {test_r2:15.4f}")
    print(f"{'RMSE':20s} {train_rmse:15.4f} {test_rmse:15.4f}")
    print(f"{'MAE':20s} {train_mae:15.4f} {test_mae:15.4f}")
    print(f"{'-'*60}")
    print(f"{'5-Fold CV R²':20s} {cv_scores.mean():15.4f} ± {cv_scores.std():6.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'FEATURE IMPORTANCE':^60}")
    print(f"{'-'*60}")
    for idx, row in feature_importance.head(8).iterrows():
        print(f"{row['feature']:30s} {row['importance']:10.4f}")

    # Store
    models[target_name] = model
    scalers[target_name] = scaler
    results[target_name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# Save XGBoost models
for target, model in models.items():
    model_file = f'models/xgboost_{target}_conversion.json'
    model.save_model(model_file)
    print(f"✅ Saved: {model_file}")

# Save scalers
scaler_file = 'models/conversion_scalers.pkl'
joblib.dump(scalers, scaler_file)
print(f"✅ Saved: {scaler_file}")

# Save encoders
encoder_file = 'models/conversion_encoders.pkl'
joblib.dump(encoders, encoder_file)
print(f"✅ Saved: {encoder_file}")

# Save feature columns
features_file = 'models/feature_columns.json'
with open(features_file, 'w') as f:
    json.dump({'features': feature_cols}, f, indent=2)
print(f"✅ Saved: {features_file}")

# Save results summary
results_df = pd.DataFrame(results).T
results_df.to_csv('models/training_results.csv')
print(f"✅ Saved: models/training_results.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('XGBoost CO2 Conversion Models - Performance',
             fontsize=14, fontweight='bold')

# Plot 1: R² Comparison
ax1 = axes[0, 0]
target_names = list(results.keys())
train_r2s = [results[t]['train_r2'] for t in target_names]
test_r2s = [results[t]['test_r2'] for t in target_names]
x_pos = np.arange(len(target_names))
width = 0.35

ax1.bar(x_pos - width/2, train_r2s, width, label='Train', color='#3498db')
ax1.bar(x_pos + width/2, test_r2s, width, label='Test', color='#e74c3c')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(target_names)
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance - R²')
ax1.legend()
ax1.set_ylim([0, 1])
ax1.grid(alpha=0.3, axis='y')

# Plot 2: RMSE Comparison
ax2 = axes[0, 1]
train_rmses = [results[t]['train_rmse'] for t in target_names]
test_rmses = [results[t]['test_rmse'] for t in target_names]

ax2.bar(x_pos - width/2, train_rmses, width, label='Train', color='#2ecc71')
ax2.bar(x_pos + width/2, test_rmses, width, label='Test', color='#f39c12')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(target_names)
ax2.set_ylabel('RMSE')
ax2.set_title('Model Performance - RMSE')
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

# Plot 3: Feature Importance for X_CO2
ax3 = axes[1, 0]
fi = pd.DataFrame({
    'feature': feature_cols,
    'importance': models['X_CO2'].feature_importances_
}).sort_values('importance', ascending=True).tail(8)

ax3.barh(fi['feature'], fi['importance'], color='#9b59b6')
ax3.set_xlabel('Importance')
ax3.set_title('Feature Importance - X_CO2')
ax3.grid(alpha=0.3, axis='x')

# Plot 4: CV Scores
ax4 = axes[1, 1]
cv_means = [results[t]['cv_r2_mean'] for t in target_names]
cv_stds = [results[t]['cv_r2_std'] for t in target_names]

ax4.bar(x_pos, cv_means, yerr=cv_stds, color='#34495e',
        alpha=0.8, capsize=5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(target_names)
ax4.set_ylabel('R² Score')
ax4.set_title('5-Fold Cross-Validation')
ax4.set_ylim([0, 1])
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models/training_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: models/training_results.png")

# ============================================================================
# PREDICTION EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("PREDICTION EXAMPLE")
print("="*80)

# Example input: Methanol synthesis, Cu/ZnO catalyst, 250°C, 40 bar
example = pd.DataFrame({
    'pathway': ['Methanol synthesis'],
    'catalyst_family': ['Cu/ZnO/Al2O3'],
    'reactor_type': ['Fixed-bed'],
    'temperature_c': [250],
    'pressure_bar': [40],
    'h2_co2_ratio': [3.0]
})

# Calculate derived features
example['1000_T_K'] = 1000 / (example['temperature_c'] + 273.15)
example['temp_pressure'] = example['temperature_c'] * example['pressure_bar']
example['ln_pressure'] = np.log(example['pressure_bar'] + 1)
example['is_thermal'] = 1
example['is_methanation'] = 0
example['is_methanol'] = 1

# Encode
example['pathway_encoded'] = encoders['pathway'].transform(example['pathway'])
example['catalyst_encoded'] = encoders['catalyst'].transform(example['catalyst_family'])
example['reactor_encoded'] = encoders['reactor'].transform(example['reactor_type'])

# Select features
X_example = example[feature_cols]

print("\nInput conditions:")
print(f"  Pathway: Methanol synthesis")
print(f"  Catalyst: Cu/ZnO/Al2O3")
print(f"  Temperature: 250°C")
print(f"  Pressure: 40 bar")
print(f"  H2:CO2 ratio: 3.0")

print("\nPredictions:")
for target in models.keys():
    X_scaled = scalers[target].transform(X_example)
    pred = models[target].predict(X_scaled)[0]
    print(f"  {target:10s}: {pred:6.2f}%")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - models/xgboost_X_CO2_conversion.json")
print("  - models/xgboost_Y_MeOH_conversion.json")
print("  - models/conversion_scalers.pkl")
print("  - models/conversion_encoders.pkl")
print("  - models/feature_columns.json")
print("  - models/training_results.csv")
print("  - models/training_results.png")
print("\n✅ Models ready for Streamlit integration!")
