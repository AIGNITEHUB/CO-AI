"""
Test script to verify forecast mode and target mode logic separation.
"""
import json
from logic.policy_commitments import (
    load_country_commitment_from_json,
    validate_commitment_json
)

def test_forecast_mode():
    """Test forecast mode with vietnam_forecast_mode.json"""
    print("\n=== Testing Forecast Mode ===")

    # Load the Vietnam forecast mode JSON
    with open("data/vietnam_forecast_mode.json", 'r', encoding='utf-8') as f:
        json_content = f.read()

    # Validate JSON
    is_valid, msg = validate_commitment_json(json_content)
    print(f"Validation: {'PASS' if is_valid else 'FAIL'} - {msg}")

    if not is_valid:
        return False

    # Load commitment data
    commitment = load_country_commitment_from_json(json_content, "Vietnam")

    # Check mode
    print(f"Mode: {commitment.mode}")
    print(f"Is forecast mode: {commitment.is_forecast_mode}")
    print(f"Target reduction %: {commitment.target_reduction_pct}")
    print(f"Target year: {commitment.target_year}")
    print(f"Baseline emissions: {commitment.baseline_emissions_gtco2} GtCO2")

    # Test projection calculation for 2030
    reduction_2030 = commitment.calculate_annual_reduction_fraction(2030)
    projected_2030 = commitment.baseline_emissions_gtco2 * (1 - reduction_2030)
    print(f"\nProjection for 2030:")
    print(f"  Reduction fraction: {reduction_2030:.4f} ({reduction_2030*100:.2f}%)")
    print(f"  Projected emissions: {projected_2030:.4f} GtCO2")

    # Test projection calculation for 2050
    reduction_2050 = commitment.calculate_annual_reduction_fraction(2050)
    projected_2050 = commitment.baseline_emissions_gtco2 * (1 - reduction_2050)
    print(f"\nProjection for 2050 (with auto-extrapolation):")
    print(f"  Reduction fraction: {reduction_2050:.4f} ({reduction_2050*100:.2f}%)")
    print(f"  Projected emissions: {projected_2050:.4f} GtCO2")

    # Check policy actions
    print(f"\nPolicy actions: {len(commitment.policy_actions)}")
    for action in commitment.policy_actions:
        print(f"  - {action.sector}: {action.action_name}")
        print(f"    Target %: {action.target_pct}% (should be 0 for forecast mode)")
        print(f"    Share %: {action.share_pct:.2f}%")

    return True

def test_target_mode():
    """Test target mode (would need a target mode JSON file)"""
    print("\n=== Testing Target Mode ===")
    print("Note: Would need a target mode JSON file to test this")
    print("Target mode should have:")
    print("  - mode: 'target'")
    print("  - target_reduction_pct at country level")
    print("  - Policy actions with specific reduction targets")

    return True

def test_validation():
    """Test validation for both modes"""
    print("\n=== Testing Validation ===")

    # Test valid forecast mode JSON
    forecast_json = """{
        "Vietnam": {
            "country": "Vietnam",
            "mode": "forecast",
            "target_year": 2050,
            "baseline_year": 2025,
            "baseline_emissions_gtco2": 0.45,
            "policy_actions": [
                {
                    "sector": "Energy",
                    "action_name": "Test",
                    "baseline_year": 2025,
                    "baseline_emissions_mtco2": 225.0,
                    "reduction_target_pct": 0,
                    "implementation_years": [2026],
                    "yearly_improvement_pct": [5.0],
                    "status": "On track"
                }
            ]
        }
    }"""

    is_valid, msg = validate_commitment_json(forecast_json)
    print(f"Forecast mode (no target_reduction_pct): {'PASS' if is_valid else 'FAIL'}")

    # Test valid target mode JSON
    target_json = """{
        "Vietnam": {
            "country": "Vietnam",
            "mode": "target",
            "target_year": 2050,
            "baseline_year": 2025,
            "baseline_emissions_gtco2": 0.45,
            "target_reduction_pct": 50.0,
            "policy_actions": [
                {
                    "sector": "Energy",
                    "action_name": "Test",
                    "baseline_year": 2025,
                    "baseline_emissions_mtco2": 225.0,
                    "reduction_target_pct": 60.0,
                    "implementation_years": [2026],
                    "yearly_improvement_pct": [5.0],
                    "status": "On track"
                }
            ]
        }
    }"""

    is_valid, msg = validate_commitment_json(target_json)
    print(f"Target mode (with target_reduction_pct): {'PASS' if is_valid else 'FAIL'}")

    # Test invalid: target mode without target_reduction_pct
    invalid_json = """{
        "Vietnam": {
            "country": "Vietnam",
            "mode": "target",
            "target_year": 2050,
            "baseline_year": 2025,
            "baseline_emissions_gtco2": 0.45,
            "policy_actions": [
                {
                    "sector": "Energy",
                    "action_name": "Test",
                    "baseline_year": 2025,
                    "baseline_emissions_mtco2": 225.0,
                    "reduction_target_pct": 60.0,
                    "implementation_years": [2026],
                    "yearly_improvement_pct": [5.0],
                    "status": "On track"
                }
            ]
        }
    }"""

    is_valid, msg = validate_commitment_json(invalid_json)
    print(f"Target mode without target_reduction_pct: {'FAIL (expected)' if not is_valid else 'PASS (unexpected!)'}")
    if not is_valid:
        print(f"  Error message: {msg}")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Forecast Mode and Target Mode Separation")
    print("=" * 60)

    test_forecast_mode()
    test_target_mode()
    test_validation()

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
