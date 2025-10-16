"""
Country-specific Net Zero policy commitments tracking.
Supports detailed sector-wise emission reduction actions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import json
import os

@dataclass
class PolicyAction:
    """Một hành động/chính sách cụ thể trong một lĩnh vực"""
    sector: str  # "Energy", "Transport", "Industry", "Agriculture", etc.
    action_name: str  # "Solar capacity expansion", "EV adoption", etc.
    baseline_year: int
    baseline_emissions_mtco2: float  # Million tons CO2
    share_pct: float  # % tỷ trọng của sector trong tổng phát thải baseline
    reduction_target_pct: float  # % giảm so với baseline
    implementation_years: List[int]  # [2024, 2025, 2026...]
    yearly_improvement_pct: List[float]  # [5%, 8%, 12%...] - tích lũy
    status: str  # "On track", "Behind schedule", "Ahead"

    def get_reduction_for_year(self, year: int, target_year: int = None) -> float:
        """
        Get cumulative reduction percentage for a specific year.

        If year is after last data point, automatically extrapolates linearly
        toward reduction_target_pct.

        Args:
            year: Year to get reduction for
            target_year: Optional target year for extrapolation (typically from CountryCommitment)

        Returns:
            Reduction percentage (0-100+)
        """
        if year < self.baseline_year:
            return 0.0

        if not self.implementation_years or not self.yearly_improvement_pct:
            return 0.0

        # If year is before first implementation year, no reduction yet
        if year < self.implementation_years[0]:
            return 0.0

        last_data_year = self.implementation_years[-1]
        last_data_reduction = self.yearly_improvement_pct[-1]

        # If year is within data range, interpolate
        if year <= last_data_year:
            # Find the reduction for this year
            for i, impl_year in enumerate(self.implementation_years):
                if year == impl_year:
                    return self.yearly_improvement_pct[i]
                elif i > 0 and self.implementation_years[i-1] < year < impl_year:
                    # Linear interpolation between years
                    prev_year = self.implementation_years[i-1]
                    prev_reduction = self.yearly_improvement_pct[i-1]
                    next_reduction = self.yearly_improvement_pct[i]

                    fraction = (year - prev_year) / (impl_year - prev_year)
                    return prev_reduction + (next_reduction - prev_reduction) * fraction
            return last_data_reduction

        # Year is after last data point - AUTO-EXTRAPOLATION
        # If we've already reached target, maintain it
        if last_data_reduction >= self.reduction_target_pct:
            return self.reduction_target_pct

        # Calculate extrapolation slope
        # Estimate time to reach target based on historical rate or reasonable timeframe
        if target_year and target_year > last_data_year:
            # Use provided target_year for extrapolation
            years_to_target = target_year - last_data_year
        else:
            # Fallback: Estimate based on data span (assume similar pace)
            data_span = last_data_year - self.implementation_years[0]
            # Extrapolate over similar timespan, minimum 10 years
            years_to_target = max(10, data_span)

        # Linear extrapolation slope
        reduction_gap = self.reduction_target_pct - last_data_reduction
        slope = reduction_gap / years_to_target

        # Calculate extrapolated value
        years_elapsed = year - last_data_year
        extrapolated = last_data_reduction + slope * years_elapsed

        # Cap at reduction_target_pct
        return min(extrapolated, self.reduction_target_pct)

    def get_absolute_reduction_mtco2(self, year: int) -> float:
        """Get absolute emission reduction in MtCO2 for a given year."""
        reduction_pct = self.get_reduction_for_year(year)
        return self.baseline_emissions_mtco2 * (reduction_pct / 100.0)


@dataclass
class CountryCommitment:
    """Cam kết Net Zero chi tiết của một quốc gia"""
    country: str
    target_year: int  # Năm mục tiêu (thay vì net_zero_year)
    target_reduction_pct: float  # % giảm so với baseline (0-100)
    baseline_year: int
    baseline_emissions_gtco2: float
    policy_actions: List[PolicyAction]

    @property
    def target_emissions_gtco2(self) -> float:
        """Calculate target emissions based on reduction percentage."""
        return self.baseline_emissions_gtco2 * (1 - self.target_reduction_pct / 100.0)

    def get_sector_breakdown(self) -> Dict[str, float]:
        """Tính tổng emissions baseline theo sector"""
        breakdown = {}
        for action in self.policy_actions:
            if action.sector not in breakdown:
                breakdown[action.sector] = 0.0
            breakdown[action.sector] += action.baseline_emissions_mtco2
        return breakdown

    def calculate_annual_reduction_fraction(self, year: int) -> float:
        """
        Calculate total reduction fraction including both emitters and carbon sinks.

        Separates calculation into:
        1. EMITTERS (baseline > 0): Use share_pct weighted reduction
        2. SINKS (baseline < 0): Calculate additional removal in MtCO2, convert to fraction

        Automatically extrapolates if year is beyond policy data range.

        Returns: fraction of baseline emissions reduced (e.g., 0.15 = 15% reduction)

        Note: For sinks, reduction_target_pct means "increase removal capacity by X%"
              E.g., baseline -15 MtCO2 with 200% target = -45 MtCO2 (triple removal)
        """
        emitters_reduction = 0.0
        sinks_contribution = 0.0

        for action in self.policy_actions:
            # Get reduction percentage for this year (with auto-extrapolation)
            reduction_pct = action.get_reduction_for_year(year, target_year=self.target_year)

            if action.baseline_emissions_mtco2 > 0:
                # EMITTER: Use share_pct weighted reduction
                weighted_reduction = (action.share_pct / 100.0) * (reduction_pct / 100.0)
                emitters_reduction += weighted_reduction

            elif action.baseline_emissions_mtco2 < 0:
                # SINK: Calculate additional removal capacity
                # baseline_emissions_mtco2 is negative (e.g., -15 MtCO2)
                # reduction_pct represents increase in removal (e.g., 200% = triple capacity)

                baseline_removal = abs(action.baseline_emissions_mtco2)  # Convert to positive
                additional_removal = baseline_removal * (reduction_pct / 100.0)

                # Convert additional removal to equivalent reduction fraction
                # Divide by baseline emissions in MtCO2 (convert GtCO2 to MtCO2)
                baseline_mtco2 = self.baseline_emissions_gtco2 * 1000
                sinks_contribution += additional_removal / baseline_mtco2

        # Total reduction = emitters reduction + sink contribution
        total_reduction = emitters_reduction + sinks_contribution

        # Cap at 100% reduction (can exceed if sinks are very strong)
        return min(total_reduction, 1.0)

    def get_sector_contributions(self) -> Dict[str, float]:
        """
        Get each sector's contribution to total reduction target.
        Returns: Dict of sector -> MtCO2 reduction
        """
        contributions = {}
        for action in self.policy_actions:
            if action.sector not in contributions:
                contributions[action.sector] = 0.0

            # Calculate total reduction potential for this action
            reduction_amount = (
                action.baseline_emissions_mtco2 *
                action.reduction_target_pct / 100.0
            )
            contributions[action.sector] += reduction_amount

        return contributions

    @classmethod
    def from_dict(cls, data: dict) -> 'CountryCommitment':
        """Create CountryCommitment from dictionary."""
        policy_actions = [
            PolicyAction(**action_data)
            for action_data in data.get('policy_actions', [])
        ]

        return cls(
            country=data['country'],
            target_year=data['target_year'],
            target_reduction_pct=data['target_reduction_pct'],
            baseline_year=data['baseline_year'],
            baseline_emissions_gtco2=data['baseline_emissions_gtco2'],
            policy_actions=policy_actions
        )


def load_country_commitment(country: str, data_dir: str = "data") -> CountryCommitment:
    """
    Load country commitment data from JSON file.

    Args:
        country: Country name (e.g., "Vietnam", "USA", "Japan")
        data_dir: Directory containing country_commitments.json

    Returns:
        CountryCommitment object

    Raises:
        FileNotFoundError: If data file doesn't exist
        KeyError: If country not found in data file
    """
    json_path = os.path.join(data_dir, "country_commitments.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Country commitments file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    if country not in all_data:
        available = list(all_data.keys())
        raise KeyError(f"Country '{country}' not found. Available: {available}")

    return CountryCommitment.from_dict(all_data[country])


def get_available_countries(data_dir: str = "data") -> List[str]:
    """
    Get list of countries with commitment data available.

    Args:
        data_dir: Directory containing country_commitments.json

    Returns:
        List of country names
    """
    json_path = os.path.join(data_dir, "country_commitments.json")

    if not os.path.exists(json_path):
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    return list(all_data.keys())


def load_country_commitment_from_json(json_content: str, country: str) -> CountryCommitment:
    """
    Load country commitment data from JSON string content.

    Args:
        json_content: JSON string content (from uploaded file)
        country: Country name to extract

    Returns:
        CountryCommitment object

    Raises:
        json.JSONDecodeError: If JSON is invalid
        KeyError: If country not found in data
    """
    all_data = json.loads(json_content)

    if country not in all_data:
        available = list(all_data.keys())
        raise KeyError(f"Country '{country}' not found. Available: {available}")

    return CountryCommitment.from_dict(all_data[country])


def get_countries_from_json(json_content: str) -> List[str]:
    """
    Get list of countries from JSON content.

    Args:
        json_content: JSON string content

    Returns:
        List of country names
    """
    try:
        all_data = json.loads(json_content)
        return list(all_data.keys())
    except json.JSONDecodeError:
        return []


def validate_commitment_json(json_content: str) -> tuple[bool, str]:
    """
    Validate country commitment JSON structure.

    Args:
        json_content: JSON string content

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        all_data = json.loads(json_content)

        if not isinstance(all_data, dict):
            return False, "JSON must be an object with country names as keys"

        for country_name, country_data in all_data.items():
            # Check required country fields
            required_fields = ['country', 'target_year', 'target_reduction_pct',
                             'baseline_year', 'baseline_emissions_gtco2', 'policy_actions']

            for field in required_fields:
                if field not in country_data:
                    return False, f"Country '{country_name}' missing required field: {field}"

            # Check policy actions
            if not isinstance(country_data['policy_actions'], list):
                return False, f"Country '{country_name}' policy_actions must be an array"

            for i, action in enumerate(country_data['policy_actions']):
                action_required = ['sector', 'action_name', 'baseline_year',
                                 'baseline_emissions_mtco2', 'share_pct',
                                 'reduction_target_pct', 'implementation_years',
                                 'yearly_improvement_pct', 'status']

                for field in action_required:
                    if field not in action:
                        return False, f"Country '{country_name}' action {i} missing field: {field}"

                # Validate arrays have same length
                impl_years = action['implementation_years']
                yearly_pct = action['yearly_improvement_pct']

                if len(impl_years) != len(yearly_pct):
                    return False, f"Country '{country_name}' action {i} ({action['sector']}): implementation_years and yearly_improvement_pct must have same length"

        return True, "JSON is valid"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON syntax: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"
