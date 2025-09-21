
from __future__ import annotations
import pandas as pd

OWID_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

def load_owid_global_timeseries() -> pd.DataFrame:
    """Download latest Our World in Data CO2 dataset and return standardized DataFrame.
    
    Returns
    -------
    pd.DataFrame
        Columns: ['country','year','emissions_gtco2']
        Uses OWID 'co2' (annual CO2 emissions in MtCO2) converted to GtCO2.
    """
    df = pd.read_csv(OWID_URL)
    df = df[['country','year','co2']].dropna(subset=['co2'])
    # Convert MtCO2 to GtCO2
    df['emissions_gtco2'] = df['co2'] / 1000.0
    df = df.drop(columns=['co2']).sort_values(['country','year'])
    return df

def list_regions(df: pd.DataFrame) -> list[str]:
    return sorted(df['country'].unique().tolist())

def filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    d = df[df['country'] == country][['year','emissions_gtco2']].dropna()
    return d.sort_values('year').reset_index(drop=True)
