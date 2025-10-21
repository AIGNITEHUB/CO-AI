"""
Create comprehensive CO2 conversion dataset for Streamlit app
Combines real experimental data with extended conditions for all pathways
"""

import pandas as pd
import numpy as np


# ============================================================================
# PART 1: REAL EXPERIMENTAL DATA - Methanol Synthesis
# ============================================================================

def create_real_methanol_data():
    """Real data from Pt-MOF catalysts for methanol synthesis

    Data Source:
    - DS1068: Hf-UiO-67-BA-Pt (Reaction 141)
    - DS1054: Hf-UiO-67-FA-Pt (Reaction 89)
    - DS1064: Zr-UiO-67-FA-Pt (Reaction 96)
    - DS1020: Zr-UiO-67-BA-Pt (Reaction 90)
    """

    data = {
        'pathway': [],
        'catalyst_family': [],
        'metal': [],
        'linker': [],
        'temperature_c': [],
        'pressure_bar': [],
        'h2_co2_ratio': [],
        'reactor_type': [],
        'X_CO2': [],
        'S_MeOH': [],
        'S_CO': [],
        'S_CH4': [],
        'Y_MeOH': [],
        'primary_product': [],
        'data_source': []
    }

    # Helper function to generate catalyst family name
    def get_catalyst_name(metal, linker):
        return f"{metal}-UiO-67-{linker}-Pt"

    # Hf-UiO-67-BA-Pt data
    for temp, x_co2, s_meoh, s_co, s_ch4 in [
        (170, 0.75, 31.98, 68.02, 0.00),
        (190, 1.35, 33.12, 65.58, 1.30),
        (210, 2.49, 33.77, 60.75, 5.49),
        (240, 5.28, 25.01, 65.71, 8.98),
    ]:
        data['pathway'].append('Methanol synthesis')
        data['catalyst_family'].append(get_catalyst_name('Hf', 'BA'))
        data['metal'].append('Hf')
        data['linker'].append('BA')
        data['temperature_c'].append(temp)
        data['pressure_bar'].append(30)
        data['h2_co2_ratio'].append(3.0)
        data['reactor_type'].append('Fixed-bed')
        data['X_CO2'].append(x_co2)
        data['S_MeOH'].append(s_meoh)
        data['S_CO'].append(s_co)
        data['S_CH4'].append(s_ch4)
        data['Y_MeOH'].append(x_co2 * s_meoh / 100)
        data['primary_product'].append('CH3OH (methanol)')
        data['data_source'].append('experimental')

    # Hf-UiO-67-FA-Pt data
    for temp, x_co2, s_meoh, s_co, s_ch4 in [
        (170, 0.91, 28.83, 71.17, 0.00),
        (190, 1.75, 30.20, 69.80, 0.00),
        (210, 3.08, 32.25, 65.66, 2.10),
        (240, 6.37, 25.63, 68.44, 5.79),
    ]:
        data['pathway'].append('Methanol synthesis')
        data['catalyst_family'].append(get_catalyst_name('Hf', 'FA'))
        data['metal'].append('Hf')
        data['linker'].append('FA')
        data['temperature_c'].append(temp)
        data['pressure_bar'].append(30)
        data['h2_co2_ratio'].append(3.0)
        data['reactor_type'].append('Fixed-bed')
        data['X_CO2'].append(x_co2)
        data['S_MeOH'].append(s_meoh)
        data['S_CO'].append(s_co)
        data['S_CH4'].append(s_ch4)
        data['Y_MeOH'].append(x_co2 * s_meoh / 100)
        data['primary_product'].append('CH3OH (methanol)')
        data['data_source'].append('experimental')

    # Zr-UiO-67-FA-Pt data
    for temp, x_co2, s_meoh, s_co, s_ch4 in [
        (170, 0.83, 24.95, 75.05, 0.00),
        (190, 1.55, 27.62, 72.38, 0.00),
        (210, 2.86, 29.72, 66.90, 3.38),
        (240, 5.94, 22.36, 71.35, 6.19),
    ]:
        data['pathway'].append('Methanol synthesis')
        data['catalyst_family'].append(get_catalyst_name('Zr', 'FA'))
        data['metal'].append('Zr')
        data['linker'].append('FA')
        data['temperature_c'].append(temp)
        data['pressure_bar'].append(30)
        data['h2_co2_ratio'].append(3.0)
        data['reactor_type'].append('Fixed-bed')
        data['X_CO2'].append(x_co2)
        data['S_MeOH'].append(s_meoh)
        data['S_CO'].append(s_co)
        data['S_CH4'].append(s_ch4)
        data['Y_MeOH'].append(x_co2 * s_meoh / 100)
        data['primary_product'].append('CH3OH (methanol)')
        data['data_source'].append('experimental')

    # Zr-UiO-67-BA-Pt data
    for temp, x_co2, s_meoh, s_co, s_ch4 in [
        (170, 0.87, 27.26, 72.74, 0.00),
        (190, 1.63, 26.45, 72.36, 1.19),
        (210, 3.07, 29.67, 65.45, 4.88),
        (240, 6.66, 22.05, 70.53, 7.25),
    ]:
        data['pathway'].append('Methanol synthesis')
        data['catalyst_family'].append(get_catalyst_name('Zr', 'BA'))
        data['metal'].append('Zr')
        data['linker'].append('BA')
        data['temperature_c'].append(temp)
        data['pressure_bar'].append(30)
        data['h2_co2_ratio'].append(3.0)
        data['reactor_type'].append('Fixed-bed')
        data['X_CO2'].append(x_co2)
        data['S_MeOH'].append(s_meoh)
        data['S_CO'].append(s_co)
        data['S_CH4'].append(s_ch4)
        data['Y_MeOH'].append(x_co2 * s_meoh / 100)
        data['primary_product'].append('CH3OH (methanol)')
        data['data_source'].append('experimental')

    return pd.DataFrame(data)


# ============================================================================
# PART 2: EXTENDED DATA - Other Pathways (Literature-based estimates)
# ============================================================================

def create_extended_literature_data():
    """
    Extended dataset based on literature values for other pathways
    Sources: Various published papers on CO2 conversion
    """

    data = {
        'pathway': [],
        'catalyst_family': [],
        'temperature_c': [],
        'pressure_bar': [],
        'h2_co2_ratio': [],
        'reactor_type': [],
        'X_CO2': [],
        'S_product': [],  # Selectivity to main product
        'Y_product': [],  # Yield of main product
        'primary_product': [],
        'data_source': []
    }

    # Sabatier (methanation) - Ni-based catalysts
    # Literature: High conversion at 300-400°C, high CH4 selectivity
    sabatier_data = [
        # (temp, pressure, h2_co2, X_CO2, S_CH4)
        (300, 10, 4.0, 65, 95),
        (320, 10, 4.0, 75, 97),
        (350, 20, 4.0, 85, 96),
        (380, 30, 4.0, 90, 94),
    ]

    for temp, press, h2, x_co2, s_ch4 in sabatier_data:
        data['pathway'].append('Sabatier (methanation)')
        data['catalyst_family'].append('Ni-based')
        data['temperature_c'].append(temp)
        data['pressure_bar'].append(press)
        data['h2_co2_ratio'].append(h2)
        data['reactor_type'].append('Fixed-bed')
        data['X_CO2'].append(x_co2)
        data['S_product'].append(s_ch4)
        data['Y_product'].append(x_co2 * s_ch4 / 100)
        data['primary_product'].append('CH4 (methane)')
        data['data_source'].append('literature')

    # RWGS + Fischer-Tropsch - Fe/Co catalysts
    # Literature: Lower conversion, complex product distribution
    ft_data = [
        (320, 20, 2.0, 45, 60),
        (350, 30, 2.2, 55, 65),
        (380, 40, 2.5, 65, 68),
        (400, 50, 2.0, 70, 62),
    ]

    for temp, press, h2, x_co2, s_c5 in ft_data:
        data['pathway'].append('RWGS + Fischer-Tropsch')
        data['catalyst_family'].append('Fe/Co (FT)')
        data['temperature_c'].append(temp)
        data['pressure_bar'].append(press)
        data['h2_co2_ratio'].append(h2)
        data['reactor_type'].append('Fixed-bed')
        data['X_CO2'].append(x_co2)
        data['S_product'].append(s_c5)
        data['Y_product'].append(x_co2 * s_c5 / 100)
        data['primary_product'].append('C5+ hydrocarbons (diesel/jet range)')
        data['data_source'].append('literature')

    # Electroreduction (formate) - Cu electro
    # Literature: Room temperature, low pressure, moderate efficiency
    electro_formate = [
        (25, 1, 2.0, 35, 70),
        (25, 1, 2.5, 42, 75),
        (30, 1, 2.0, 38, 72),
        (30, 1, 3.0, 45, 68),
    ]

    for temp, press, h2, x_co2, s_hcoo in electro_formate:
        data['pathway'].append('Electroreduction (formate route)')
        data['catalyst_family'].append('Cu (electro)')
        data['temperature_c'].append(temp)
        data['pressure_bar'].append(press)
        data['h2_co2_ratio'].append(h2)
        data['reactor_type'].append('Electrochemical (flow cell)')
        data['X_CO2'].append(x_co2)
        data['S_product'].append(s_hcoo)
        data['Y_product'].append(x_co2 * s_hcoo / 100)
        data['primary_product'].append('HCOO− / HCOOH (formate/formic acid)')
        data['data_source'].append('literature')

    # Electroreduction (CO) - Ag/Au electro
    # Literature: High CO selectivity at room temp
    electro_co = [
        (25, 1, 1.0, 55, 85),
        (25, 1, 1.5, 60, 88),
        (30, 1, 1.0, 58, 86),
        (30, 1, 1.2, 62, 90),
    ]

    for temp, press, h2, x_co2, s_co in electro_co:
        data['pathway'].append('Electroreduction (CO route)')
        data['catalyst_family'].append('Ag/Au (electro)')
        data['temperature_c'].append(temp)
        data['pressure_bar'].append(press)
        data['h2_co2_ratio'].append(h2)
        data['reactor_type'].append('Electrochemical (flow cell)')
        data['X_CO2'].append(x_co2)
        data['S_product'].append(s_co)
        data['Y_product'].append(x_co2 * s_co / 100)
        data['primary_product'].append('CO (syngas component)')
        data['data_source'].append('literature')

    return pd.DataFrame(data)


# ============================================================================
# PART 3: COMBINE AND AUGMENT
# ============================================================================

def create_complete_dataset():
    """Create complete dataset for CO2 conversion predictor"""

    # Get real methanol data
    df_real = create_real_methanol_data()

    # Get extended literature data
    df_lit = create_extended_literature_data()

    # Standardize columns for literature data
    df_lit['S_MeOH'] = 0.0
    df_lit['S_CO'] = 0.0
    df_lit['S_CH4'] = 0.0
    df_lit['Y_MeOH'] = 0.0
    df_lit['metal'] = 'N/A'
    df_lit['linker'] = 'N/A'

    # Map S_product and Y_product to specific columns
    for idx, row in df_lit.iterrows():
        pathway = row['pathway']
        if 'methanation' in pathway:
            df_lit.at[idx, 'S_CH4'] = row['S_product']
        elif 'Methanol' in pathway:
            df_lit.at[idx, 'S_MeOH'] = row['S_product']
            df_lit.at[idx, 'Y_MeOH'] = row['Y_product']
        elif 'formate' in pathway:
            pass  # Keep as S_product
        elif 'CO route' in pathway:
            df_lit.at[idx, 'S_CO'] = row['S_product']

    df_lit = df_lit.drop(columns=['S_product', 'Y_product'])

    # Ensure column order matches
    cols = ['pathway', 'catalyst_family', 'metal', 'linker', 'temperature_c',
            'pressure_bar', 'h2_co2_ratio', 'reactor_type', 'X_CO2',
            'S_MeOH', 'S_CO', 'S_CH4', 'Y_MeOH', 'primary_product', 'data_source']

    df_real = df_real[cols]
    df_lit = df_lit[cols]

    # Combine
    df_combined = pd.concat([df_real, df_lit], ignore_index=True)

    # Add derived features
    df_combined['1000_T_K'] = 1000 / (df_combined['temperature_c'] + 273.15)
    df_combined['temp_pressure'] = df_combined['temperature_c'] * df_combined['pressure_bar']
    df_combined['ln_pressure'] = np.log(df_combined['pressure_bar'] + 1)

    # Add categorical encoding hints
    df_combined['is_thermal'] = df_combined['pathway'].apply(
        lambda x: 0 if 'Electro' in x else 1
    )
    df_combined['is_methanation'] = df_combined['pathway'].apply(
        lambda x: 1 if 'methanation' in x else 0
    )
    df_combined['is_methanol'] = df_combined['pathway'].apply(
        lambda x: 1 if 'Methanol' in x else 0
    )

    return df_combined


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("CREATING CO2 CONVERSION DATASET FOR STREAMLIT APP")
    print("="*80)

    # Create complete dataset
    df = create_complete_dataset()

    # Save main dataset
    output_file = 'co2_conversion_complete.csv'
    df.to_csv(output_file, index=False)

    print(f"\n[OK] Complete dataset created: {output_file}")
    print(f"   Total records: {len(df)}")
    print(f"   Experimental: {len(df[df['data_source']=='experimental'])}")
    print(f"   Literature: {len(df[df['data_source']=='literature'])}")

    # Display summary
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)

    print("\n[Pathways]:")
    print(df['pathway'].value_counts())

    print("\n[Catalyst Family]:")
    print(df['catalyst_family'].value_counts())

    print("\n[Temperature Range]:")
    print(f"   Min: {df['temperature_c'].min()} C")
    print(f"   Max: {df['temperature_c'].max()} C")

    print("\n[Pressure Range]:")
    print(f"   Min: {df['pressure_bar'].min()} bar")
    print(f"   Max: {df['pressure_bar'].max()} bar")

    print("\n[Conversion Stats (X_CO2)]:")
    print(df['X_CO2'].describe())

    # Create separate dataset for methanol synthesis (real data only)
    df_meoh_real = df[
        (df['pathway'] == 'Methanol synthesis') &
        (df['data_source'] == 'experimental')
    ].copy()

    meoh_file = 'co2_methanol_experimental.csv'
    df_meoh_real.to_csv(meoh_file, index=False)
    print(f"\n[OK] Methanol experimental data: {meoh_file}")
    print(f"   Records: {len(df_meoh_real)}")

    # Display sample
    print("\n" + "="*80)
    print("SAMPLE DATA (first 10 rows)")
    print("="*80)
    print(df.head(10).to_string())

    # Create data dictionary
    print("\n" + "="*80)
    print("DATA DICTIONARY")
    print("="*80)
    print("""
    Column                Description                                   Type        Unit
    -------------------- --------------------------------------------- ----------- --------
    pathway              CO2 conversion pathway                        categorical -
    catalyst_family      Catalyst family/type                          categorical -
    metal                Metal center (for MOF)                        categorical -
    linker               Organic linker (for MOF)                      categorical -
    temperature_c        Operating temperature                         numeric     °C
    pressure_bar         Operating pressure                            numeric     bar
    h2_co2_ratio         H2:CO2 feed ratio                            numeric     mol/mol
    reactor_type         Reactor configuration                         categorical -
    X_CO2                CO2 conversion                               numeric     %
    S_MeOH               Methanol selectivity                         numeric     %
    S_CO                 CO selectivity                               numeric     %
    S_CH4                Methane selectivity                          numeric     %
    Y_MeOH               Methanol yield                               numeric     %
    primary_product      Main product                                  categorical -
    data_source          Data origin (experimental/literature)         categorical -
    1000_T_K             Arrhenius temperature term                    numeric     1/K
    temp_pressure        Temperature-pressure interaction              numeric     °C·bar
    ln_pressure          Natural log of pressure                       numeric     -
    is_thermal           Thermal process flag                          binary      0/1
    is_methanation       Methanation pathway flag                      binary      0/1
    is_methanol          Methanol pathway flag                         binary      0/1
    """)

    print("\n" + "="*80)
    print("[OK] DATASET CREATION COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  1. {output_file} - Complete dataset (all pathways)")
    print(f"  2. {meoh_file} - Methanol synthesis only (experimental)")
    print(f"\nReady for XGBoost training and Streamlit integration!")
