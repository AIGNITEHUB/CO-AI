
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from logic.forecast import forecast_best, filter_years, forecast_with_commitment
from logic.owid import load_owid_global_timeseries, list_regions, filter_country
from logic.conversion import predict_conversion_advanced, generate_catalyst_pathway_matrix, get_optimal_conditions_table
from logic.policy_commitments import (
    load_country_commitment_from_json,
    get_countries_from_json,
    validate_commitment_json
)

st.set_page_config(page_title="CO₂ AI Demo", layout="wide")

st.title("CO₂ AI Demo")

tab1, tab2 = st.tabs(["🌍 Emissions Forecast", "⚗️ CO₂ → Biofuels Predictor"])

# -------------------------
# TAB 1: Emissions Forecast
# -------------------------
with tab1:
    st.subheader("Filter & Forecast Global CO₂ Emissions (Real Data via OWID)")

    st.markdown("Upload a CSV (year, emissions_gtco2) or fetch the latest OWID dataset.")

    up = st.file_uploader("Upload CSV", type=["csv"])
    use_owid = st.checkbox("Fetch latest Our World in Data (OWID) dataset", value=True)
    selected_country = None

    if up:
        df = pd.read_csv(up)
    else:
        if use_owid:
            try:
                ow = load_owid_global_timeseries()
                countries = list_regions(ow)
                default_idx = countries.index("World") if "World" in countries else 0
                selected_country = st.selectbox("Country/Region (OWID)", options=countries, index=default_idx)
                df = filter_country(ow, selected_country)
            except Exception as e:
                st.warning(f"Could not fetch OWID data: {e}. Falling back to sample.")
                df = pd.read_csv("data/sample_co2_global.csv")
        else:
            df = pd.read_csv("data/sample_co2_global.csv")

    if not {"year", "emissions_gtco2"}.issubset(df.columns):
        st.error("CSV must include columns: year, emissions_gtco2")
    else:
        df["year"] = df["year"].astype(int)
        df = df.sort_values("year")

        col1, col2, col3 = st.columns(3)
        with col1:
            start_year = int(st.number_input("Start year", value=int(df["year"].min())))
        with col2:
            end_year = int(st.number_input("End year", value=int(df["year"].max())))
        with col3:
            degree = int(st.slider("Polynomial degree (for comparison)", min_value=1, max_value=4, value=2))

        fdf = filter_years(df, start_year, end_year)
        st.write(f"Rows selected: {len(fdf)}")

        fig1, ax1 = plt.subplots()
        ax1.plot(df["year"], df["emissions_gtco2"], marker="o", linewidth=1)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Emissions (GtCO₂)")
        ax1.set_title("Global CO₂ Emissions (historical)")
        st.pyplot(fig1)

        horizon = st.slider("Forecast horizon (years)", 1, 30, 20)
        try:
            res = forecast_best(fdf, degree=degree, horizon=horizon)
            fig2, ax2 = plt.subplots()
            ax2.plot(fdf["year"], fdf["emissions_gtco2"], marker="o", linewidth=1, label="History")
            ax2.plot(res.future_years, res.y_pred, marker="x", linestyle="--", label=f"Forecast — {res.model_name}")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Emissions (GtCO₂)")
            ax2.set_title("Forecast with automatic model selection")
            ax2.legend()
            st.pyplot(fig2)

            if res.diagnostics:
                colx, coly = st.columns(2)
                colx.metric("MAPE (Poly)", f"{res.diagnostics.get('poly_mape', float('nan')):.3f}")
                coly.metric("MAPE (ARIMA)", f"{res.diagnostics.get('arima_mape', float('nan')):.3f}")
                st.caption("Lower MAPE is better. Last-N backtest used.")

            out = pd.DataFrame({"year": res.future_years, "forecast_gtco2": np.round(res.y_pred, 3), "model": res.model_name})
            st.download_button("Download forecast CSV", out.to_csv(index=False), file_name="forecast_co2.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Could not forecast: {e}")

        # -------------------------
        # Commitment Scenario
        # -------------------------
        st.markdown("---")
        with st.expander("🎯 Add Net Zero / Commitment Scenario"):
            st.markdown("Compare Business-as-Usual (BAU) forecast with a commitment pathway (Net Zero, reduction targets, etc.)")

            # Initialize variables
            use_policy_tracking = False
            commitment_data = None
            json_content = None

            # Policy Tracking Option - Only for Vietnam
            if selected_country == "Vietnam":
                # Show mode selection radio buttons
                st.markdown("### 📋 Policy Tracking Mode")
                policy_mode = st.radio(
                    "Select policy tracking mode:",
                    ["None", "Forecast Mode", "Target Mode"],
                    index=0,
                    help="**Forecast Mode**: Project future emissions based on policy actions and historical trends (no specific target). **Target Mode**: Calculate pathway to achieve a specific emission reduction target.",
                    horizontal=True,
                    key="policy_mode_selection"
                )

                enable_policy_upload = policy_mode != "None"

                # Only show upload section if mode is selected
                if enable_policy_upload:
                    st.markdown("### 📤 Upload Policy Commitment Data")

                    if policy_mode == "Forecast Mode":
                        st.info("📊 **Forecast Mode**: Upload a JSON file with policy actions. The system will project emissions based on historical trends from your policy data. No target_reduction_pct needed at country level.")
                    elif policy_mode == "Target Mode":
                        st.info("🎯 **Target Mode**: Upload a JSON file with policy actions and target_reduction_pct. The system will calculate the pathway to achieve your specific emission reduction target.")

                    uploaded_json = st.file_uploader(
                        "Upload country commitment JSON file",
                        type=["json"],
                        help=f"Upload a JSON file for {policy_mode}",
                        key="policy_json_upload"
                    )

                    if uploaded_json is not None:
                        try:
                            # Read JSON content
                            json_content = uploaded_json.read().decode('utf-8')

                            # Validate JSON
                            is_valid, validation_msg = validate_commitment_json(json_content)

                            if not is_valid:
                                st.error(f"❌ Invalid JSON: {validation_msg}")
                            else:
                                # Additional validation: Check if mode matches selection
                                import json
                                json_data = json.loads(json_content)
                                mode_mismatch = False
                                mode_error_msg = ""

                                for country_name, country_data in json_data.items():
                                    json_mode = country_data.get('mode', 'target')
                                    expected_mode = 'forecast' if policy_mode == "Forecast Mode" else 'target'

                                    if json_mode != expected_mode:
                                        mode_mismatch = True
                                        mode_error_msg = f"Country '{country_name}' has mode '{json_mode}' but you selected '{policy_mode}'. Please select the correct mode or update your JSON file."
                                        break

                                if mode_mismatch:
                                    st.error(f"❌ Mode mismatch: {mode_error_msg}")
                                elif not mode_mismatch:
                                    st.success("✓ JSON structure is valid and mode matches")

                                    # Get available countries from JSON
                                    available_countries = get_countries_from_json(json_content)

                                if available_countries and not mode_mismatch:
                                    st.info(f"📋 Found {len(available_countries)} country/countries: {', '.join(available_countries)}")

                                    # Let user select country from JSON
                                    policy_country = st.selectbox(
                                        "Select country from uploaded data",
                                        options=available_countries,
                                        key="policy_country_select"
                                    )

                                    # Use policy tracking automatically when file is uploaded
                                    use_policy_tracking = True

                                    try:
                                        commitment_data = load_country_commitment_from_json(json_content, policy_country)
                                        st.success(f"✓ Loaded {len(commitment_data.policy_actions)} policy actions for {policy_country}")

                                        # Display summary based on mode
                                        if commitment_data.is_forecast_mode:
                                            # Forecast Mode: Show projection parameters
                                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                                            with col_sum1:
                                                st.metric("Projection Year", commitment_data.target_year)
                                            with col_sum2:
                                                st.metric("Baseline Emissions", f"{commitment_data.baseline_emissions_gtco2:.3f} GtCO₂")
                                            with col_sum3:
                                                st.metric("Baseline Year", commitment_data.baseline_year)
                                        else:
                                            # Target Mode: Show target parameters
                                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                                            with col_sum1:
                                                st.metric("Target Year", commitment_data.target_year)
                                            with col_sum2:
                                                st.metric("Target Reduction", f"{commitment_data.target_reduction_pct}%")
                                            with col_sum3:
                                                st.metric("Baseline Year", commitment_data.baseline_year)

                                        # Check for data gaps (policy data doesn't extend to target year)
                                        max_data_year = max(
                                            max(action.implementation_years) if action.implementation_years else 0
                                            for action in commitment_data.policy_actions
                                        )
                                        data_gap_years = commitment_data.target_year - max_data_year

                                        if data_gap_years > 5:
                                            st.warning(
                                                f"⚠️ **Data gap detected**: Policy data extends to {max_data_year}, "
                                                f"but target year is {commitment_data.target_year} ({data_gap_years} year gap). \n\n"
                                                f"**Auto-extrapolation will be applied**: The pathway will automatically extrapolate "
                                                f"from the last data point ({max_data_year}) to the target year using linear progression. "
                                                f"For more accurate projections, consider adding policy milestones for years {max_data_year+1}-{commitment_data.target_year}."
                                            )
                                        elif data_gap_years > 0:
                                            st.info(
                                                f"ℹ️ Policy data extends to {max_data_year}. "
                                                f"Values for {max_data_year+1}-{commitment_data.target_year} will be automatically extrapolated."
                                            )

                                    except Exception as e:
                                        st.error(f"Could not load policy data: {e}")
                                        use_policy_tracking = False
                                        commitment_data = None
                                else:
                                    st.warning("No countries found in JSON file")

                        except Exception as e:
                            st.error(f"Error reading JSON file: {e}")
                    else:
                        st.info("💡 Upload a JSON file to enable policy-driven pathway based on sector-specific actions")

            # Show info message if policy tracking is enabled
            if use_policy_tracking and commitment_data:
                if commitment_data.is_forecast_mode:
                    st.info(f"📊 Forecast Mode enabled: Projecting emissions for {commitment_data.country} based on policy actions and historical trends. Projection year and pathway are fixed from uploaded data.")
                else:
                    st.info(f"🎯 Target Mode enabled: Using parameters from uploaded JSON ({commitment_data.country}). Target year, reduction %, and pathway are fixed based on policy data.")

            # Inputs outside form for realtime updates
            colC1, colC2, colC3 = st.columns(3)

            with colC1:
                # Pre-fill from commitment data if available
                default_target_year = commitment_data.target_year if (use_policy_tracking and commitment_data) else 2050

                # Different labels for forecast vs target mode
                year_label = "Projection year" if (use_policy_tracking and commitment_data and commitment_data.is_forecast_mode) else "Target year"
                year_help = "Fixed from JSON - projection endpoint" if (use_policy_tracking and commitment_data and commitment_data.is_forecast_mode) else \
                           ("Fixed from JSON when policy tracking enabled" if (use_policy_tracking and commitment_data) else "Year to achieve target emissions")

                target_year = int(st.number_input(
                    year_label,
                    value=default_target_year,
                    min_value=int(fdf["year"].max())+1,
                    max_value=2100,
                    step=1,
                    key="target_year",
                    disabled=(use_policy_tracking and commitment_data is not None),
                    help=year_help
                ))

                # Commitment type based on mode
                if use_policy_tracking and commitment_data:
                    if commitment_data.is_forecast_mode:
                        commitment_type = "Policy-driven Projection"
                        st.selectbox("Mode", [
                            "Policy-driven Projection"
                        ], key="commitment_type", disabled=True,
                        help="Forecast mode: Projects emissions based on policy actions and historical trends")
                    else:
                        commitment_type = "Percentage Reduction"
                        st.selectbox("Commitment type", [
                            "Percentage Reduction"
                        ], key="commitment_type", disabled=True,
                        help="Fixed to 'Percentage Reduction' when using target mode policy data")
                else:
                    commitment_type = st.selectbox("Commitment type", [
                        "Net Zero (0 emissions)",
                        "Percentage Reduction",
                        "Absolute Target"
                    ], key="commitment_type")

            with colC2:
                current_emissions = float(fdf.iloc[-1]["emissions_gtco2"])

                if use_policy_tracking and commitment_data and commitment_data.is_forecast_mode:
                    # FORECAST MODE: Show projected emissions (calculated from policy actions)
                    # Calculate projected emissions at target year based on policy reductions
                    reduction_fraction = commitment_data.calculate_annual_reduction_fraction(target_year)
                    baseline_emissions = commitment_data.baseline_emissions_gtco2
                    projected_emissions = baseline_emissions * (1 - reduction_fraction)

                    st.metric(
                        "Projected Emissions",
                        f"{projected_emissions:.3f} GtCO₂",
                        delta=f"{-reduction_fraction*100:.1f}% from baseline",
                        help=f"Calculated from policy actions at {target_year}. Baseline: {baseline_emissions:.3f} GtCO₂"
                    )
                    # Set target_emissions for internal calculations
                    target_emissions = projected_emissions
                else:
                    # TARGET MODE or no policy tracking
                    # Pre-fill reduction from commitment data if available
                    if use_policy_tracking and commitment_data and commitment_type == "Percentage Reduction":
                        default_reduction = commitment_data.target_reduction_pct
                    else:
                        default_reduction = 80.0

                    if commitment_type == "Net Zero (0 emissions)":
                        target_emissions = 0.0
                        st.metric("Target emissions", f"{target_emissions} GtCO₂")
                    elif commitment_type == "Percentage Reduction":
                        reduction_pct = st.slider(
                            "Reduction from baseline (%)",
                            0.0, 100.0,
                            float(default_reduction),
                            step=5.0,
                            key="reduction_pct",
                            disabled=(use_policy_tracking and commitment_data is not None),
                            help="Fixed from JSON when policy tracking enabled" if (use_policy_tracking and commitment_data) else "Percentage reduction from baseline emissions"
                        )
                        # Calculate from baseline if commitment data available, else from current
                        if use_policy_tracking and commitment_data:
                            baseline_emissions = commitment_data.baseline_emissions_gtco2
                            target_emissions = baseline_emissions * (1 - reduction_pct / 100)
                            st.metric("Target emissions", f"{target_emissions:.3f} GtCO₂",
                                    help=f"Based on baseline: {baseline_emissions:.3f} GtCO₂")
                        else:
                            target_emissions = current_emissions * (1 - reduction_pct / 100)
                            st.metric("Target emissions", f"{target_emissions:.2f} GtCO₂")
                    else:  # Absolute Target
                        target_emissions = float(st.number_input("Target emissions (GtCO₂)", value=10.0, min_value=0.0, step=0.5, key="target_emissions"))
                        st.metric("Target emissions", f"{target_emissions:.2f} GtCO₂")

            with colC3:
                # Force Policy-driven when policy tracking is enabled
                if use_policy_tracking and commitment_data:
                    if commitment_data.is_forecast_mode:
                        pathway_shape = st.selectbox(
                            "Projection pathway",
                            ["Policy-driven projection"],
                            key="pathway_shape",
                            disabled=True,
                            help="Forecast mode: Pathway calculated from policy actions and historical trends"
                        )
                    else:
                        pathway_shape = st.selectbox(
                            "Reduction pathway shape",
                            ["Policy-driven (based on actual actions)"],
                            key="pathway_shape",
                            disabled=True,
                            help="Target mode: Fixed to 'Policy-driven' when using policy data"
                        )
                    pathway_type = "policy_driven"
                else:
                    # Add policy-driven option if policy tracking is enabled
                    pathway_options = [
                        "Exponential decay (realistic)",
                        "Linear reduction",
                        "S-curve (slow-fast-slow)",
                        "Custom milestones"
                    ]
                    pathway_map = {
                        "Exponential decay (realistic)": "exponential",
                        "Linear reduction": "linear",
                        "S-curve (slow-fast-slow)": "scurve",
                        "Custom milestones": "milestones"
                    }

                    pathway_shape = st.selectbox("Reduction pathway shape", pathway_options, key="pathway_shape")
                    pathway_type = pathway_map[pathway_shape]

            # Milestones input (only show if not using policy tracking)
            milestones = None
            if pathway_type == "milestones" and not (use_policy_tracking and commitment_data):
                st.markdown("**Add intermediate milestones** (year: emissions)")
                n_milestones = st.number_input("Number of milestones", min_value=1, max_value=5, value=2, step=1, key="n_milestones")
                milestones = {}
                cols_m = st.columns(int(n_milestones))
                for i in range(int(n_milestones)):
                    with cols_m[i]:
                        m_year = int(st.number_input(f"Year {i+1}", value=int(fdf["year"].max()) + 10*(i+1), min_value=int(fdf["year"].max())+1, max_value=target_year-1, key=f"m_year_{i}"))
                        m_emissions = float(st.number_input(f"Emissions {i+1} (GtCO₂)", value=current_emissions * (1 - 0.2*(i+1)), min_value=0.0, key=f"m_emis_{i}"))
                        milestones[m_year] = m_emissions

            # Display sector breakdown if policy tracking enabled
            if use_policy_tracking and commitment_data and pathway_type == "policy_driven":
                st.markdown("---")
                st.markdown(f"### 📊 {policy_country if 'policy_country' in locals() else 'Country'} - Policy Actions Breakdown")

                # Create tabs for each sector
                sector_tabs = st.tabs([f"{action.sector} ({action.share_pct:.1f}%)" for action in commitment_data.policy_actions])

                for idx, action in enumerate(commitment_data.policy_actions):
                    with sector_tabs[idx]:
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Baseline Emissions",
                                f"{action.baseline_emissions_mtco2:.1f} Mt CO₂",
                                help=f"Baseline year: {action.baseline_year}"
                            )

                        with col2:
                            st.metric(
                                "Sector Share",
                                f"{action.share_pct:.2f}%",
                                help="Percentage of total emissions from this sector"
                            )

                        with col3:
                            # Show appropriate label based on sector type
                            target_label = "Removal Increase" if action.baseline_emissions_mtco2 < 0 else "Reduction Target"
                            target_help = "Target increase in removal capacity" if action.baseline_emissions_mtco2 < 0 else "Target reduction from baseline"
                            st.metric(
                                target_label,
                                f"{action.target_pct}%",
                                help=target_help
                            )

                        with col4:
                            # Year-over-year improvement
                            if len(action.yearly_improvement_pct) >= 2:
                                latest = action.yearly_improvement_pct[-1]
                                previous = action.yearly_improvement_pct[-2]
                                yoy_change = latest - previous

                                st.metric(
                                    f"Progress ({action.implementation_years[-1]})",
                                    f"{latest:.1f}%",
                                    delta=f"{yoy_change:+.1f}% YoY",
                                    help="Cumulative reduction achieved"
                                )
                            elif action.yearly_improvement_pct:
                                st.metric(
                                    f"Progress ({action.implementation_years[0]})",
                                    f"{action.yearly_improvement_pct[0]:.1f}%",
                                    help="Cumulative reduction achieved"
                                )

                        # Status indicator
                        status_colors = {
                            "On track": "🟢",
                            "Behind schedule": "🟡",
                            "Ahead": "🔵"
                        }
                        st.markdown(f"{status_colors.get(action.status, '⚪')} **Status**: {action.status}")
                        st.caption(f"**Action**: {action.action_name}")

                        # Progress visualization
                        if action.implementation_years and action.yearly_improvement_pct:
                            progress_df = pd.DataFrame({
                                'Year': action.implementation_years,
                                'Cumulative Reduction (%)': action.yearly_improvement_pct
                            })

                            fig_prog, ax_prog = plt.subplots(figsize=(8, 3))
                            ax_prog.plot(progress_df['Year'], progress_df['Cumulative Reduction (%)'],
                                       marker='o', linewidth=2, markersize=6, color='green')
                            ax_prog.axhline(action.target_pct, color='red', linestyle='--',
                                          linewidth=1, alpha=0.7, label=f'Target ({action.target_pct}%)')
                            ax_prog.set_xlabel('Year')
                            # Update ylabel based on sector type
                            ylabel = 'Cumulative Increase (%)' if action.baseline_emissions_mtco2 < 0 else 'Cumulative Reduction (%)'
                            ax_prog.set_ylabel(ylabel)
                            ax_prog.set_title(f'{action.sector} - Progress Tracking')
                            ax_prog.legend()
                            ax_prog.grid(alpha=0.3)
                            st.pyplot(fig_prog)
                            plt.close()

            # Generate button with mode-specific text
            if use_policy_tracking and commitment_data and commitment_data.is_forecast_mode:
                button_text = "📊 Generate emissions projection"
            else:
                button_text = "🚀 Generate commitment scenario"

            submitted = st.button(button_text, type="primary", use_container_width=True, key="generate_commitment")

            # Only run when button is clicked
            if submitted:
                try:
                    with st.spinner("Generating commitment scenario..."):
                        commitment_res = forecast_with_commitment(
                            df=fdf,
                            target_year=target_year,
                            target_emissions=target_emissions,
                            pathway_type=pathway_type,
                            milestones=milestones,
                            bau_degree=degree,
                            country_commitment=commitment_data if pathway_type == "policy_driven" else None
                        )

                    # Save to session state to persist after download
                    st.session_state['commitment_res'] = commitment_res
                    st.session_state['commitment_params'] = {
                        'pathway_shape': pathway_shape,
                        'target_emissions': target_emissions,
                        'target_year': target_year,
                        'selected_country': selected_country,
                        'milestones': milestones,
                        'is_forecast_mode': commitment_data.is_forecast_mode if (use_policy_tracking and commitment_data) else False
                    }

                    if use_policy_tracking and commitment_data and commitment_data.is_forecast_mode:
                        st.success(f"📊 Emissions projection generated: {pathway_shape} projecting {target_emissions:.3f} GtCO₂ by {target_year}")
                    else:
                        st.success(f"🎯 Commitment scenario generated: {pathway_shape} to {target_emissions:.1f} GtCO₂ by {target_year}")

                except Exception as e:
                    st.error(f"Could not generate commitment scenario: {e}")

            # Clear results button
            if 'commitment_res' in st.session_state and st.session_state['commitment_res'] is not None:
                if st.button("🗑️ Clear results", key="clear_commitment"):
                    st.session_state['commitment_res'] = None
                    st.session_state['commitment_params'] = None
                    st.rerun()

            # Display results from session state (persists after download)
            if 'commitment_res' in st.session_state and st.session_state['commitment_res'] is not None:
                commitment_res = st.session_state['commitment_res']
                params = st.session_state.get('commitment_params', {})
                pathway_shape = params.get('pathway_shape', pathway_shape)
                target_emissions = params.get('target_emissions', target_emissions)
                target_year = params.get('target_year', target_year)
                saved_country = params.get('selected_country', selected_country)
                milestones = params.get('milestones', milestones)

                # Visualization
                fig3, ax3 = plt.subplots(figsize=(12, 6))

                # Historical
                ax3.plot(fdf["year"], fdf["emissions_gtco2"], 'o-', label='Historical', linewidth=2, color='black', markersize=4)

                # BAU forecast
                ax3.plot(commitment_res.future_years, commitment_res.bau_forecast, 's--',
                        label=f'Business as Usual ({commitment_res.bau_model_name})',
                        color='red', linewidth=2, alpha=0.8, markersize=5)

                # Commitment pathway
                ax3.plot(commitment_res.future_years, commitment_res.commitment_forecast, '^-',
                        label=f'Commitment Pathway ({pathway_shape})',
                        color='green', linewidth=2.5, markersize=5)

                # Emissions gap (shaded area)
                ax3.fill_between(commitment_res.future_years, commitment_res.bau_forecast,
                                commitment_res.commitment_forecast,
                                alpha=0.25, color='orange', label='Emissions Gap (reduction needed)')

                # Target point
                ax3.scatter(target_year, target_emissions, s=300, marker='*',
                           color='darkgreen', zorder=10, edgecolors='black', linewidths=1.5,
                           label=f'Target: {target_emissions:.1f} GtCO₂ in {target_year}')

                # Milestones
                if milestones:
                    milestone_years = list(milestones.keys())
                    milestone_vals = list(milestones.values())
                    ax3.scatter(milestone_years, milestone_vals, s=150, marker='D',
                               color='blue', zorder=9, edgecolors='black', linewidths=1,
                               label='Milestones')

                # Zero line
                ax3.axhline(0, color='black', linestyle=':', alpha=0.4, linewidth=1)

                ax3.set_xlabel("Year", fontsize=12)
                ax3.set_ylabel("Emissions (GtCO₂)", fontsize=12)
                ax3.set_title(f"Emissions Forecast: BAU vs. Commitment Scenario\n{saved_country if saved_country else 'Global'}", fontsize=14, fontweight='bold')
                ax3.legend(loc='best', fontsize=10)
                ax3.grid(alpha=0.3, linestyle='--')

                st.pyplot(fig3)

                # Metrics
                st.markdown("### 📊 Scenario Analysis")
                colM1, colM2, colM3, colM4 = st.columns(4)

                with colM1:
                    st.metric("Current emissions", f"{commitment_res.current_emissions:.2f} GtCO₂")

                with colM2:
                    total_gap = float(np.sum(commitment_res.emissions_gap))
                    st.metric("Total emissions gap", f"{total_gap:.1f} GtCO₂",
                             help="Cumulative reduction needed vs. BAU")

                with colM3:
                    avg_annual_reduction = total_gap / len(commitment_res.future_years)
                    st.metric("Avg. annual reduction needed", f"{avg_annual_reduction:.2f} GtCO₂/yr")

                with colM4:
                    reduction_rate = ((commitment_res.current_emissions - target_emissions) / commitment_res.current_emissions) * 100
                    st.metric("Total reduction required", f"{reduction_rate:.1f}%")

                # Download
                commitment_df = pd.DataFrame({
                    "year": commitment_res.future_years,
                    "bau_forecast_gtco2": np.round(commitment_res.bau_forecast, 3),
                    "commitment_forecast_gtco2": np.round(commitment_res.commitment_forecast, 3),
                    "emissions_gap_gtco2": np.round(commitment_res.emissions_gap, 3),
                })
                st.download_button("Download commitment scenario CSV",
                                  commitment_df.to_csv(index=False),
                                  file_name="commitment_scenario.csv",
                                  mime="text/csv",
                                  key="download_commitment_csv")

# -------------------------
# TAB 2: Conversion Predictor
# -------------------------
with tab2:
    st.subheader("CO₂ → Biofuels Conversion Predictor (Demo)")
    st.markdown("This is a heuristic educational demo — not scientific.")

    # -------------------------
    # Reference Tables
    # -------------------------

    # Catalyst-Pathway Performance Matrix
    with st.expander("📊 Catalyst-Pathway Performance Matrix"):
        st.markdown("**Comprehensive comparison of all pathway × catalyst combinations at optimal conditions**")
        st.caption("Shows theoretical yield (%) for each combination. N/A indicates incompatible pairs.")

        if st.button("🔄 Generate Performance Matrix", key="gen_matrix"):
            with st.spinner("Generating catalyst-pathway matrix..."):
                try:
                    matrix_df = generate_catalyst_pathway_matrix()

                    st.success("Matrix generated successfully!")

                    # Plain table with better formatting
                    display_df = matrix_df.copy()
                    for col in display_df.columns:
                        if col != 'Pathway':
                            display_df[col] = display_df[col].apply(lambda x: 'N/A' if pd.isna(x) else f'{x:.1f}%')

                    st.dataframe(display_df, use_container_width=True, height=280)

                    # Summary statistics
                    st.markdown("**📈 Summary Statistics:**")
                    col_sum1, col_sum2, col_sum3 = st.columns(3)

                    # Find best overall combination
                    max_val = -1
                    best_pathway = ""
                    best_catalyst = ""
                    for col in matrix_df.columns:
                        if col != 'Pathway':
                            col_max = matrix_df[col].max()
                            if not pd.isna(col_max) and col_max > max_val:
                                max_val = col_max
                                best_catalyst = col
                                best_pathway = matrix_df[matrix_df[col] == col_max]['Pathway'].values[0]

                    with col_sum1:
                        st.metric("Best Combination", f"{best_pathway[:20]}... + {best_catalyst}", f"{max_val:.1f}%")

                    with col_sum2:
                        # Average yield (excluding N/A)
                        all_vals = []
                        for col in matrix_df.columns:
                            if col != 'Pathway':
                                all_vals.extend(matrix_df[col].dropna().tolist())
                        avg_yield = sum(all_vals) / len(all_vals) if all_vals else 0
                        st.metric("Average Yield", f"{avg_yield:.1f}%")

                    with col_sum3:
                        # Count valid combinations
                        valid_count = sum([len(matrix_df[col].dropna()) for col in matrix_df.columns if col != 'Pathway'])
                        total_count = len(matrix_df) * (len(matrix_df.columns) - 1)
                        st.metric("Valid Combinations", f"{valid_count}/{total_count}")


                    # Download button
                    csv_matrix = matrix_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Matrix CSV",
                        csv_matrix,
                        file_name="catalyst_pathway_matrix.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error generating matrix: {e}")

    # Optimal Operating Conditions Table
    with st.expander("⚙️ Optimal Operating Conditions Reference"):
        st.markdown("**Quick reference for optimal conditions and best catalysts for each pathway**")
        st.caption("Shows temperature, pressure, H₂:CO₂ ratio ranges and expected max yields.")

        if st.button("🔄 Generate Conditions Table", key="gen_conditions"):
            with st.spinner("Generating optimal conditions table..."):
                try:
                    conditions_df = get_optimal_conditions_table()

                    st.success("Conditions table generated successfully!")

                    # Display the table
                    st.dataframe(conditions_df, use_container_width=True)

                    # Interpretation guide
                    st.markdown("**📖 Key insights:**")

                    # Find highest and lowest yields
                    max_row = conditions_df.loc[conditions_df['Max Yield (%)'].idxmax()]
                    min_row = conditions_df.loc[conditions_df['Max Yield (%)'].idxmin()]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.info(f"""
                        **🏆 Highest Yield**
                        {max_row['Pathway']}
                        - Yield: {max_row['Max Yield (%)']}%
                        - Catalyst: {max_row['Best Catalyst']}
                        - Product: {max_row['Primary Product']}
                        """)

                    with col2:
                        st.warning(f"""
                        **⚠️ Lowest Yield** (improvement opportunity)
                        {min_row['Pathway']}
                        - Yield: {min_row['Max Yield (%)']}%
                        - Catalyst: {min_row['Best Catalyst']}
                        - Product: {min_row['Primary Product']}
                        """)

                    # Download button
                    csv_conditions = conditions_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Conditions CSV",
                        csv_conditions,
                        file_name="optimal_conditions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error generating conditions table: {e}")

    st.markdown("---")

    # -------------------------
    # Single Prediction Tool
    # -------------------------
    st.markdown("### 🧪 Single Prediction Tool")

    colA, colB, colC = st.columns(3)
    with colA:
        pathway = st.selectbox("Pathway", [
            "Sabatier (methanation)",
            "RWGS + Fischer-Tropsch",
            "Methanol synthesis",
            "Electroreduction (formate route)",
            "Electroreduction (CO route)",
        ])
        catalyst = st.selectbox("Catalyst family", [
            "Ni-based",
            "Cu/ZnO/Al2O3",
            "Fe/Co (FT)",
            "Cu (electro)",
            "Ag/Au (electro)",
            "Other",
        ])
    with colB:
        temperature_c = st.number_input("Temperature (°C)", value=300.0, step=5.0)
        pressure_bar = st.number_input("Pressure (bar)", value=10.0, step=0.5)
    with colC:
        h2_to_co2 = st.number_input("H₂:CO₂ feed ratio", value=3.0, step=0.1)
        reactor_type = st.selectbox("Reactor type", ["Fixed-bed", "Slurry", "Electrochemical (flow cell)", "Other"])

    grid_ci = st.slider("Grid carbon intensity (gCO₂/kWh) – for electro routes", min_value=0, max_value=900, value=450, step=10)

    if st.button("Predict conversion"):
        out = predict_conversion_advanced(
            pathway=pathway,
            catalyst_family=catalyst,
            temperature_c=float(temperature_c),
            pressure_bar=float(pressure_bar),
            h2_to_co2=float(h2_to_co2),
            reactor_type=reactor_type,
            grid_ci_gco2_per_kwh=float(grid_ci),
        )
        st.success("Prediction complete.")

        st.json({"primary_product": out["primary_product"], "yield_percent_estimate": out["yield_percent_estimate"], "yield_percent_range": out["yield_percent_range"]})

        slate_df = pd.DataFrame([out["product_slate_selectivity"]]).T.reset_index()
        slate_df.columns = ["Product", "Selectivity (%)"]
        st.caption("Product slate (selectivity, demo)")
        st.dataframe(slate_df, use_container_width=True)

        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("Estimated H₂ per CO₂ (mol/mol)", out["estimated_H2_per_CO2_mol"])
        colm2.metric("Carbon utilization (proxy)", f"{out['carbon_utilization_proxy_percent']}%")
        sc = out["operating_scores"]
        colm3.metric("Op. score (avg)", f"{round((sc['temperature_score']+sc['pressure_score']+sc['h2_ratio_score'])/3,3)}")

        if out.get("suggestions"):
            st.markdown("**Suggestions**")
            for s in out["suggestions"]:
                st.write("- " + s)

        if out.get("notes"):
            st.markdown("**Notes**")
            for n in out["notes"]:
                st.caption("• " + n)

        exp = pd.DataFrame([{
            "pathway": pathway, "catalyst": catalyst, "temperature_c": temperature_c, "pressure_bar": pressure_bar,
            "h2_to_co2": h2_to_co2, "reactor_type": reactor_type, "grid_ci_gco2_per_kwh": grid_ci, **out
        }])
        st.download_button("Download prediction CSV", exp.to_csv(index=False), file_name="co2_conversion_prediction.csv", mime="text/csv")

    # --- Optimization (demo) ---
    with st.expander("🔎 Grid search optimizer (demo)"):
        st.caption("Search T–P–H₂:CO₂ grid to maximize a chosen objective (demo heuristic).")

        obj = st.selectbox("Objective", [
            "Maximize yield",
            "Maximize CH3OH selectivity",
            "Maximize CH4 selectivity",
            "Maximize CO selectivity",
            "Maximize C5+ selectivity",
            "Maximize HCOO/HCOOH selectivity",
        ], index=0)

        colr1, colr2, colr3 = st.columns(3)
        with colr1:
            t_min = st.number_input("T min (°C)", value=250.0, step=5.0)
            t_max = st.number_input("T max (°C)", value=380.0, step=5.0)
            t_steps = st.slider("T steps", 3, 15, 7)
        with colr2:
            p_min = st.number_input("P min (bar)", value=1.0, step=0.5)
            p_max = st.number_input("P max (bar)", value=60.0, step=0.5)
            p_steps = st.slider("P steps", 3, 15, 7)
        with colr3:
            h_min = st.number_input("H₂:CO₂ min", value=1.0, step=0.1)
            h_max = st.number_input("H₂:CO₂ max", value=5.0, step=0.1)
            h_steps = st.slider("H₂:CO₂ steps", 3, 15, 7)

        total = t_steps * p_steps * h_steps
        st.write(f"Grid size: {total} points")
        if total > 2500:
            st.warning("Grid too large for demo (limit 2500). Reduce steps.")
        else:
            if st.button("Run grid search"):
                Ts = np.linspace(t_min, t_max, t_steps)
                Ps = np.linspace(p_min, p_max, p_steps)
                Hs = np.linspace(h_min, h_max, h_steps)

                rows = []
                for T in Ts:
                    for P in Ps:
                        for H in Hs:
                            res = predict_conversion_advanced(
                                pathway=pathway, catalyst_family=catalyst, temperature_c=float(T),
                                pressure_bar=float(P), h2_to_co2=float(H), reactor_type=reactor_type,
                                grid_ci_gco2_per_kwh=float(grid_ci),
                            )
                            slate = res.get("product_slate_selectivity", {})
                            if obj == "Maximize yield":
                                score = res["yield_percent_estimate"]
                            elif obj == "Maximize CH3OH selectivity":
                                score = slate.get("CH3OH", 0.0)
                            elif obj == "Maximize CH4 selectivity":
                                score = slate.get("CH4", 0.0)
                            elif obj == "Maximize CO selectivity":
                                score = slate.get("CO", 0.0)
                            elif obj == "Maximize C5+ selectivity":
                                score = slate.get("C5+", 0.0)
                            elif obj == "Maximize HCOO/HCOOH selectivity":
                                score = slate.get("HCOO/HCOOH", 0.0)
                            else:
                                score = res["yield_percent_estimate"]

                            rows.append({
                                "temperature_c": round(float(T), 3),
                                "pressure_bar": round(float(P), 3),
                                "h2_to_co2": round(float(H), 3),
                                "yield_%": res["yield_percent_estimate"],
                                "yield_range": res["yield_percent_range"],
                                "primary_product": res["primary_product"],
                                "carbon_util_%": res["carbon_utilization_proxy_percent"],
                                "sel_CH3OH_%": slate.get("CH3OH", 0.0),
                                "sel_CH4_%": slate.get("CH4", 0.0),
                                "sel_CO_%": slate.get("CO", 0.0),
                                "sel_C5+_%": slate.get("C5+", 0.0),
                                "sel_HCOO/HCOOH_%": slate.get("HCOO/HCOOH", 0.0),
                                "objective_score": score,
                            })

                grid_df = pd.DataFrame(rows).sort_values("objective_score", ascending=False).reset_index(drop=True)
                st.success("Grid search done.")
                st.write("Top 10 results:")
                st.dataframe(grid_df.head(10), use_container_width=True)

                st.download_button("Download full grid results (CSV)", grid_df.to_csv(index=False), file_name="conversion_grid_results.csv", mime="text/csv")

                try:
                    h_mid = float(Hs[len(Hs)//2])
                    slice_df = grid_df[np.isclose(grid_df["h2_to_co2"], h_mid, atol=1e-6)].copy()
                    piv = slice_df.pivot_table(index="temperature_c", columns="pressure_bar", values="yield_%", aggfunc="max")
                    fig, ax = plt.subplots()
                    im = ax.imshow(piv.values, aspect="auto", origin="lower")
                    ax.set_xticks(range(len(piv.columns)))
                    ax.set_yticks(range(len(piv.index)))
                    ax.set_xticklabels([f"{c:.0f}" for c in piv.columns])
                    ax.set_yticklabels([f"{r:.0f}" for r in piv.index])
                    ax.set_xlabel("Pressure (bar)")
                    ax.set_ylabel("Temperature (°C)")
                    ax.set_title(f"Yield (%) heatmap at H₂:CO₂ = {h_mid:.2f}")
                    st.pyplot(fig)
                except Exception as e:
                    st.caption(f"Heatmap not available: {e}")
