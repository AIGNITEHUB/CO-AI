
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logic.forecast import forecast_best, filter_years
from logic.owid import load_owid_global_timeseries, list_regions, filter_country
from logic.conversion import predict_product_and_yield

st.set_page_config(page_title="CO₂ AI Demo", layout="wide")

st.title("CO₂ AI Demo (Basic)")

tab1, tab2 = st.tabs(["🌍 Emissions Forecast", "⚗️ CO₂ → Biofuels Predictor"])

with tab1:
    st.subheader("Filter & Forecast Global CO₂ Emissions (Real Data via OWID)")

    st.markdown(
        "Upload a CSV with columns **year, emissions_gtco2**, or use the sample dataset. "
        "Then set a forecast horizon and polynomial degree."
    )
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

    # Basic checks
    if not {'year', 'emissions_gtco2'}.issubset(df.columns):
        st.error("CSV must include columns: year, emissions_gtco2")
    else:
        df['year'] = df['year'].astype(int)
        df = df.sort_values('year')

        col1, col2, col3 = st.columns(3)
        with col1:
            start_year = int(st.number_input("Start year", value=int(df['year'].min())))
        with col2:
            end_year = int(st.number_input("End year", value=int(df['year'].max())))
        with col3:
            degree = int(st.slider("Polynomial degree", min_value=1, max_value=4, value=2))

        fdf = filter_years(df, start_year, end_year)
        st.write(f"Rows selected: {len(fdf)}")

        # Plot historical
        fig1, ax1 = plt.subplots()
        ax1.plot(df['year'], df['emissions_gtco2'], marker="o", linewidth=1)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Emissions (GtCO₂)")
        ax1.set_title("Global CO₂ Emissions (historical sample)")
        st.pyplot(fig1)

        # Forecast
        horizon = st.slider("Forecast horizon (years)", 1, 30, 20)
        try:
            res = forecast_best(fdf, degree=degree, horizon=horizon)
            fig2, ax2 = plt.subplots()
            ax2.plot(fdf['year'], fdf['emissions_gtco2'], marker="o", linewidth=1, label="History")
            ax2.plot(res.future_years, res.y_pred, marker="x", linestyle="--", label="Forecast")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Emissions (GtCO₂)")
            ax2.set_title("Forecast with automatic model selection")
            ax2.legend()
            st.pyplot(fig2)

            # Export
            out = pd.DataFrame({"year": res.future_years, "forecast_gtco2": np.round(res.y_pred, 3), "model": res.model_name})
            st.download_button("Download forecast CSV", out.to_csv(index=False), file_name="forecast_co2.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Could not forecast: {e}")

with tab2:
    st.subheader("CO₂ → Biofuels Conversion Predictor (Demo)")
    st.markdown("This is a heuristic educational demo — not scientific.")

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
        out = predict_product_and_yield(
            pathway=pathway,
            catalyst_family=catalyst,
            temperature_c=float(temperature_c),
            pressure_bar=float(pressure_bar),
            h2_to_co2=float(h2_to_co2),
            reactor_type=reactor_type,
            grid_ci_gco2_per_kwh=float(grid_ci),
        )
        st.success("Prediction complete.")
        st.json(out)

        # Simple "efficiency" panel (toy C-utilization notion)
        cue = max(0.05, min(0.95, out['yield_percent_estimate']/100 * (1.05 if "Electro" in pathway and grid_ci < 150 else 1.0)))
        st.metric("Carbon Utilization (toy)", f"{round(100*cue,1)}%")

        # Allow export
        exp = pd.DataFrame([{
            "pathway": pathway,
            "catalyst": catalyst,
            "temperature_c": temperature_c,
            "pressure_bar": pressure_bar,
            "h2_to_co2": h2_to_co2,
            "reactor_type": reactor_type,
            "grid_ci_gco2_per_kwh": grid_ci,
            **out
        }])
        st.download_button("Download prediction CSV", exp.to_csv(index=False), file_name="co2_conversion_prediction.csv", mime="text/csv")
