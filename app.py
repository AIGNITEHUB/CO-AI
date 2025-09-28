
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from logic.forecast import forecast_best, filter_years
from logic.owid import load_owid_global_timeseries, list_regions, filter_country
from logic.conversion import predict_conversion_advanced

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
# TAB 2: Conversion Predictor
# -------------------------
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
