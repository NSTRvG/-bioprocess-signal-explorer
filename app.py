import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

st.set_page_config(
    page_title="BioProcess Excursion Monitor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "Exploratory detection of process excursions using causal dynamic bands. "
    "This dashboard is not a validated control or safety system."
)

# ============================================
# AUTO-GENERATE DATA ON FIRST RUN (Cloud Deploy)
# ============================================
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "synthetic" / "synthetic_batch.csv"

if not SYNTHETIC_PATH.exists():
    st.info("🔄 Generating synthetic data for first run...")
    
    # Guardar los argumentos originales de Streamlit
    original_argv = sys.argv.copy()
    
    try:
        # Limpiar sys.argv para que argparse no vea argumentos de Streamlit
        sys.argv = [str(PROJECT_ROOT / "scripts" / "generate_synthetic.py")]
        
        # Importar y ejecutar la función de generación
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.generate_synthetic import main as generate_synthetic
        
        # Llamar a main() sin argumentos (usará los valores por defecto)
        generate_synthetic()
        
        # Restaurar los argumentos originales
        sys.argv = original_argv
        
        st.success("✅ Synthetic data generated successfully!")
        st.rerun()  # Recargar la app con los nuevos datos
        
    except Exception as e:
        # Restaurar argv incluso si hay error
        sys.argv = original_argv
        st.error(f"❌ Error generating data: {e}")
        st.stop()


from src.detector import detect_excursions
from src.models import DetectionConfig, DetectionMethod, Severity
from src.preprocessing import load_and_preprocess, validate_data
from src.viz import create_band_chart, create_events_table

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

st.set_page_config(
    page_title="BioProcess Excursion Monitor",
    page_icon="🔬",
    layout="wide",
)


def main() -> None:
    st.title("🔬 BioProcess Excursion Monitor")
    st.markdown(
        "Exploratory detection of process excursions using causal dynamic bands. "
        "This dashboard is not a validated control or safety system."
    )

    df = None
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader(
            "Upload a CSV",
            type=["csv"],
            help="CSV with a timestamp column and numeric measurement columns.",
        )
        use_sample = st.checkbox("Use bundled sample", value=True)
        sample_choice = st.selectbox(
            "Sample data",
            ["Synthetic batch", "IndPenSim batch 95 (benchmark)"],
            disabled=not use_sample or uploaded_file is not None,
        )
        use_benchmark = (
            uploaded_file is None
            and use_sample
            and sample_choice == "IndPenSim batch 95 (benchmark)"
        )
        window_key = "window_indpensim" if use_benchmark else "window_default"
        window = st.select_slider(
            "Rolling window",
            options=["30min", "60min", "120min", "180min", "240min", "360min"],
            value="180min" if use_benchmark else "60min",
            key=window_key,
        )
        k = st.slider("Threshold k", 2.0, 4.0, 3.0, 0.1)
        method_label = st.radio(
            "Band method",
            ["Robust (median + MAD)", "Classic (mean + std)"],
        )
        min_periods = st.slider("Minimum baseline samples", 5, 20, 10)
        min_event_points = st.slider("Minimum samples per event", 1, 5, 2)
        max_gap_minutes = st.slider("Maximum event gap (minutes)", 0, 60, 10)

        if uploaded_file is not None:
            if uploaded_file.size > MAX_UPLOAD_BYTES:
                st.error("CSV exceeds the 50 MB upload limit.")
            else:
                try:
                    df = load_and_preprocess(uploaded_file)
                except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                    st.error(f"Could not load CSV: {exc}")
        elif use_sample:
            if sample_choice == "Synthetic batch":
                sample_path = (
                    PROJECT_ROOT / "data" / "synthetic" / "synthetic_batch.csv"
                )
            else:
                sample_path = (
                    PROJECT_ROOT / "data" / "sample" / "indpensim_batch_095.csv"
                )
            if sample_path.is_file():
                try:
                    df = load_and_preprocess(sample_path)
                except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                    st.error(f"Could not load selected sample: {exc}")
            else:
                st.info(
                    "Selected sample not found. Generate the synthetic data with "
                    "`python scripts/generate_synthetic.py` or download the "
                    "benchmark sample with "
                    "`python scripts/download_indpensim_sample.py`."
                )
            if (
                df is not None
                and sample_choice == "IndPenSim batch 95 (benchmark)"
            ):
                st.info(
                    "IndPenSim is a simulator-generated published benchmark, not "
                    "industrial measurement data. Its sample cadence is 12 minutes; "
                    "a rolling window of at least 180 minutes is recommended for "
                    "10-point baselines."
                )
            elif df is not None and sample_choice == "Synthetic batch":
                st.info(
                    "Synthetic units: pH is dimensionless, temperature is in °C, "
                    "DO uses an arbitrary scale (not mg/L or calibrated saturation), "
                    "and biomass uses arbitrary model units."
                )

        variables: list[str] = []
        if df is not None:
            variables = st.multiselect(
                "Variables to analyze",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:2],
            )

    if df is None:
        st.info("Upload a CSV or generate/select a bundled sample.")
        return
    if not variables:
        st.info("Select at least one variable in the sidebar.")
        return

    config = DetectionConfig(
        window=window,
        k=k,
        method=(
            DetectionMethod.ROBUST
            if method_label.startswith("Robust")
            else DetectionMethod.CLASSIC
        ),
        min_periods=min_periods,
        min_event_points=min_event_points,
        max_gap_minutes=max_gap_minutes,
    )
    results = {
        variable: detect_excursions(df, variable, config) for variable in variables
    }
    all_events = [event for result in results.values() for event in result.events]

    total_col, critical_col, variables_col, duration_col = st.columns(4)
    total_col.metric("Total events", len(all_events))
    critical_col.metric(
        "Critical",
        sum(event.severity == Severity.CRITICAL for event in all_events),
    )
    variables_col.metric("Affected variables", len({e.variable for e in all_events}))
    duration_col.metric(
        "Total duration (min)", round(sum(e.duration_minutes for e in all_events), 1)
    )

    st.subheader("Time series and dynamic bands")
    for variable, result in results.items():
        st.plotly_chart(
            create_band_chart(result, df),
            use_container_width=True,
            key=f"chart-{variable}",
        )

    st.subheader("Detected events")
    if all_events:
        events_df = create_events_table(all_events)
        filter_col, severity_col = st.columns(2)
        selected_variable = filter_col.selectbox(
            "Filter by variable",
            ["All"] + sorted({event.variable for event in all_events}),
        )
        selected_severity = severity_col.selectbox(
            "Filter by severity",
            ["All", "low", "moderate", "critical"],
        )
        filtered = events_df
        if selected_variable != "All":
            filtered = filtered[filtered["Variable"] == selected_variable]
        if selected_severity != "All":
            filtered = filtered[filtered["Severity"] == selected_severity]
        st.dataframe(filtered, use_container_width=True)
        st.download_button(
            "Download events as CSV",
            data=filtered.to_csv(index=False),
            file_name="excursion_events.csv",
            mime="text/csv",
        )
    else:
        st.info("No events meet the current detector settings.")

    report = validate_data(df)
    st.subheader("Input data validation")
    row_col, column_col, duration_col = st.columns(3)
    row_col.metric("Rows", report["n_rows"])
    column_col.metric("Numeric variables", report["n_columns"])
    duration_col.metric(
        "Time span (hours)", round(report["time_range"]["duration_hours"], 1)
    )
    if any(report["missing_values"].values()):
        st.warning(f"Unfilled missing values remain: {report['missing_values']}")


if __name__ == "__main__":
    main()
