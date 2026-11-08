
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# ENERGY PULSE - STREAMLIT DASHBOARD
# ==========================================

st.set_page_config(
    page_title="Energy Pulse",
    page_icon="⚡",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "energy_pulse_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ==========================================
# LOAD MODEL
# ==========================================

if not MODEL_PATH.exists():
    st.error("energy_pulse_model.pkl was not found.")
    st.stop()

try:
    bundle = load_model()

    model = bundle["model"]
    model_name = bundle["model_name"]
    features = bundle["features"]
    defaults = bundle["defaults"]
    minimums = bundle["minimums"]
    maximums = bundle["maximums"]

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #111827;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label, span {
    color: #cbd5e1;
}

.hero {
    background: linear-gradient(120deg, #172554, #164e63);
    padding: 30px;
    border-radius: 16px;
    border: 1px solid #256080;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin: 0;
}

.hero p {
    color: #dbeafe;
    margin-top: 10px;
}

.metric-card {
    background: #172033;
    padding: 22px;
    border: 1px solid #293b56;
    border-radius: 14px;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    color: white;
    margin-top: 8px;
}

div.stButton > button {
    background: #0891b2;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: bold;
    min-height: 45px;
}

div.stButton > button:hover {
    background: #0e7490;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚡ ENERGY PULSE")
st.sidebar.caption("Smart Energy Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Energy Prediction",
        "Model Analytics",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success(f"Active Model: {model_name}")
st.sidebar.caption(f"Input Features: {len(features)}")


# ==========================================
# REUSABLE METRIC CARD
# ==========================================

def show_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# OVERVIEW
# ==========================================

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>⚡ Energy Pulse</h1>
        <p>Smart Energy Consumption Prediction System</p>
        <p>
        Machine learning based energy consumption
        estimation using environmental measurements.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        show_card("Active Algorithm", model_name)

    with c2:
        show_card("Input Features", str(len(features)))

    with c3:
        show_card("Prediction Target", "Appliances (Wh)")

    st.markdown("---")

    st.subheader("How Energy Pulse Works")

    st.markdown("""
    1. Collect environmental measurements.
    2. Process the input features.
    3. Pass the values to the trained ML model.
    4. Generate an appliance energy consumption estimate.
    5. Display the prediction on the dashboard.
    """)

    st.info(
        "This model estimates energy consumption for the "
        "same observation interval. It does not forecast "
        "future energy demand."
    )


# ==========================================
# ENERGY PREDICTION
# ==========================================

elif page == "Energy Prediction":

    st.title("⚡ Energy Consumption Prediction")

    st.write(
        "Adjust the 25 environmental input features "
        "to estimate appliance energy consumption."
    )

    st.caption(
        "Default values are taken from your original training data."
    )

    input_values = {}

    with st.form("energy_prediction_form"):

        st.subheader("Environmental Parameters")

        left, right = st.columns(2)

        for i, feature in enumerate(features):

            low = float(minimums[feature])
            high = float(maximums[feature])
            default = float(defaults[feature])

            if low == high:
                value = default

                with (left if i % 2 == 0 else right):
                    input_values[feature] = st.number_input(
                        feature,
                        value=value,
                        format="%.4f"
                    )

            else:
                with (left if i % 2 == 0 else right):
                    input_values[feature] = st.slider(
                        feature,
                        min_value=low,
                        max_value=high,
                        value=min(max(default, low), high),
                        help=f"Training range: {low:.2f} to {high:.2f}"
                    )

        st.markdown("---")

        submitted = st.form_submit_button(
            "⚡ Predict Energy Consumption",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [input_values],
            columns=features
        )

        try:
            prediction = float(model.predict(input_df)[0])

            st.markdown("---")
            st.subheader("Prediction Result")

            st.metric(
                "Estimated Appliance Energy Consumption",
                f"{prediction:.2f} Wh"
            )

            st.success("Prediction completed successfully!")

            with st.expander("View Input Parameters"):
                st.dataframe(
                    input_df.T.rename(
                        columns={0: "Input Value"}
                    ),
                    use_container_width=True
                )

        except Exception as error:
            st.error(f"Prediction error: {error}")


# ==========================================
# MODEL ANALYTICS
# ==========================================

elif page == "Model Analytics":

    st.title("📊 Model Analytics")

    st.write("Details of your existing trained model.")

    c1, c2 = st.columns(2)

    with c1:
        show_card("Selected Algorithm", model_name)

    with c2:
        show_card("Number of Features", str(len(features)))

    st.markdown("---")

    st.subheader("Model Input Features")

    feature_df = pd.DataFrame({
        "Feature": features,
        "Default Value": [
            defaults[f] for f in features
        ],
        "Minimum": [
            minimums[f] for f in features
        ],
        "Maximum": [
            maximums[f] for f in features
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The saved model package does not contain evaluation "
        "metrics. MAE, RMSE and R² are therefore not displayed "
        "as verified results here."
    )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("About Energy Pulse")

    st.markdown("""
    ## Smart Energy Consumption Prediction System

    **Project Domain:** Machine Learning and Data Science

    ### Problem Statement

    Understanding energy consumption is important for
    monitoring usage and supporting energy management.

    ### Objectives

    - Analyze energy consumption data.
    - Apply data preprocessing techniques.
    - Train machine learning regression models.
    - Predict appliance energy consumption.
    - Build an interactive dashboard.

    ### Algorithm Used

    **Linear Regression**

    A supervised machine learning algorithm used to
    predict continuous numerical values.

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Joblib
    - Streamlit

    ### Dataset

    UCI Appliances Energy Prediction Dataset.

    The dataset contains measurements from a residential
    building, including temperature, humidity and
    appliance energy consumption.

    ### Limitation

    This is a residential energy prediction prototype.
    It does not use actual campus meter readings or
    predict future energy demand.
    """)


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "⚡ Energy Pulse | Machine Learning Project"
)
