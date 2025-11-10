"""
Streamlit Web Application: Gastric Cancer Liver Metastasis Prediction
Includes Personalized SHAP Force Plot Interpretation
"""

import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
import streamlit.components.v1 as components  # 导入组件，用于渲染HTML

# --- Feature Mapping ---
# 用于将数字编码映射到英文标签
GENDER_MAP = {0: "Female", 1: "Male"}
N_STAGE_MAP = {0: "N0/N1", 1: "N2/N3"}
BINARY_MAP = {0: "No", 1: "Yes"}
TUMOR_SIZE_CODE_MAP = {0: "< 5cm", 1: ">= 5cm"}

# --- Page Configuration ---
st.set_page_config(
    page_title="Gastric Cancer Liver Metastasis Prediction",
    page_icon="🩺",
    layout="wide"
)

# --- Custom CSS Styles (Fixing Text Color) ---
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    /* Prediction box base styling */
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    /* HIGH RISK: Red background, black text for legibility */
    .high-risk {
        background-color: #ffe4e4; 
        border-left-color: #ff4444;
    }
    .high-risk h2, .high-risk h3, .high-risk p {
        color: black !important; /* Force black text for high contrast */
    }
    /* MEDIUM RISK: Orange background, black text for legibility */
    .medium-risk {
        background-color: #fff8e1;
        border-left-color: #ff9800;
    }
    .medium-risk h2, .medium-risk h3, .medium-risk p {
        color: black !important; /* Force black text for high contrast */
    }
    /* LOW RISK: Green background, black text for legibility */
    .low-risk {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    .low-risk h2, .low-risk h3, .low-risk p {
        color: black !important; /* Force black text for high contrast */
    }
</style>
""", unsafe_allow_html=True)


# --- Model and Explainer Loading (FIXED) ---
@st.cache_resource
def load_model():
    try:
        model = xgb.Booster()
        # Assume your model file is 'xgb_liver_metastasis.json'
        model.load_model('xgb_liver_metastasis.json')

        # ⚠️ SHAP VALUE FIX: Manually set base_score for SHAP TreeExplainer compatibility
        # For binary classification (logistic), XGBoost's default base_score is 0.5.
        # This addresses the ValueError when SHAP attempts to read model attributes.
        if 'base_score' not in model.attributes() or model.attributes().get('base_score') is None:
             # Setting the default base_score for binary:logistic models
            model.set_param({'base_score': 0.5})

        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, feature_names
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'xgb_liver_metastasis.json' exists.")
        st.stop()


model, feature_names = load_model()


@st.cache_resource
def get_explainer(_model):
    # This function should now succeed because the base_score is set in load_model
    return shap.TreeExplainer(_model)


explainer = get_explainer(model)
# --- End of FIXED section ---


# --- App Layout ---

# Title
st.markdown('<h1 class="main-header">🩺 Gastric Cancer Liver Metastasis Risk Prediction System</h1>',
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Feature Input
st.sidebar.header("📋 Patient Data Input")
st.sidebar.markdown("Please input the patient's clinical features:")

# Input Widgets (English labels)
gender = st.sidebar.selectbox(
    "Gender",
    options=[0, 1],
    format_func=lambda x: GENDER_MAP[x],
    index=1
)

tumor_size_raw = st.sidebar.number_input(
    "Tumor Size (cm)",
    min_value=0.1,
    max_value=30.0,
    value=3.0,
    step=0.1,
    help="Maximum diameter of the primary tumor"
)

# Tumor size encoding (>=5cm is 1, <5cm is 0)
tumor_size = 1 if tumor_size_raw >= 5 else 0

n_stage = st.sidebar.selectbox(
    "N Stage",
    options=[0, 1],
    format_func=lambda x: N_STAGE_MAP[x],
    help="Lymph Node Metastasis Stage"
)

radiation = st.sidebar.selectbox(
    "Radiation",
    options=[0, 1],
    format_func=lambda x: BINARY_MAP[x]
)

surgery = st.sidebar.selectbox(
    "Surgery",
    options=[0, 1],
    format_func=lambda x: BINARY_MAP[x],
    index=1
)

bone_met = st.sidebar.selectbox(
    "Bone Metastasis",
    options=[0, 1],
    format_func=lambda x: BINARY_MAP[x]
)

lung_met = st.sidebar.selectbox(
    "Lung Metastasis",
    options=[0, 1],
    format_func=lambda x: BINARY_MAP[x]
)

# Predict Button
predict_button = st.sidebar.button("🔍 Run Prediction", type="primary", use_container_width=True)

# Main Content
col1, col2 = st.columns([1, 1])

# --- Column 1: Prediction Results and Input Details ---
with col1:
    st.subheader("📊 Prediction Result")

    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Tumor.size': [tumor_size],  # Use the encoded feature
            'Radiation': [radiation],
            'Surgery': [surgery],
            'Bone.metastasis': [bone_met],
            'Lung.metastasis': [lung_met],
            'N.stage': [n_stage]
        })

        # Ensure column order matches training
        input_data = input_data[feature_names]

        # Convert to DMatrix
        dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)

        # Predict
        prediction = model.predict(dmatrix)[0]
        pred_percentage = prediction * 100

        # Risk Level Logic
        if prediction < 0.1:
            risk_level = "Low Risk"
            risk_class = "low-risk"
            risk_emoji = "🟢"
            risk_advice = "The patient's risk of liver metastasis is low. Routine follow-up is recommended."
        elif prediction < 0.3:
            risk_level = "Medium Risk"
            risk_class = "medium-risk"
            risk_emoji = "🟡"
            risk_advice = "The patient has a moderate risk of liver metastasis. Close monitoring and consideration of preventive measures are advised."
        else:
            risk_level = "High Risk"
            risk_class = "high-risk"
            risk_emoji = "🔴"
            risk_advice = "The patient's risk of liver metastasis is high. Further evaluation and an aggressive treatment plan are recommended."

        # Display Prediction Result (Text Color Fixed via CSS)
        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h2>{risk_emoji} Liver Metastasis Probability: {pred_percentage:.2f}%</h2>
            <h3>Risk Level: {risk_level}</h3>
            <p style="font-size: 1.1rem; margin-top: 1rem;">{risk_advice}</p>
        </div>
        """, unsafe_allow_html=True)

        # Display Input Feature Values
        st.subheader("📝 Input Feature Details")
        feature_display = pd.DataFrame({
            'Feature': ['Gender', 'Tumor Size', 'N Stage', 'Radiation', 'Surgery', 'Bone Metastasis',
                        'Lung Metastasis'],
            'Value': [
                GENDER_MAP[gender],
                f'{tumor_size_raw:.2f} cm (Encoded: {TUMOR_SIZE_CODE_MAP[tumor_size]})',
                N_STAGE_MAP[n_stage],
                BINARY_MAP[radiation],
                BINARY_MAP[surgery],
                BINARY_MAP[bone_met],
                BINARY_MAP[lung_met]
            ]
        })
        st.dataframe(feature_display, use_container_width=True, hide_index=True)

# --- Column 2: SHAP Analysis (Force Plot Fix) ---
with col2:
    st.subheader("🔥 Personalized Feature Impact Analysis (SHAP)")

    if predict_button:
        with st.spinner('Generating SHAP Force Plot...'):

            # --- SHAP Force Plot Generation (FIXED) ---
            # shap.force_plot() returns a visualization object. For Streamlit, we render it as HTML.

            # Compute SHAP values for the first sample
            # The shap_values for XGBoost DMatrix is a single array of shape (N_samples, N_features)
            shap_values = explainer.shap_values(dmatrix)
            expected_value = explainer.expected_value

            # Generate the HTML plot object
            # Note: matplotlib=False (default) is necessary for web rendering.
            shap_html = shap.force_plot(
                expected_value,
                shap_values[0],
                input_data.iloc[0],
                # link="logit" is often used for probability output, but 'identity' works fine too
                link="identity"
            )

            # Render the HTML plot in Streamlit
            # The height needs to be set manually for the HTML component
            components.html(shap.getjs() + shap_html.html(), height=300)

            st.markdown("""
            **📖 How to read the SHAP Force Plot:**
            - 🔴 **Red** features increase the risk of liver metastasis.
            - 🔵 **Blue** features decrease the risk of liver metastasis.
            - The arrows show how each feature pushes the prediction from the base value (expected value) to the final prediction.
            - Feature values are displayed on the plot (e.g., Lung.metastasis = Yes).
            """)

            # SHAP Detailed Table
            st.subheader("📊 Feature Contribution Details")

            # Use the feature names from the input data (ensures correct order)
            feature_names_list = input_data.columns.tolist()

            # Combine original feature names and SHAP values for sorting
            shap_data = zip(feature_names_list, shap_values[0])
            shap_df = pd.DataFrame(shap_data, columns=['Feature', 'SHAP Value'])

            # Add Impact Direction
            shap_df['Impact Direction'] = np.where(shap_df['SHAP Value'] > 0, 'Increase Risk ↑', 'Decrease Risk ↓')

            # Sort by absolute SHAP value for importance
            shap_df = shap_df.sort_values('SHAP Value', key=abs, ascending=False).reset_index(drop=True)

            st.dataframe(shap_df, use_container_width=True, hide_index=True)
    else:
        st.info("👈 Please enter patient data on the left and click the [Run Prediction] button.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>⚕️ This system is for clinical reference only and cannot replace professional medical diagnosis.</p>
    <p>Powered by XGBoost & SHAP | Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)