"""
Streamlit Web Application: Gastric Cancer Liver Metastasis Prediction
Optimized for Streamlit Cloud Deployment
"""

import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
import streamlit.components.v1 as components
import os

# --- Feature Mapping ---
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

# --- Custom CSS Styles ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffe4e4; 
        border-left-color: #ff4444;
    }
    .high-risk h2, .high-risk h3, .high-risk p {
        color: black !important;
    }
    .medium-risk {
        background-color: #fff8e1;
        border-left-color: #ff9800;
    }
    .medium-risk h2, .medium-risk h3, .medium-risk p {
        color: black !important;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    .low-risk h2, .low-risk h3, .low-risk p {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Model and Explainer Loading ---
@st.cache_resource
def load_model():
    """Load XGBoost model and feature names with error handling"""
    try:
        # 检查文件是否存在
        model_path = 'xgb_liver_metastasis.json'
        feature_path = 'feature_names.json'

        if not os.path.exists(model_path):
            st.error(f"❌ Model file '{model_path}' not found!")
            st.info("📁 Current directory files: " + ", ".join(os.listdir('.')))
            st.stop()

        if not os.path.exists(feature_path):
            st.error(f"❌ Feature names file '{feature_path}' not found!")
            st.stop()

        # 加载模型
        model = xgb.Booster()
        model.load_model(model_path)

        # 修正 base_score
        base_score_str = model.attributes().get('base_score')
        final_base_score_float = 0.5

        if base_score_str:
            cleaned_base_score_str = base_score_str.strip('[]').strip()
            try:
                final_base_score_float = float(cleaned_base_score_str)
            except ValueError:
                st.warning("⚠️ Model base_score format error, using default 0.5")

        model.set_param({'base_score': final_base_score_float})

        # 加载特征名称
        with open(feature_path, 'r') as f:
            feature_names = json.load(f)

        return model, feature_names

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("📁 Please ensure all required files are uploaded to the GitHub repository.")
        st.stop()


@st.cache_resource
def get_explainer(_model):
    """Create SHAP explainer with caching"""
    return shap.TreeExplainer(_model)


# 加载模型和解释器
try:
    model, feature_names = load_model()
    explainer = get_explainer(model)
except Exception as e:
    st.error(f"❌ Initialization failed: {str(e)}")
    st.stop()


# --- App Layout ---

# Title
st.markdown('<h1 class="main-header">🩺 Gastric Cancer Liver Metastasis Risk Prediction System</h1>',
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Feature Input
st.sidebar.header("📋 Patient Data Input")
st.sidebar.markdown("Please input the patient's clinical features:")

# Input Widgets
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

# --- Column 1: Prediction Results ---
with col1:
    st.subheader("📊 Prediction Result")

    if predict_button:
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Tumor.size': [tumor_size],
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

            # Display Prediction Result
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

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# --- Column 2: SHAP Analysis ---
with col2:
    st.subheader("🔥 Personalized Feature Impact Analysis (SHAP)")

    if predict_button:
        try:
            with st.spinner('Generating SHAP Force Plot...'):

                # Compute SHAP values
                shap_values = explainer.shap_values(dmatrix)
                expected_value = explainer.expected_value

                # Generate SHAP force plot
                shap_html = shap.force_plot(
                    expected_value,
                    shap_values[0],
                    input_data.iloc[0],
                    link="identity"
                )

                # Render the HTML plot
                components.html(shap.getjs() + shap_html.html(), height=300)

                st.markdown("""
                **📖 How to read the SHAP Force Plot:**
                - 🔴 **Red** features increase the risk of liver metastasis.
                - 🔵 **Blue** features decrease the risk of liver metastasis.
                - The arrows show how each feature pushes the prediction from the base value to the final prediction.
                - Feature values are displayed on the plot (e.g., Lung.metastasis = Yes).
                """)

                # SHAP Detailed Table
                st.subheader("📊 Feature Contribution Details")

                feature_names_list = input_data.columns.tolist()
                shap_data = zip(feature_names_list, shap_values[0])
                shap_df = pd.DataFrame(shap_data, columns=['Feature', 'SHAP Value'])
                shap_df['Impact Direction'] = np.where(shap_df['SHAP Value'] > 0, 'Increase Risk ↑', 'Decrease Risk ↓')
                shap_df = shap_df.sort_values('SHAP Value', key=abs, ascending=False).reset_index(drop=True)

                st.dataframe(shap_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ SHAP analysis error: {str(e)}")
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