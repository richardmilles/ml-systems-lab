import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import xgboost as xgb  # CRITICAL: Must be imported for joblib to unpickle the model correctly

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, "Enterprise" look (No Emojis, Clean fonts)
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #34495e;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 4px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #34495e;
        border-color: #34495e;
    }
    .stAlert {
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING (ROBUST)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_prediction_model():
    """
    Loads the XGBoost model from the pickle file safely.
    """
    model_path = 'modele_xgboost_attrition.pkl'
    try:
        # We explicitly rely on the imports at the top of the file
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Critical Error: The file '{model_path}' was not found.")
        st.stop()
    except Exception as e:
        st.error(f"Critical Error loading the model: {e}")
        st.info("Ensure that 'xgboost' version in requirements.txt matches the training environment.")
        st.stop()

model = load_prediction_model()

# -----------------------------------------------------------------------------
# 3. PREPROCESSING FUNCTION (CORRECTED FOR ONE-HOT ENCODING)
# -----------------------------------------------------------------------------
def preprocess_user_input(user_data):
    """
    Transforms user input into the EXACT One-Hot Encoded format expected by the model.
    Based on the error log, the model expects 17 specific columns.
    """
    df = user_data.copy()

    # 1. Handle Numerical/Ordinal Columns (Direct Mapping)
    # ----------------------------------------------------
    # We explicitly convert 'OverTime' to 0 or 1 here
    df['OverTime'] = 1 if df['OverTime'].iloc[0] == 'Yes' else 0

    # 2. Manual One-Hot Encoding (To match model expectation)
    # ----------------------------------------------------
    
    # JOB ROLE: The model expects specific binary columns for roles
    role = df['JobRole'].iloc[0]
    df['JobRole_Human Resources'] = 1 if role == 'Human Resources' else 0
    df['JobRole_Laboratory Technician'] = 1 if role == 'Laboratory Technician' else 0
    df['JobRole_Manager'] = 1 if role == 'Manager' else 0
    df['JobRole_Manufacturing Director'] = 1 if role == 'Manufacturing Director' else 0
    df['JobRole_Research Director'] = 1 if role == 'Research Director' else 0
    df['JobRole_Research Scientist'] = 1 if role == 'Research Scientist' else 0
    df['JobRole_Sales Executive'] = 1 if role == 'Sales Executive' else 0
    df['JobRole_Sales Representative'] = 1 if role == 'Sales Representative' else 0

    # DEPARTMENT: 
    dept = df['Department'].iloc[0]
    df['Department_Research & Development'] = 1 if dept == 'Research & Development' else 0
    df['Department_Sales'] = 1 if dept == 'Sales' else 0

    # MARITAL STATUS:
    status = df['MaritalStatus'].iloc[0]
    df['MaritalStatus_Married'] = 1 if status == 'Married' else 0
    df['MaritalStatus_Single'] = 1 if status == 'Single' else 0

    # 3. Select ONLY the required columns in the EXACT ORDER from the error log
    # ----------------------------------------------------
    expected_columns = [
        'Age', 
        'JobSatisfaction', 
        'YearsInCurrentRole', 
        'YearsAtCompany', 
        'OverTime', 
        'JobRole_Human Resources', 
        'JobRole_Laboratory Technician', 
        'JobRole_Manager', 
        'JobRole_Manufacturing Director', 
        'JobRole_Research Director', 
        'JobRole_Research Scientist', 
        'JobRole_Sales Executive', 
        'JobRole_Sales Representative', 
        'Department_Research & Development', 
        'Department_Sales', 
        'MaritalStatus_Married', 
        'MaritalStatus_Single'
    ]

    # Return only these columns
    return df[expected_columns]

# -----------------------------------------------------------------------------
# 4. SIDEBAR - USER INPUTS
# -----------------------------------------------------------------------------
st.sidebar.title("Employee Profile")
st.sidebar.markdown("Configure the parameters below:")

def get_user_input():
    # Group 1: Demographics
    st.sidebar.markdown("### 1. Personal Details")
    age = st.sidebar.slider("Age", 18, 65, 30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    marital = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4, 5], index=2, help="1: Below College, 5: Doctor")
    edu_field = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])

    # Group 2: Job Specifics
    st.sidebar.markdown("### 2. Job Information")
    dept = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    role = st.sidebar.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician", 
        "Manufacturing Director", "Healthcare Representative", "Manager", 
        "Sales Representative", "Research Director", "Human Resources"
    ])
    travel = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    distance = st.sidebar.number_input("Distance From Home (km)", 1, 50, 5)
    
    # Group 3: Financial & Conditions
    st.sidebar.markdown("### 3. Compensation & Conditions")
    income = st.sidebar.number_input("Monthly Income ($)", 1000, 50000, 5000, step=250)
    overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])
    stock = st.sidebar.selectbox("Stock Option Level", [0, 1, 2, 3], index=1)
    
    # Group 4: Satisfaction
    st.sidebar.markdown("### 4. Satisfaction & Tenure")
    satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
    env_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
    years_in_role = st.sidebar.slider("Years in Current Role", 0, 20, 2)

    # Create Dictionary
    user_data = {
        'Age': age,
        'Gender': gender,
        'MaritalStatus': marital,
        'Education': education,
        'EducationField': edu_field,
        'Department': dept,
        'JobRole': role,
        'BusinessTravel': travel,
        'DistanceFromHome': distance,
        'MonthlyIncome': income,
        'OverTime': overtime,
        'StockOptionLevel': stock,
        'JobSatisfaction': satisfaction,
        'EnvironmentSatisfaction': env_satisfaction,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_role
    }
    
    return pd.DataFrame(user_data, index=[0])

# Get data from sidebar
input_df = get_user_input()

# -----------------------------------------------------------------------------
# 5. MAIN PAGE LAYOUT
# -----------------------------------------------------------------------------
st.title("HR Analytics: Attrition Prediction")
st.markdown("""
This application uses a machine learning model (XGBoost) to assess the probability 
of an employee leaving the organization (Attrition).
""")

st.markdown("---")

# Layout: 2 Columns (Left: Result, Right: Summary)
col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("Prediction Analysis")
    st.write("Click the button below to process the input data.")
    
    if st.button("RUN PREDICTION"):
        # Show a spinner while calculating
        with st.spinner("Processing data..."):
            try:
                # 1. Preprocess
                processed_data = preprocess_user_input(input_df)
                
                # 2. Predict Probability
                # We get [Prob_Class_0, Prob_Class_1] -> We want Class 1 (Yes)
                prediction_proba = model.predict_proba(processed_data)[0][1]
                
                # 3. Predict Class
                prediction_class = int(prediction_proba > 0.5)

                # 4. Display Results
                st.markdown("### Result:")
                
                risk_percentage = prediction_proba * 100
                
                if risk_percentage > 50:
                    st.error(f"ATTRITION RISK: HIGH ({risk_percentage:.1f}%)")
                    st.markdown("**Recommendation:** Initiate retention interview immediately. Review salary and work-life balance.")
                else:
                    st.success(f"ATTRITION RISK: LOW ({risk_percentage:.1f}%)")
                    st.markdown("**Recommendation:** Maintain current engagement strategies.")

                # 5. Professional Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_percentage,
                    number = {'suffix': "%", 'font': {'size': 40}},
                    title = {'text': "Probability of Departure", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
                        'bar': {'color': "#2c3e50"}, # Professional Dark Blue
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': "#d4edda"},  # Soft Green
                            {'range': [50, 75], 'color': "#fff3cd"}, # Soft Yellow
                            {'range': [75, 100], 'color': "#f8d7da"} # Soft Red
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_percentage
                        }
                    }
                ))
                
                fig.update_layout(height=300, margin={'t':0, 'b':0, 'l':20, 'r':20})
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.write("Debug info: Check if all columns match the model requirements.")

with col_right:
    st.subheader("Profile Summary")
    st.write("Review the key indicators selected:")
    
    # Create a clean summary table
    summary_data = {
        "Metric": ["Age", "Role", "Department", "Income", "OverTime", "Distance"],
        "Value": [
            input_df['Age'][0],
            input_df['JobRole'][0],
            input_df['Department'][0],
            f"${input_df['MonthlyIncome'][0]}",
            input_df['OverTime'][0],
            f"{input_df['DistanceFromHome'][0]} km"
        ]
    }
    st.table(pd.DataFrame(summary_data).set_index("Metric"))
    
    st.info("""
    **Note on Methodology:**
    The model focuses on 17 specific factors including OverTime, Role, Age, and Satisfaction.
    Other inputs are collected for profile completeness.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    HR Analytics Dashboard | v2.0 Stable | Powered by Nerva AI Solutions
</div>
""", unsafe_allow_html=True)