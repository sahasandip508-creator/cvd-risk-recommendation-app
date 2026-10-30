import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI-Assisted CVD Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ AI-Assisted Cardiovascular Disease Risk Prediction")
st.write(
    "Predict cardiovascular disease risk and receive "
    "personalized, rule-based health suggestions."
)

# ==========================================
# 2. LOAD TRAINED MODEL (joblib, not pickle)
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("cvd_xgb_model.pkl")
    feature_order = joblib.load("cvd_feature_order.pkl")
    return model, feature_order


try:
    model, feature_order = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# ==========================================
# 3. USER INPUT SECTION
# ==========================================

st.header("Enter Health Information")
st.info("Enter the available health information as accurately as possible.")

with st.form("cvd_prediction_form"):

    st.subheader("A. Demographic Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 18, 120, 40)

        gender = st.selectbox(
            "Gender", [1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female"
        )

        race = st.selectbox(
            "Race / Ethnicity",
            [1, 2, 3, 4, 6, 7],
            format_func=lambda x: {
                1: "Mexican American",
                2: "Other Hispanic",
                3: "Non-Hispanic White",
                4: "Non-Hispanic Black",
                6: "Non-Hispanic Asian",
                7: "Other / Multiracial"
            }[x]
        )

    with col2:
        education = st.selectbox(
            "Education level",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Less than 9th grade",
                2: "9th–11th grade",
                3: "High school graduate",
                4: "Some college / Associate degree",
                5: "College graduate or above"
            }[x]
        )

        income_ratio = st.number_input(
            "Family income-to-poverty ratio",
            min_value=0.0, max_value=20.0, value=2.0, step=0.1
        )

    st.subheader("B. Body Measurements")
    col1, col2, col3 = st.columns(3)

    with col1:
        weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0)
    with col2:
        height = st.number_input("Height (cm)", 100.0, 230.0, 170.0)
    with col3:
        bmi = st.number_input("BMI", 10.0, 80.0, 24.0)

    waist = st.number_input("Waist circumference (cm)", 30.0, 200.0, 85.0)

    st.subheader("C. Blood Pressure")
    col1, col2 = st.columns(2)

    with col1:
        systolic = st.number_input("Systolic blood pressure (mmHg)", 50, 300, 120)
    with col2:
        diastolic = st.number_input("Diastolic blood pressure (mmHg)", 20, 200, 80)

    st.subheader("D. Medical Information")

    diabetes = st.selectbox(
        "Doctor told you that you have diabetes?",
        [1, 2, 3],
        format_func=lambda x: {1: "Yes", 2: "No", 3: "Borderline"}[x]
    )

    hba1c = st.number_input("HbA1c (%)", 3.0, 20.0, 5.5, step=0.1)
    total_cholesterol = st.number_input("Total cholesterol (mg/dL)", 50.0, 600.0, 180.0)
    hdl = st.number_input("HDL cholesterol (mg/dL)", 5.0, 200.0, 50.0)

    st.subheader("E. Lifestyle Information")

    yes_no = lambda x: "Yes" if x == 1 else "No"

    smoking = st.selectbox(
        "Have you smoked at least 100 cigarettes in your life?",
        [1, 2], format_func=yes_no
    )
    vigorous_work = st.selectbox("Vigorous work activity?", [1, 2], format_func=yes_no)
    moderate_work = st.selectbox("Moderate work activity?", [1, 2], format_func=yes_no)
    walk_bicycle = st.selectbox("Walk or bicycle for transportation?", [1, 2], format_func=yes_no)
    vigorous_recreation = st.selectbox("Vigorous recreational activity?", [1, 2], format_func=yes_no)
    moderate_recreation = st.selectbox("Moderate recreational activity?", [1, 2], format_func=yes_no)

    sedentary_time = st.number_input("Daily sedentary time (minutes)", 0, 1440, 300)

    submit = st.form_submit_button("Predict CVD Risk", use_container_width=True)


# ==========================================
# 4. SUGGESTION ENGINE
# ==========================================

GENERAL_TIPS = {
    "Low": [
        "Keep up regular physical activity, such as walking, according to your ability.",
        "Continue a balanced diet with vegetables, fruits and less excess salt.",
        "Check your blood pressure, cholesterol and blood glucose at routine intervals.",
        "Stay tobacco-free and avoid exposure to smoke.",
        "Attend routine health check-ups and discuss any concerns with a healthcare professional.",
    ],
    "Moderate": [
        "Maintain regular physical activity, such as walking, according to your ability.",
        "Choose a balanced diet with vegetables, fruits and less excess salt.",
        "Monitor blood pressure, cholesterol and blood glucose.",
        "Avoid smoking and seek support to quit if needed.",
        "Discuss your risk factors with a healthcare professional.",
    ],
    "High": [
        "Arrange a medical consultation soon to review your cardiovascular risk factors.",
        "Ask your healthcare professional which tests (blood pressure, lipid profile, glucose) you need.",
        "Follow a heart-healthy diet with vegetables, fruits and less salt, as advised for you.",
        "Be physically active only to the level that is safe for you, after medical advice.",
        "Avoid smoking and seek support to quit if needed.",
        "Seek urgent medical care if you have chest pain, breathlessness or fainting.",
    ],
}


def personal_suggestions(data):
    """Extra suggestions based on the individual values entered."""
    s = []

    if data["LBXGH"] >= 6.5:
        s.append("Your HbA1c is in the diabetes-range threshold. Consult a healthcare professional for testing and interpretation.")
    elif data["LBXGH"] >= 5.7:
        s.append("Your HbA1c is in the prediabetes range. Discuss glucose monitoring and lifestyle changes with a healthcare professional.")

    if data["BPXOSY1"] >= 140 or data["BPXODI1"] >= 90:
        s.append("Your entered blood pressure is elevated. Repeat the measurement correctly and consult a healthcare professional.")
    elif data["BPXOSY1"] >= 130 or data["BPXODI1"] >= 80:
        s.append("Your blood pressure is above the normal range. Monitor it and discuss it with a healthcare professional.")

    if data["LBXTC"] >= 240:
        s.append("Your total cholesterol is high. Discuss a complete lipid profile and management with a healthcare professional.")

    if data["LBDHDD"] < 40:
        s.append("Your HDL cholesterol is low by commonly used thresholds. Discuss your lipid profile with a healthcare professional.")

    if data["BMXBMI"] >= 30:
        s.append("Your BMI is in the obesity range. Consider a safe, sustainable weight-management plan with a healthcare professional.")
    elif data["BMXBMI"] >= 25:
        s.append("Your BMI is in the overweight range. Balanced nutrition and suitable physical activity may support long-term health.")

    if data["SMQ020"] == 1:
        s.append("You reported a history of smoking. Avoid tobacco and consider professional support if you currently smoke.")

    activity = [data["PAQ605"], data["PAQ620"], data["PAQ635"], data["PAQ650"], data["PAQ665"]]
    if all(v == 2 for v in activity):
        s.append("You reported no listed physical activity. Consider gradually adding suitable activity after checking what is safe for you.")

    if data["PAD680"] >= 480:
        s.append("Your sedentary time is high. Try regular movement breaks during prolonged sitting.")

    return s


# ==========================================
# 5. PREDICTION AND RESULTS
# ==========================================

if submit:

    input_data = {
        "RIDAGEYR": age,
        "RIAGENDR": gender,
        "RIDRETH3": race,
        "DMDEDUC2": education,
        "INDFMPIR": income_ratio,
        "BMXWT": weight,
        "BMXHT": height,
        "BMXBMI": bmi,
        "BMXWAIST": waist,
        "BPXOSY1": systolic,
        "BPXODI1": diastolic,
        "DIQ010": diabetes,
        "SMQ020": smoking,
        "PAQ605": vigorous_work,
        "PAQ620": moderate_work,
        "PAQ635": walk_bicycle,
        "PAQ650": vigorous_recreation,
        "PAQ665": moderate_recreation,
        "PAD680": sedentary_time,
        "LBXGH": hba1c,
        "LBXTC": total_cholesterol,
        "LBDHDD": hdl,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_order]

    try:
        probability = float(model.predict_proba(input_df)[0][1])

        if probability < 0.20:
            risk = "Low"
        elif probability < 0.40:
            risk = "Moderate"
        else:
            risk = "High"

        st.divider()
        st.header("CVD Risk Assessment")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model-estimated probability", f"{probability * 100:.0f}%")
        with col2:
            st.metric("Risk category", risk)

        st.progress(min(max(probability, 0.0), 1.0))

        st.divider()
        st.header("Health suggestions")
        st.caption("Rule-based suggestions that depend on your risk category and the values you entered.")

        for tip in GENERAL_TIPS[risk]:
            st.info(tip)

        extra = personal_suggestions(input_data)
        if extra:
            st.subheader("Based on your entered values")
            for tip in extra:
                st.info(tip)

        st.warning(
            "This application is an educational research prototype. "
            "Its probability is not a clinically validated individual "
            "risk score or a medical diagnosis. Suggestions are general "
            "educational guidance, not medical treatment."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
