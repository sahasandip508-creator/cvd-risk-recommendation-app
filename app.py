import streamlit as st
import pandas as pd
import pickle

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
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    with open("cvd_xgb_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("cvd_feature_order.pkl", "rb") as file:
        feature_order = pickle.load(file)

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
        age = st.number_input(
            "Age (years)", 18, 120, 40
        )

        gender = st.selectbox(
            "Gender",
            [1, 2],
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
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1
        )

    st.subheader("B. Body Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        weight = st.number_input(
            "Weight (kg)", 20.0, 300.0, 70.0
        )

    with col2:
        height = st.number_input(
            "Height (cm)", 100.0, 230.0, 170.0
        )

    with col3:
        bmi = st.number_input(
            "BMI", 10.0, 80.0, 24.0
        )

    waist = st.number_input(
        "Waist circumference (cm)",
        30.0, 200.0, 85.0
    )

    st.subheader("C. Blood Pressure")

    col1, col2 = st.columns(2)

    with col1:
        systolic = st.number_input(
            "Systolic blood pressure (mmHg)",
            50, 300, 120
        )

    with col2:
        diastolic = st.number_input(
            "Diastolic blood pressure (mmHg)",
            20, 200, 80
        )

    st.subheader("D. Medical Information")

    diabetes = st.selectbox(
        "Doctor told you that you have diabetes?",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Yes",
            2: "No",
            3: "Borderline"
        }[x]
    )

    hba1c = st.number_input(
        "HbA1c (%)", 3.0, 20.0, 5.5, step=0.1
    )

    total_cholesterol = st.number_input(
        "Total cholesterol (mg/dL)",
        50.0, 600.0, 180.0
    )

    hdl = st.number_input(
        "HDL cholesterol (mg/dL)",
        5.0, 200.0, 50.0
    )

    st.subheader("E. Lifestyle Information")

    smoking = st.selectbox(
        "Have you smoked at least 100 cigarettes in your life?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    vigorous_work = st.selectbox(
        "Vigorous work activity?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    moderate_work = st.selectbox(
        "Moderate work activity?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    walk_bicycle = st.selectbox(
        "Walk or bicycle for transportation?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    vigorous_recreation = st.selectbox(
        "Vigorous recreational activity?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    moderate_recreation = st.selectbox(
        "Moderate recreational activity?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    sedentary_time = st.number_input(
        "Daily sedentary time (minutes)",
        0, 1440, 300
    )

    submit = st.form_submit_button(
        "Predict CVD Risk",
        use_container_width=True
    )


# ==========================================
# 4. PYTHON RECOMMENDATION ENGINE
# ==========================================

def generate_suggestions(probability, data):

    suggestions = []

    # Probability-based suggestions

    if probability < 0.20:
        suggestions.append(
            "Your model-estimated CVD probability is in the "
            "Low category. Continue maintaining healthy habits "
            "and attend routine health check-ups."
        )

    elif probability < 0.40:
        suggestions.append(
            "Your model-estimated CVD probability is in the "
            "Moderate category. Review your health and lifestyle "
            "factors and discuss them with a healthcare professional."
        )

    else:
        suggestions.append(
            "Your model-estimated CVD probability is in the "
            "High category. Arrange a medical consultation to "
            "review your cardiovascular risk factors."
        )

    # HbA1c suggestions

    if data["LBXGH"] >= 6.5:
        suggestions.append(
            "Your HbA1c value is in the diabetes-range threshold. "
            "Consult a healthcare professional for appropriate "
            "testing and interpretation."
        )

    elif data["LBXGH"] >= 5.7:
        suggestions.append(
            "Your HbA1c value is in the prediabetes range. "
            "Discuss blood glucose monitoring and lifestyle "
            "changes with a healthcare professional."
        )

    # Blood pressure suggestions

    if data["BPXOSY1"] >= 140 or data["BPXODI1"] >= 90:
        suggestions.append(
            "Your entered blood pressure is elevated. Repeat "
            "measurements correctly and consult a healthcare "
            "professional."
        )

    elif data["BPXOSY1"] >= 130 or data["BPXODI1"] >= 80:
        suggestions.append(
            "Your blood pressure reading is above the normal "
            "range. Monitor it and discuss the result with a "
            "healthcare professional."
        )

    # Cholesterol suggestions

    if data["LBXTC"] >= 240:
        suggestions.append(
            "Your total cholesterol is high. Discuss a complete "
            "lipid profile and appropriate management with "
            "your healthcare professional."
        )

    if data["LBDHDD"] < 40:
        suggestions.append(
            "Your HDL cholesterol is low by commonly used "
            "reference thresholds. Discuss your overall lipid "
            "profile with a healthcare professional."
        )

    # BMI suggestions

    if data["BMXBMI"] >= 30:
        suggestions.append(
            "Your BMI is in the obesity range. Consider discussing "
            "a safe, sustainable weight-management plan with "
            "a healthcare professional."
        )

    elif data["BMXBMI"] >= 25:
        suggestions.append(
            "Your BMI is in the overweight range. Balanced "
            "nutrition and appropriate physical activity may "
            "support long-term health."
        )

    # Smoking suggestions

    if data["SMQ020"] == 1:
        suggestions.append(
            "You reported a history of smoking. Avoid tobacco "
            "and consider professional support if you currently "
            "smoke or want to quit."
        )

    # Physical activity suggestions

    activity_values = [
        data["PAQ605"],
        data["PAQ620"],
        data["PAQ635"],
        data["PAQ650"],
        data["PAQ665"]
    ]

    if all(value == 2 for value in activity_values):
        suggestions.append(
            "You reported no participation in the listed physical "
            "activities. Consider gradually introducing suitable "
            "physical activity after checking what is safe for you."
        )

    # Sedentary behaviour

    if data["PAD680"] >= 480:
        suggestions.append(
            "Your reported sedentary time is high. Try taking "
            "regular movement breaks during prolonged sitting."
        )

    return suggestions


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
        "LBDHDD": hdl
    }

    input_df = pd.DataFrame([input_data])

    # Maintain exact saved feature order
    input_df = input_df[feature_order]

    try:

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        if probability < 0.20:
            risk = "Low"
            color = "success"

        elif probability < 0.40:
            risk = "Moderate"
            color = "warning"

        else:
            risk = "High"
            color = "error"

        st.divider()

        st.header("CVD Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted CVD Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Risk Category",
                risk
            )

        st.progress(probability)

        st.divider()

        st.header("Personalized Health Suggestions")

        suggestions = generate_suggestions(
            probability,
            input_data
        )

        for i, suggestion in enumerate(suggestions, 1):
            st.info(f"**{i}.** {suggestion}")

        st.success(
            "Suggestions generated using the Python-based "
            "personalized recommendation engine."
        )

        st.warning(
            "This application is an educational research prototype. "
            "Its probability is not a clinically validated individual "
            "risk score or a medical diagnosis. Recommendations are "
            "general educational guidance, not medical treatment."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
