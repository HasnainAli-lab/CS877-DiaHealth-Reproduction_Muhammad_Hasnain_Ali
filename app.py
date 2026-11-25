# CS877 project - Streamlit app (paper Figures 7-9)
# Run on your laptop:  pip install -r requirements.txt
#                      streamlit run app.py
# model.pkl must be in the same folder (Step 3 of CS877_Colab_Notebook.ipynb creates it).

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺")
bundle = pickle.load(open("model.pkl", "rb"))
model, scaler, features, importance = bundle["model"], bundle["scaler"], bundle["features"], bundle["importance"]

st.title("Diabetes Risk Prediction")
st.write("Random Forest model trained on the DiaHealth dataset (5,437 patients, Bangladesh).")
st.caption("Research prototype for a university course (CS877, NUST-SEECS). Not a medical tool. It does not replace a doctor.")

yn = {"No": 0, "Yes": 1}
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age (years)", 1, 120, 45)
    gender = st.selectbox("Gender", ["Female", "Male"])
    pulse = st.number_input("Pulse rate (beats/min)", 30, 200, 76)
    sys_bp = st.number_input("Systolic BP (mmHg)", 60, 260, 130)
    dia_bp = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
    glucose = st.number_input("Glucose (mmol/L)", 2.0, 40.0, 6.9, step=0.1)
    height = st.number_input("Height (m)", 1.0, 2.2, 1.55, step=0.01)
with c2:
    weight = st.number_input("Weight (kg)", 20.0, 200.0, 53.0, step=0.5)
    fam_dia = st.selectbox("Family history of diabetes", list(yn))
    hyper = st.selectbox("Hypertensive", list(yn))
    fam_hyp = st.selectbox("Family history of hypertension", list(yn))
    cvd = st.selectbox("Cardiovascular disease", list(yn))
    stroke = st.selectbox("Stroke", list(yn))

if st.button("Predict"):
    bmi = weight / (height ** 2)
    row = {
        "age": age,
        "gender": int(gender == "Male"),
        "pulse_rate": pulse,
        "systolic_bp": sys_bp,
        "diastolic_bp": dia_bp,
        "glucose": glucose,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "family_diabetes": yn[fam_dia],
        "hypertensive": yn[hyper],
        "family_hypertension": yn[fam_hyp],
        "cardiovascular_disease": yn[cvd],
        "stroke": yn[stroke],
    }
    X = scaler.transform(pd.DataFrame([row])[features])
    prob = model.predict_proba(X)[0][1]
    if model.predict(X)[0] == 1:
        st.error(f"Diabetes detected (model probability {prob:.0%})")
    else:
        st.success(f"No diabetes detected (model probability of diabetes {prob:.0%})")
    st.write(f"BMI used by the model: {bmi:.1f}")
    st.subheader("Top factors the model uses overall")
    top = pd.Series(importance).sort_values(ascending=False).head(5)
    st.table(top.rename("importance").round(3))
