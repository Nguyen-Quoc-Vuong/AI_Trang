import streamlit as st
import joblib
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import json


# Load model và scaler
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')

# Kết nối Google Sheet
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# creds = ServiceAccountCredentials.from_json_keyfile_name("student-predictions.json", scope)
json_account_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(json_account_info, scope)

client = gspread.authorize(creds)
sheet = client.open_by_key("12Fuj0GmbROQupAreLyqPAqMjPEMxoh7gNLEy3BIDOR4").sheet1

# Giao diện Streamlit
st.title("🎓 Dự đoán kết quả học tập của học sinh")

gender = st.selectbox("Giới tính", ["male", "female"])
race = st.selectbox("Lớp/ Nhóm", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Trình độ học vấn của phụ huynh", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Bữa trưa", ["standard", "free/reduced", "none"])
test_prep = st.selectbox("Có ôn bài sau khi thi giữa kỳ không", ["completed", "none"])
midterm_score = st.slider("Midterm Score", 0, 100, 50)

def preprocess_input():
    input_dict = {
        "gender": 0 if gender == "female" else 1,
        "race/ethnicity": {"group A": 1, "group B": 2, "group C": 3, "group D": 4, "group E": 5}[race],
        "parental level of education": {
            "bachelor's degree": 1,
            'some college': 2,
            "master's degree": 3,
            "associate's degree": 4,
            'high school': 5,
            'some high school': 6
        }[parent_edu],
        "lunch": 1 if lunch == "standard" else 0,
        "test preparation course": 1 if test_prep == "completed" else 0,
        "midterm score": midterm_score
    }
    return np.array(list(input_dict.values())).reshape(1, -1)

def save_to_sheet(result):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([
        now, gender, race, parent_edu, lunch, test_prep, midterm_score, result
    ])

# Xử lý dự đoán và lưu dữ liệu
if st.button("Dự đoán"):
    X_input = preprocess_input()
    X_input_scaled = scaler.transform(X_input)
    prediction = model.predict(X_input_scaled)

    if prediction[0] == 1:
        result = "Pass"
        st.success("✅ Học sinh **có khả năng vượt qua** (Pass)")
    else:
        result = "Fail"
        st.error("❌ Học sinh **có thể trượt** (Fail)")

    save_to_sheet(result)
