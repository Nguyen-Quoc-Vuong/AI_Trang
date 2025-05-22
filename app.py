import streamlit as st
import joblib
import numpy as np

# Load mô hình đã huấn luyện
model = joblib.load('student_model.pkl')
# Load scaler
scaler = joblib.load('scaler.pkl')


st.title("🎓 Dự đoán kết quả học tập của học sinh")

st.write("Hãy nhập các thông tin bên dưới để dự đoán điểm số trung bình của học sinh có thể trên 50 điểm không !")

# Nhập liệu từ người dùng
gender = st.selectbox("Giới tính", ["male", "female"])
race = st.selectbox("Lớp/ Nhóm", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Trình độ học vấn của phụ huynh", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Bữa trưa", ["standard", "free/reduced", "none"])
test_prep = st.selectbox("Có ôn bài sau khi thi giữa kỳ không", ["completed", "none"])
midterm_score = st.slider("Midterm Score", 0, 100, 50)

# Mã hóa đầu vào tương tự như lúc train
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

# Dự đoán
if st.button("Dự đoán"):
    X_input = preprocess_input()
    X_input_scaled = scaler.transform(X_input)
    prediction = model.predict(X_input_scaled)
    if prediction[0] == 1:
        st.success("✅ Học sinh **có khả năng vượt qua** (Pass)")
    else:
        st.error("❌ Học sinh **có thể trượt** (Fail)")


