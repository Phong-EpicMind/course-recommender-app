# app_minimalist.py

import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Hàm load model và encoders
@st.cache_resource
def load_model_and_encoders(model_path, dataset_path):
    model = joblib.load('random_forest_model_new.pkl')
    df = pd.read_csv('dataset_200_hoc_vien_goi_y_khoa_hoc.csv')

    le_goal = LabelEncoder()
    le_industry = LabelEncoder()
    le_personality = LabelEncoder()
    le_ai = LabelEncoder()
    le_course = LabelEncoder()

    df["Muc Tieu"] = le_goal.fit_transform(df["Muc Tieu"])
    df["Nganh Nghe"] = le_industry.fit_transform(df["Nganh Nghe"])
    df["So Thich"] = le_personality.fit_transform(df["So Thich"])
    df["Trinh Do AI"] = le_ai.fit_transform(df["Trinh Do AI"])
    df["Khoa Hoc Goi Y"] = le_course.fit_transform(df["Khoa Hoc Goi Y"])

    encoders = {
        "goal": le_goal,
        "industry": le_industry,
        "personality": le_personality,
        "ai": le_ai,
        "course": le_course
    }

    return model, encoders

# Hàm dự đoán khóa học
def predict_course(model, encoders, muc_tieu, nganh_nghe, so_thich, trinh_do_ai):
    input_encoded = [
        encoders["goal"].transform([muc_tieu])[0],
        encoders["industry"].transform([nganh_nghe])[0],
        encoders["personality"].transform([so_thich])[0],
        encoders["ai"].transform([trinh_do_ai])[0]
    ]
    input_df = pd.DataFrame([input_encoded], columns=["Muc Tieu", "Nganh Nghe", "So Thich", "Trinh Do AI"])
    predicted_label = model.predict(input_df)[0]
    predicted_course = encoders["course"].inverse_transform([predicted_label])[0]
    return predicted_course

# ========== Giao diện Apple Style ==========

# Cấu hình trang
st.set_page_config(
    page_title="Course Recommender",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #333333;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }
    </style>
""", unsafe_allow_html=True)

# Nội dung chính
st.markdown("<h1 style='text-align: center; font-weight: 500;'>Find Your Perfect Course</h1>", unsafe_allow_html=True)
st.write("")
st.write("<p style='text-align: center; font-size: 18px; color: #666;'>Answer a few quick questions and get a personalized recommendation.</p>", unsafe_allow_html=True)
st.write("")

# Load model và encoders
model_path = "random_forest_model_new.pkl"
dataset_path = "dataset_200_hoc_vien_can_bang.csv"
model, encoders = load_model_and_encoders('random_forest_model_new.pkl', 'dataset_200_hoc_vien_goi_y_khoa_hoc.csv')

# Layout form
with st.form(key='form_hocvien'):
    muc_tieu = st.selectbox("Your Learning Goal", [
        "Học AI ứng dụng",
        "Cải thiện giao tiếp",
        "Nâng cao đàm phán",
        "Tăng sự tự tin nói chuyện",
        "Kể chuyện lôi cuốn"
    ])
    nganh_nghe = st.selectbox("Your Current Industry", [
        "Công nghệ", "Giáo dục", "Kinh doanh", "Marketing", "Nghệ thuật"
    ])
    so_thich = st.selectbox("Your Personal Interest", [
        "Thích thực hành", "Thích lý thuyết", "Ưa sáng tạo", "Phân tích logic", "Thích kể chuyện"
    ])
    trinh_do_ai = st.selectbox("Your AI Knowledge Level", [
        "Chưa biết gì", "Biết cơ bản", "Ứng dụng thành thạo"
    ])
    submit_button = st.form_submit_button(label='Get Recommendation')

# Dự đoán và kết quả
if submit_button:
    khoa_hoc_goi_y = predict_course(model, encoders, muc_tieu, nganh_nghe, so_thich, trinh_do_ai)
    st.write("")
    st.success(f"🎯 Recommended Course: **{khoa_hoc_goi_y}**")
