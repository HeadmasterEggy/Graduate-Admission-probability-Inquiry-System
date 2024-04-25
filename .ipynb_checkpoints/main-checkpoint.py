import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from joblib import load



# 加载模型
model = load_model('model.keras')
scaler = load('scaler.joblib')  # 加载标准化器

# 创建Web页面的标题和介绍
st.title('Graduate Admission Prediction')
st.write("Please enter the application details to predict the chances of admission.")

# 创建输入字段
gre_score = st.number_input('GRE Score', min_value=260, max_value=340, value=315)
toefl_score = st.number_input('TOEFL Score', min_value=0, max_value=120, value=100)
university_rating = st.number_input('University Rating', min_value=1, max_value=5, value=3)
sop = st.number_input('Statement of Purpose (SOP)', min_value=1, max_value=5, value=3)
lor = st.number_input('Letter of Recommendation (LOR)', min_value=1, max_value=5, value=3)
cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, value=8.0)
research = st.selectbox('Research Experience', options=[0, 1])


# 预测函数
def predict_admission(data):
    # 将数据转换为 NumPy 数组并应用标准化
    data_array = np.array([data])
    data_scaled = scaler.transform(data_array)  # 标准化输入数据
    predictions = model.predict(data_scaled)  # 使用模型进行预测
    return predictions[0][0]


# 当用户点击按钮时执行预测
if st.button('Predict'):
    prediction = predict_admission([gre_score, toefl_score, university_rating, sop, lor, cgpa, research])
    st.write(f"Predicted Chance of Admit: {prediction * 100:.2f}%")
