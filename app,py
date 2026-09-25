import streamlit as st
import pandas as pd
import joblib

# 1. 載入我們剛才訓練好的 AI 大腦
model = joblib.load('stroke_mvp_model.pkl')

# 2. 設計網頁標題與外觀
st.title("🏥 臨床中風風險評估系統")
st.markdown("本系統採用機器學習 Random Forest 演算法，輔助醫療人員進行快速風險分級。")
st.divider()

# 3. 設計輸入介面 (切分成左右兩排)
st.header("請輸入病患生理數據：")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("年齡 (Age)", min_value=0, max_value=120, value=60)
    bmi = st.number_input("身體質量指數 (BMI)", min_value=10.0, max_value=60.0, value=25.0)

with col2:
    glucose = st.number_input("平均血糖 (Avg Glucose Level)", min_value=50.0, max_value=300.0, value=100.0)
    hypertension = st.selectbox("是否有高血壓?", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    heart_disease = st.selectbox("是否有心臟病史?", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

# 4. 預測按鈕與邏輯
st.divider()
if st.button("進行風險評估", type="primary"):
    
    # 將使用者輸入的資料，整理成模型看得懂的表格 (順序必須跟訓練時一樣)
    input_data = pd.DataFrame({
        'age': [age],
        'avg_glucose_level': [glucose],
        'bmi': [bmi],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease]
    })
    
    # 讓模型進行預測
    prediction = model.predict(input_data)[0]
    
    # 顯示判斷結果
    if prediction == 1:
        st.error("⚠️ 警告：該病患為【中風高風險群】！建議立即安排心血管詳細檢查。")
    else:
        st.success("✅ 評估結果：該病患目前為【低風險】。請持續保持健康作息。")