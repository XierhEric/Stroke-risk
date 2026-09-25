import streamlit as st
import pandas as pd
import joblib

# 1. 頁面基本設定 (必須寫在第一行，設定網頁標籤與展開模式)
st.set_page_config(
    page_title="臨床中風風險評估系統",
    page_icon="⚕️",
    layout="centered"
)

# 2. 載入我們訓練好的 AI 大腦
model = joblib.load('stroke_mvp_model.pkl')

# ==========================================
# 側邊欄 (Sidebar)：專屬醫護人員的輸入區
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100) # 加上一個醫療小圖示
st.sidebar.header("📋 病患生理數據輸入")
st.sidebar.markdown("請設定下方參數進行風險模擬：")

age = st.sidebar.number_input("年齡 (Age)", min_value=0, max_value=120, value=60, step=1)
bmi = st.sidebar.number_input("身體質量指數 (BMI)", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.sidebar.number_input("平均血糖 (Avg Glucose Level)", min_value=50.0, max_value=300.0, value=100.0)
hypertension = st.sidebar.selectbox("是否有高血壓?", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
heart_disease = st.sidebar.selectbox("是否有心臟病史?", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

# 將按鈕也放在側邊欄，並讓它填滿寬度
run_button = st.sidebar.button("進行風險評估", type="primary", use_container_width=True)

# ==========================================
# 主畫面 (Main Screen)：顯示決策結果
# ==========================================
st.title("🏥 臨床中風風險評估系統")
st.markdown("本系統採用機器學習 **Random Forest 演算法**，輔助醫療人員進行快速風險分級。")
st.divider()

if run_button:
    # 整理輸入資料
    input_data = pd.DataFrame({
        'age': [age],
        'avg_glucose_level': [glucose],
        'bmi': [bmi],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease]
    })
    
    # 讓模型進行預測
    prediction = model.predict(input_data)[0]
    
    # 【關鍵優化】取得模型預測為「中風(1)」的機率
    probability = model.predict_proba(input_data)[0][1]
    
    # 顯示結果儀表板
    st.subheader("📊 風險評估結果")
    
    # 用大字體顯示機率數字
    st.metric(label="中風風險機率 (Stroke Risk Probability)", value=f"{probability * 100:.1f}%")
    
    # 動態進度條 (機率越高，長度越長)
    st.progress(float(probability))
    
    # 根據預測結果給予不同顏色的警語
    if prediction == 1:
        st.error("⚠️ **警告：該病患被歸類為【中風高風險群】！**\n\n系統建議：請立即安排心血管詳細檢查，並啟動預防性照護方案。")
    else:
        st.success("✅ **評估結果：該病患目前為【低風險】。**\n\n系統建議：請持續保持健康作息與定期追蹤。")

else:
    # 還沒按按鈕時的提示畫面
    st.info("👈 請在左側欄位輸入病患的各項生理數據，並點擊「進行風險評估」。")
