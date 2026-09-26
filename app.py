import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="臨床中風風險評估系統", page_icon="⚕️", layout="centered")
model = joblib.load('stroke_mvp_model.pkl')

st.title("🏥 臨床中風風險評估系統")
st.markdown("本系統採用機器學習 **Random Forest 演算法**，輔助醫療人員進行快速風險分級。")

# ==========================================
# 狀態管理 (Session State)：控制面板的開關
# ==========================================
# 如果一開始還沒有這個狀態，預設把它設為 True (展開)
if 'form_expanded' not in st.session_state:
    st.session_state.form_expanded = True

# 定義一個動作：當按下預測按鈕時，把狀態改為 False (收合)
def collapse_form():
    st.session_state.form_expanded = False

# ==========================================
# 輸入區：改用「折疊面板 (Expander)」
# ==========================================
# 透過 expanded 參數綁定我們上面設定的狀態
with st.expander("📋 點此填寫 / 修改病患生理數據", expanded=st.session_state.form_expanded):
    age = st.number_input("年齡 (Age)", min_value=0, max_value=120, value=60, step=1)
    bmi = st.number_input("身體質量指數 (BMI)", min_value=10.0, max_value=60.0, value=25.0)
    glucose = st.number_input("平均血糖 (Avg Glucose Level)", min_value=50.0, max_value=300.0, value=100.0)
    hypertension = st.selectbox("是否有高血壓?", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    heart_disease = st.selectbox("是否有心臟病史?", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    
    # 按鈕加上 on_click 參數，綁定「自動收合」的動作
    run_button = st.button("進行風險評估", type="primary", use_container_width=True, on_click=collapse_form)

st.divider()

# ==========================================
# 顯示結果區 (與先前相同)
# ==========================================
if run_button:
    input_data = pd.DataFrame({
        'age': [age],
        'avg_glucose_level': [glucose],
        'bmi': [bmi],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("📊 風險評估結果")
    st.metric(label="中風風險機率 (Stroke Risk Probability)", value=f"{probability * 100:.1f}%")
    st.progress(float(probability))
    
    if prediction == 1:
        st.error("⚠️ **警告：該病患被歸類為【中風高風險群】！**\n\n系統建議：請立即安排心血管詳細檢查，並啟動預防性照護方案。")
    else:
        st.success("✅ **評估結果：該病患目前為【低風險】。**\n\n系統建議：請持續保持健康作息與定期追蹤。")
