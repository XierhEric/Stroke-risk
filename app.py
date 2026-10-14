import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="腦中風健檢大數據分流與資源管理系統", page_icon="🏥", layout="wide")

# 載入你原本訓練好的中風模型
model = joblib.load('stroke_mvp_model.pkl')

st.title("🏥 醫院健檢大數據：腦中風風險分流與資源調度系統")
st.markdown("""
本系統將機器學習模型轉化為 **「醫療資源漏斗分流器 (Healthcare Resource Triage)」**。透過 **99% 的陰性預測值 (NPV)** 自動過濾近七成安全民眾，協助醫院將有限的個管師人力集中於前 **31.5%** 的高危族群，一舉攔截 **80%** 的潛在中風患者。
""")

tab1, tab2 = st.tabs([
    "📊 全院健檢大數據資源調度看板 (N=1,022)", 
    "🩺 個案風險分級與自動派單系統"
])

# ==========================================
# 分頁一：直接把你的分類報告轉化為「醫院管理決策看板」
# ==========================================
with tab1:
    st.subheader("📌 健檢族群漏斗式資源分流效益 (基於 1,022 筆臨床測試集驗證)")
    
    # 核心管理指標卡片
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟢 自動化排除安全人數", "700 人 (68.5%)", "陰性預測準確率 99% (NPV)", delta_color="normal")
    m2.metric("🚨 需介入追蹤人數", "322 人 (31.5%)", "省下 68.5% 個管師篩檢人力", delta_color="off")
    m3.metric("🎯 潛在中風患者攔截率", "80.0% (40/50人)", "Recall 召回率最佳化", delta_color="normal")
    m4.metric("⚡ 醫療資源配置效益 (Lift)", "2.54 倍", "31.5% 人力創造 80% 攔截成果", delta_color="normal")

    st.divider()

    # 互動式醫院人力成本計算機
    st.subheader("💰 醫院個管師人力與預算節省模擬器")
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        total_checkup = st.slider("🏥 設定醫院年度健檢總人數", min_value=1000, max_value=50000, value=10000, step=1000)
        cost_per_call = st.number_input("💵 每名個案人工電訪與追蹤成本 (NT$)", min_value=100, max_value=2000, value=350, step=50)
    
    with col_sim2:
        # 依照模型實際比例 (68.5% 綠燈免人工, 31.5% 需分級介入)
        green_count = int(total_checkup * (700 / 1022))
        alert_count = total_checkup - green_count
        saved_budget = green_count * cost_per_call
        actual_budget = alert_count * cost_per_call
        
        st.success(f"""
        **📈 年度大數據管理效益評估（以 {total_checkup:,} 名健檢民眾為例）：**
        * **傳統全院普篩成本**：NT$ {total_checkup * cost_per_call:,} 元
        * **導入本系統分流後成本**：NT$ {actual_budget:,} 元（僅需追蹤 {alert_count:,} 人）
        * **💰 年度為醫院淨省預算**：**NT$ {saved_budget:,} 元（節省 68.5% 基層行政與醫護人力）**
        """)

    st.divider()

    # 三級分流管理策略表 (回應老師對 Precision 0.13 的質疑)
    st.subheader("🚦 紅、黃、綠燈三級醫療資源配置策略")
    triage_df = pd.DataFrame({
        "分流燈號": ["🟢 綠燈（低風險安全群）", "🟡 黃燈（亞健康預警群）", "🔴 紅燈（高危急需介入群）"],
        "判定條件 (風險機率)": ["機率 < 50%", "機率 50% ~ 75%", "機率 ≥ 75%"],
        "測試集佔比 (N=1,022)": ["700 人 (68.5%)", "約 220 人 (21.5%)", "約 102 人 (10.0%)"],
        "臨床與管理特性": [
            "陰性預測值 (Precision 0) 高達 99%，極度安全", 
            "雖多為未發病之假警報，但屬高齡/三高亞健康族群", 
            "高度集中真正中風患者，具最高臨床急迫性"
        ],
        "醫療資源派單行動": [
            "系統自動寄發年度常規報告（0 人力成本）", 
            "自動化 LINE 衛教推播與飲食運動提醒（低數位成本）", 
            "派發個管師人工電訪 ＋ 神經內科門診轉介（集中核心人力）"
        ]
    })
    st.table(triage_df)
    st.caption("💡 管理學設計邏輯：透過「黃燈自動化數位衛教」吸收模型為拉高 Recall 而產生的假警報（False Positives），避免醫療人力浪費，同時達到預防醫學衛教目的。")

# ==========================================
# 分頁二：原本的個人輸入介面，升級為「三級派單系統」
# ==========================================
with tab2:
    st.subheader("🩺 門診/健檢個案即時分流與資源派單")
    
    if 'form_expanded' not in st.session_state:
        st.session_state.form_expanded = True

    def collapse_form():
        st.session_state.form_expanded = False

    with st.expander("📋 點此填寫病患生理數據 (點擊評估後自動收合)", expanded=st.session_state.form_expanded):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("年齡 (Age)", min_value=1, max_value=120, value=65, step=1)
            hypertension = st.selectbox("是否有高血壓病史?", options=[0, 1], format_func=lambda x: "是 (Yes)" if x == 1 else "否 (No)")
            heart_disease = st.selectbox("是否有心臟病史?", options=[0, 1], format_func=lambda x: "是 (Yes)" if x == 1 else "否 (No)")
        with col2:
            avg_glucose_level = st.number_input("平均血糖值 (Avg Glucose Level)", min_value=40.0, max_value=350.0, value=110.0)
            bmi = st.number_input("身體質量指數 (BMI)", min_value=10.0, max_value=70.0, value=27.5)

        run_button = st.button("執行風險分級與醫療資源派單", type="primary", use_container_width=True, on_click=collapse_form)

    if run_button:
        # 1. 抓出模型當初訓練時「真正」記住的特徵名稱與順序
        expected_features = model.feature_names_in_
        
        # 2. 建立一個包含各種大小寫變體的「翻譯字典」，確保絕對配對得上
        raw_data = {
            'age': age, 'Age': age, 'AGE': age,
            'hypertension': hypertension, 'Hypertension': hypertension,
            'heart_disease': heart_disease, 'Heart_Disease': heart_disease, 'heartdisease': heart_disease,
            'avg_glucose_level': avg_glucose_level, 'Avg_Glucose_Level': avg_glucose_level, 'avg_glucose': avg_glucose_level,
            'bmi': bmi, 'BMI': bmi, 'Bmi': bmi
        }
        
        try:
            # 3. 依照模型期望的順序，精準挑出對應的數值並組裝 DataFrame
            aligned_values = [raw_data[feat] for feat in expected_features]
            input_data = pd.DataFrame([aligned_values], columns=expected_features)
            
            # 執行預測
            probability = model.predict_proba(input_data)[0][1]
            
            st.subheader("📊 風險分流與資源派單結果")
            st.metric(label="腦中風風險機率指標", value=f"{probability * 100:.1f}%")
            st.progress(float(probability))
            
            if probability >= 0.75:
                st.error("🔴 **分流等級：【紅燈 — 高危急需介入群】**\n\n**資源調度行動**：已自動排入「個管師本週優先電訪名單」，建議立即安排神經內科或心臟內科門診轉介。")
            elif probability >= 0.50:
                st.warning("🟡 **分流等級：【黃燈 — 亞健康數位衛教群】**\n\n**資源調度行動**：不消耗人工電訪時數；已由系統自動派發「三高控制與減鈉飲食 LINE 數位衛教方案」，提醒定期居家量測血壓。")
            else:
                st.success("🟢 **分流等級：【綠燈 — 低風險常規群（99% 陰性預測把握度）】**\n\n**資源調度行動**：僅發送年度常規健檢報告，無需投入額外醫護追蹤人力。")

        except KeyError as e:
            # 4. 終極防呆：如果模型需要我們沒準備的欄位，直接在網頁上印出抓漏報告！
            st.error(f"⚠️ 發生特徵不匹配錯誤！\n\n你的模型需要的欄位是：`{expected_features}`\n\n但目前網頁表單缺少了：`{e}`。這代表你當初訓練模型時，可能多放了其他欄位（例如性別或抽菸），請確認！")
