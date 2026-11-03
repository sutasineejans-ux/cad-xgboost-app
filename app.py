import streamlit as st
import numpy as np
import joblib
import json

@st.cache_resource
def load_resources():
    model  = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_names.json') as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_resources()

st.set_page_config(page_title="ประเมินความเสี่ยง CAD", 
                   page_icon="❤️", layout="centered")
st.title("❤️ ประเมินความเสี่ยงโรคหลอดเลือดหัวใจ (CAD)")
st.caption("สำหรับประชาชนไทยอายุ 35 ปีขึ้นไป")
st.divider()

# รับค่าตามชื่อตัวแปรโมเดลก่อน (ยังไม่ mapping)
input_values = {}
cols = st.columns(2)
for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        val = st.number_input(f"{feature}", value=0.0)
        input_values[feature] = val

st.divider()

if st.button("🔍 ประเมินความเสี่ยง", 
             use_container_width=True, type="primary"):

    input_array  = np.array([[input_values[f] for f in feature_names]])
    input_scaled = scaler.transform(input_array)
    prob         = float(model.predict_proba(input_scaled)[0][1])
    risk_pct     = prob * 100

    st.subheader("📊 ผลการประเมิน")

    if prob >= 0.5:
        st.error("⚠️ ความเสี่ยงสูง ต่อโรคหลอดเลือดหัวใจ")
    else:
        st.success("✅ ความเสี่ยงต่ำ ต่อโรคหลอดเลือดหัวใจ")

    st.metric(label="ระดับความเสี่ยง", value=f"{risk_pct:.1f}%")
    st.progress(min(prob, 1.0))
    st.caption("⚠️ ผลนี้เป็นการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยทางการแพทย์")

st.divider()
st.caption("Biomedical Data Science | Kasetsart University")