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

# ==================================================
# Mapping ตัวแปร → ภาษาไทย + ประเภท input
# ==================================================
FIELD_CONFIG = {
    'age':          {'label': 'อายุ (ปี)',
                     'type': 'number', 'min': 35, 'max': 100, 'default': 50},
    'in_md_care':   {'label': 'สิทธิการรักษา',
                     'type': 'select', 'options': {
                         'บัตรทอง (30 บาท)': 1,
                         'ประกันสังคม': 2,
                         'ข้าราชการ/รัฐวิสาหกิจ': 3,
                         'ชำระเงินเอง': 4,
                         'อื่นๆ': 5}},
    'in_smk':       {'label': 'การสูบบุหรี่',
                     'type': 'select', 'options': {
                         'ไม่สูบ': 0, 'สูบ': 1}},
    'in_intent02':  {'label': 'ตั้งใจจะออกกำลังกาย/เดินแทนขึ้นลิฟต์',
                     'type': 'select', 'options': {
                         'ไม่': 0, 'ใช่': 1}},
    'sc_waist_res': {'label': 'รอบเอว (ซม.)',
                     'type': 'number', 'min': 50.0, 'max': 150.0, 'default': 80.0},
    'in_abi_ph4':   {'label': 'ความถี่ดื่มน้ำอัดลมไดเอท/แคลต่ำ',
                     'type': 'select', 'options': {
                         'ไม่ดื่มเลย': 0, 'นานๆ ครั้ง': 1,
                         'บางครั้ง': 2, 'บ่อย': 3, 'ทุกวัน': 4}},
    'in_abi_alc5':  {'label': 'การดื่มแอลกอฮอล์',
                     'type': 'select', 'options': {
                         'ไม่ดื่ม': 0, 'ดื่ม': 1}},
    'in_fd73':      {'label': 'ดื่มน้ำหวานไดเอท/แคลต่ำ',
                     'type': 'select', 'options': {
                         'ไม่': 0, 'ใช่': 1}},
    'in_fd66':      {'label': 'เติมผงชูรส',
                     'type': 'select', 'options': {
                         'ไม่เติม': 0, 'เติม': 1}},
    'sc_hdl_res':   {'label': 'HDL Cholesterol (mg/dL)',
                     'type': 'number', 'min': 10.0, 'max': 150.0, 'default': 50.0},
    'in_intent07':  {'label': 'ตั้งใจจะไม่สูบบุหรี่',
                     'type': 'select', 'options': {
                         'ไม่': 0, 'ใช่': 1}},
    'in_salty1':    {'label': 'เติมน้ำปลา/ซอส',
                     'type': 'select', 'options': {
                         'ไม่เติม': 0, 'เติม': 1}},
    'in_fd15':      {'label': 'กินข้าวเหนียวเป็นประจำ',
                     'type': 'select', 'options': {
                         'ไม่': 0, 'ใช่': 1}},
    'sc_tg_res':    {'label': 'Triglyceride (mg/dL)',
                     'type': 'number', 'min': 30.0, 'max': 800.0, 'default': 150.0},
    'in_fd44':      {'label': 'กินโยเกิร์ตไขมันต่ำ',
                     'type': 'select', 'options': {
                         'ไม่': 0, 'ใช่': 1}},
    'in_ht_son':    {'label': 'ลูกมีประวัติความดันโลหิตสูง',
                     'type': 'select', 'options': {
                         'ไม่ใช่': 0, 'ใช่': 1}},
    'in_edu_cad1':  {'label': 'รู้จักโรค CAD คืออะไร',
                     'type': 'select', 'options': {
                         'ไม่รู้': 0, 'รู้': 1}},
    'in_edu_cad3':  {'label': 'รู้อาการนำของ CAD',
                     'type': 'select', 'options': {
                         'ไม่รู้': 0, 'รู้': 1}},
    'gender_Male':  {'label': 'เพศ',
                     'type': 'select', 'options': {
                         'หญิง': 0, 'ชาย': 1}},
}

# ==================================================
# จัดกลุ่ม
# ==================================================
GROUPS = {
    '👤 ข้อมูลส่วนตัว': ['age', 'gender_Male', 'in_md_care', 'in_ht_son'],
    '🩺 ผลตรวจสุขภาพ': ['sc_waist_res', 'sc_hdl_res', 'sc_tg_res'],
    '🚬 พฤติกรรมสุขภาพ': ['in_smk', 'in_intent02', 'in_intent07',
                          'in_abi_ph4', 'in_abi_alc5', 'in_salty1',
                          'in_fd15', 'in_fd44', 'in_fd66', 'in_fd73'],
    '📚 ความรอบรู้สุขภาพ': ['in_edu_cad1', 'in_edu_cad3'],
}

# ==================================================
# UI
# ==================================================
st.set_page_config(page_title="ประเมินความเสี่ยง CAD",
                   page_icon="❤️", layout="centered")
st.title("❤️ ประเมินความเสี่ยงโรคหลอดเลือดหัวใจ (CAD) สำหรับประชาชนไทยอายุ 35 ปีขึ้นไป")

# Model info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("โมเดล", "XGBoost")
with col2:
    st.metric("ROC-AUC", "0.9611")
with col3:
    st.metric("จำนวนตัวแปร", "19")

st.divider()

input_values = {}

for group_name, fields in GROUPS.items():
    st.subheader(group_name)
    cols = st.columns(2)
    for i, field in enumerate(fields):
        if field not in FIELD_CONFIG:
            continue
        cfg = FIELD_CONFIG[field]
        with cols[i % 2]:
            if cfg['type'] == 'number':
                val = st.number_input(
                    cfg['label'],
                    min_value=float(cfg['min']),
                    max_value=float(cfg['max']),
                    value=float(cfg['default']),
                    step=1.0)
                input_values[field] = val
            elif cfg['type'] == 'select':
                choice = st.selectbox(cfg['label'], list(cfg['options'].keys()))
                input_values[field] = cfg['options'][choice]
    st.divider()

# ==================================================
# ปุ่มประเมิน
# ==================================================
if st.button("🔍 ประเมินความเสี่ยง",
             use_container_width=True, type="primary"):
    input_array  = np.array([[input_values.get(f, 0) for f in feature_names]])
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
    st.warning("""
⚠️ **คำเตือนสำคัญ**

ผลการประเมินนี้เป็นเพียงการ **คัดกรองความเสี่ยงเบื้องต้น** เท่านั้น 
ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้ 

กรุณาปรึกษาแพทย์หรือบุคลากรทางการแพทย์ 
เพื่อการวินิจฉัยและรักษาที่ถูกต้อง
""")

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.caption("🎓 Biomedical Data Science | Kasetsart University")
with col2:
    st.caption("🤖 Model: XGBoost | Features: 19 | ROC-AUC: 0.9611")
