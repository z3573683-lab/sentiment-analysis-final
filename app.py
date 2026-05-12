import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import nltk
nltk.download("vader_lexicon")
model = tf.keras.models.load_model('best_multi_sentiment_model(1).keras', compile=False)

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Sentiment AI Pro", page_icon="🤖")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #007bff; color: white; font-weight: bold; }
    .res-box { padding: 20px; border-radius: 15px; text-align: center; color: white; font-size: 25px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. تحميل الملفات (الموديل والتوكنايزر) ---
@st.cache_resource
def load_assets():
    # تأكد أن هذه الملفات مرفوعة في الكولاب بنفس الأسماء
    model = tf.keras.models.load_model('best_multi_sentiment_model(1).keras')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

# --- 3. دالة تنظيف النص ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# --- 4. تشغيل التطبيق ---
try:
    model, tokenizer = load_assets()

    st.title("🤖 Professional Sentiment Analyzer")
    st.write("أدخل النص لتحليله باستخدام موديل الـ Deep Learning الخاص بك")

    user_input = st.text_area("Write your tweet here:", placeholder="e.g., This is the best day ever!")

    if st.button("Analyze Sentiment ✨"):
        if user_input.strip():
            # معالجة النص
            cleaned = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            # الـ maxlen لازم يكون نفس اللي استخدمته في التدريب (غالباً 100)
            padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

            # التوقع
            pred = model.predict(padded)
            idx = np.argmax(pred)
            conf = np.max(pred) * 100

            # خريطة الألوان والنتائج (0=Neg, 1=Neu, 2=Pos)
            labels = {
            0: ("Positive 🟢", "#2ecc71"),
            1: ("Semi-Positive 🟡", "#f1c40f"),
            2: ("Neutral ⚪", "#95a5a6"),
            3: ("Semi-Negative 🟠", "#e67e22"),
            4: ("Negative 🔴", "#e74c3c")}
            res_text, res_color = labels.get(idx, ("Unknown", "#808080"))

            st.markdown(f"<div class='res-box' style='background-color:{res_color};'>{res_text}</div>", unsafe_allow_html=True)
            st.write(f"**Confidence Level:** {conf:.2f}%")
            st.progress(int(conf))
        else:
            st.warning("Please enter some text first!")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("تأكد من رفع ملفات .keras و .pkl في الفولدر الرئيسي للكولاب قبل التشغيل.")
