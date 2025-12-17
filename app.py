import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from utils.unet_model import build_super_light_unet
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="USG Liver",
    page_icon="🩺",
    layout="centered"
)

IMG_SIZE = (128, 128)
CLASS_NAMES = ["Normal", "Benign", "Malignant"]

ROI_ALPHA = 1.5   

@st.cache_resource
def load_models():
    unet = build_super_light_unet(input_size=(128,128,1))
    unet.load_weights("models/unet_best.h5")
    unet.trainable = False

    clf = load_model("models/final_classifier.h5")
    return unet, clf

unet, clf = load_models()

def preprocess_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

def segment_and_attention(img_norm):
    inp = img_norm[None, ..., None]
    pred = unet.predict(inp, verbose=0)[0]

    lesion_map = pred[...,1] + pred[...,2]
    lesion_map = np.clip(lesion_map, 0, 1)

    lesion_strength = np.mean(lesion_map)

    
    if lesion_strength < 0.05:
        attended = img_norm.copy()
    else:
        attended = img_norm * (1 + ROI_ALPHA * lesion_map)
        attended = np.clip(attended, 0, 1)

    return lesion_map, attended, lesion_strength

def overlay_segmentation(img, lesion):
    img_rgb = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    heatmap = np.zeros_like(img_rgb)
    heatmap[...,0] = (lesion * 255).astype(np.uint8)  

    overlay = cv2.addWeighted(img_rgb, 0.7, heatmap, 0.5, 0)
    return overlay

st.title("🩺 Segmentasi dan Klasifikasi Lesi Hati pada Citra USG Hati Berbasis Deep Learning")
st.markdown("""
Aplikasi ini menggunakan **Deep Learning (U-Net + CNN)** untuk:
- Segmentasi area lesi
- Klasifikasi kondisi hati
""")

uploaded = st.file_uploader(
    "📤 Upload citra USG (PNG / JPG)",
    type=["png","jpg","jpeg"]
)

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Gagal membaca citra")
    else:
        img_norm = preprocess_image(img)

        lesion_map, attended, lesion_strength = segment_and_attention(img_norm)

        
        pred_cls = clf.predict(attended[None, ..., None], verbose=0)[0]
        cls_idx = np.argmax(pred_cls)
        label = CLASS_NAMES[cls_idx]
        confidence = float(pred_cls[cls_idx]) * 100

        
        overlay = overlay_segmentation(img_norm, lesion_map)

        st.subheader("📊 Hasil Analisis")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img_norm, caption="Citra USG Asli", clamp=True)

        with col2:
            st.image(overlay, caption="Visualisasi Segmentasi Lesi", clamp=True)

        st.markdown("---")

        if label == "Normal":
            st.success(f"🟢 **Prediksi: {label}**")
        elif label == "Benign":
            st.warning(f"🟡 **Prediksi: {label}**")
        else:
            st.error(f"🔴 **Prediksi: {label}**")

        st.markdown(f"**Confidence:** `{confidence:.2f}%`")
        st.caption(f"Lesion strength (mean): {lesion_strength:.4f}")

else:
    st.info("Silakan upload citra USG untuk memulai analisis")
