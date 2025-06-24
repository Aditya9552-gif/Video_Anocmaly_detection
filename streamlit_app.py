import os
import gc
import time
import torch
import imageio
import numpy as np
import streamlit as st
import tempfile
import shutil
import cv2
import subprocess

from pytorch_i3d import InceptionI3d
from tensorflow.keras.models import load_model

TF_ENABLE_ONEDNN_OPTS = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Overlay Settings =====
MESSAGE_COLORS = {
    "Normal activity detected": (0, 255, 0),
    "Minor irregularities observed": (0, 255, 255),
    "Potential anomaly — monitor activity": (0, 165, 255),
    "High-confidence anomaly — immediate review needed": (0, 0, 255),
}

def get_message(prob):
    if prob < 0.5:
        return "Normal activity detected"
    elif prob < 0.7:
        return "Minor irregularities observed"
    elif prob < 0.9:
        return "Potential anomaly — monitor activity"
    else:
        return "High-confidence anomaly — immediate review needed"

def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

@st.cache_resource
def load_models():
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load("rgb_imagenet.pt", map_location=device), strict=False)
    i3d = i3d.to(device).eval()

    model_paths = {
        "mean": "mean.keras",
        "max": "max.keras",
        "flatten": "flatten.keras",
        "cnn": "cnn.keras",
        "lstm": "lstm.keras",
        "gru": "gru.keras"
    }
    models = {k: load_model(v) for k, v in model_paths.items()}

    dummy_input = np.zeros((1, 7, 1, 1, 1024), dtype=np.float32)
    for model in models.values():
        model.predict(dummy_input, verbose=0)

    return i3d, models

def ensure_30fps_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 30:
        return video_path
    temp_dir = tempfile.mkdtemp()
    out_path = os.path.join(temp_dir, "converted_30fps.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-r", "30", out_path], check=True)
    return out_path

def predict_anomaly_probability(seq_feature, models):
    reshaped = seq_feature.reshape(1, 7, 1, 1, 1024)
    preds = [m.predict(reshaped, verbose=0).squeeze() for m in models.values()]
    return np.mean(preds)

def annotate_video(video_path, i3d, models, progress):
    video_path = ensure_30fps_video(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    extended_height = int(height * 1.18)
    font_scale = height / 720


    # font_scale = max(0.5, min(width, height) / 600)

    output_path = tempfile.mktemp(suffix=".mp4")
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")

    frames = []
    scores = []
    seq_length = 64
    step = 64
    start_time = time.time()
    last_update = 0
    overlay_score = 0.0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        f = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
        f = f / 255.0 * 2 - 1
        frames.append(f)

        if len(frames) == seq_length:
            seq = np.array(frames, dtype=np.float32)
            seq_tensor = video_to_tensor(seq).unsqueeze(0).to(device)
            with torch.no_grad():
                features = i3d.extract_features(seq_tensor).squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            score = predict_anomaly_probability(features, models)
            overlay_score = score
            scores.append(score)
            frames = frames[step:]
            del seq_tensor, features
            torch.cuda.empty_cache()
            gc.collect()

        prob = overlay_score
        message_text = get_message(prob)
        font_color = MESSAGE_COLORS[message_text]
        overlay_text = f"Anomaly Score: {prob * 100:.1f}%"

        pad_height = int(np.ceil(extended_height / 16) * 16)
        new_frame = np.zeros((pad_height, width, 3), dtype=np.uint8)
        new_frame[:height] = frame

        cv2.putText(new_frame, overlay_text, (3, height + int(30 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1, cv2.LINE_AA)
        cv2.putText(new_frame, message_text, (3, height + int(70 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1, cv2.LINE_AA)

        new_frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        writer.append_data(new_frame_rgb)

        pct = (i + 1) / (total_frames + 1)
        elapsed = time.time() - start_time
        est_total = elapsed / pct if pct > 0 else 0
        est_remaining = est_total - elapsed
        if time.time() - last_update > 0.25:
            progress.progress(min(pct, 1.0), f"{pct*100:.1f}% | ETA: {est_remaining:.1f}s")
            last_update = time.time()

    cap.release()
    writer.close()
    progress.progress(1.0, "✅ Done!")
    return output_path, scores


# ==== Streamlit UI ====
st.set_page_config(page_title="Video Anomaly Detection", layout="wide")
st.title("🎥 Video Anomaly Detection (I3D + Keras)")
st.write("Upload a video to analyze for anomalous activity.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.markdown("#### 🔍 Analyzing...")
    progress = st.progress(0, "Initializing...")

    i3d, models = load_models()
    annotated_path, scores = annotate_video(tmp_path, i3d, models, progress)


    num_high_scores = sum(score >= 0.9 for score in scores)
    threshold = 0.058 
    required_count = max(1, int(len(scores) * threshold))
    final_label = "Anomalous" if num_high_scores >= required_count else "Normal"
    label_color = "red" if final_label == "Anomalous" else "green"


    st.markdown(f"### Final Prediction: <span style='color:{label_color}'>{final_label}</span>", unsafe_allow_html=True)
    st.markdown(f"**High-Confidence Segments**: {num_high_scores} / {len(scores)}")
 

    st.markdown("#### 🎞️ Annotated Video")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.video(annotated_path, format="video/mp4", start_time=0)
