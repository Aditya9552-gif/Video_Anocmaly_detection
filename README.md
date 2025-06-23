🚨 CCTV Video Anomaly Detection
This project focuses on detecting anomalies in CCTV surveillance footage using deep learning techniques. The pipeline uses spatiotemporal feature extraction with a pre-trained I3D model and multiple deep learning classifiers to predict whether a video segment contains anomalous activity.

🎯 Objective
To automatically detect unusual or suspicious activity in surveillance videos, which can help enhance public safety, assist security personnel, and automate incident reporting.

📁 Project Overview
1. Feature Extraction (I3D Model)
Each input video is divided into short frame sequences of 64 frames and overlap of 8 frames.

We use a pretrained RGB I3D model (Inception Inflated 3D ConvNet) to extract meaningful spatiotemporal features from each video segment.

These features capture both motion and appearance in 3D space.

2. Sequence-Level Classification
Six different deep learning models were trained on the extracted I3D features:

CNN, LSTM, GRU, Flatten-based MLP, Max-based pooling model, Mean-based pooling model

Each model predicts whether a given sequence (clip) is normal or anomalous based on a score threshold of 0.5.

3. Model Ensembling (Video-Level Prediction)
Outputs of all six models are averaged using probability averaging to create a more robust prediction.

A test video is classified as anomalous if at least 5.8% of its sequences are predicted as anomalous.

Otherwise, the video is considered normal.

📊 Evaluation Results
The ensemble model was evaluated at the video level and achieved the following metrics:

Accuracy: 88.07%

Precision: 89.13%

Recall: 86.62%

F1 Score: 87.86%

ROC AUC: 88.07%

These results indicate a strong balance between false positives and false negatives, making it suitable for real-world anomaly detection applications.

🖥️ Live Demo
👉 Try the live Streamlit app on 

Upload a video file (e.g., .mp4)

The app:

Extracts features

Runs inference

Annotates the video with anomaly scores

Displays whether the video is Normal or Anomalous

🧠 Technologies Used
PyTorch — for I3D feature extraction

TensorFlow / Keras — for classifier models

OpenCV & FFmpeg — for frame extraction and video processing

Streamlit — for building the interactive web app

Hugging Face Spaces — for deployment and hosting

📦 Files Included
streamlit_app.py — Streamlit interface code

*.keras — Trained classifier models

rgb_imagenet.pt — Pretrained I3D model weights

pytorch_i3d.py — I3D model implementation

i3d-feature-extraction-and-splitting.ipynb — Feature extraction notebook

Video_anomaly_detection_models_and_testing.ipynb — Training and evaluation

🧩 Use Cases
Smart Surveillance Systems

Retail Store Monitoring

Public Area Safety

Traffic Incident Detection

Industrial Process Monitoring
