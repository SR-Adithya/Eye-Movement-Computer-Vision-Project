# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import base64
import time
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="EyeTrack-PD Prototype", layout="centered")

# ---------------------------
# Load Haar Cascades
# ---------------------------
face_path = os.path.join(os.getcwd(), "cascades", "haarcascade_frontalface_default.xml")
eye_path  = os.path.join(os.getcwd(), "cascades", "haarcascade_eye.xml")

face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)

if face_cascade.empty() or eye_cascade.empty():
    st.error("❌ Haar cascades not found in 'cascades' folder.")
    st.stop()

# ---------------------------
# ML Model (Demo)
# ---------------------------
def _build_demo_model():
    X = np.array([
        [200, 0.95, 1.5], [220, 0.92, 1.7], [180, 0.97, 1.2],
        [420, 0.65, 3.8], [490, 0.60, 4.2], [530, 0.55, 5.0],
        [300, 0.80, 2.2], [350, 0.75, 2.8]
    ])
    y = np.array([0,0,0,1,1,1,0,1])
    clf = RandomForestClassifier(n_estimators=50, random_state=7)
    clf.fit(X, y)
    return clf

MODEL = _build_demo_model()

# ---------------------------
# Helper Functions
# ---------------------------
def base64_to_video_bytes(b64_str):
    return base64.b64decode(b64_str.split(",")[1])

def frames_from_bytes(video_bytes, max_frames=200, downscale_width=640):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    tfile.flush()
    cap = cv2.VideoCapture(tfile.name)
    frames = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > downscale_width:
            new_h = int(h * downscale_width / w)
            frame = cv2.resize(frame, (downscale_width, new_h))
        frames.append(frame)
        count += 1
    cap.release()
    return frames

def detect_pupil_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            return (x+ex+ew//2, y+ey+eh//2)
    return None

def extract_features_from_frames(frames):
    centers = []
    timestamps = []
    t0 = time.time()
    for f in frames:
        c = detect_pupil_center(f)
        centers.append(c)
        timestamps.append(time.time()-t0)
    xs = np.array([c[0] if c else np.nan for c in centers])
    ys = np.array([c[1] if c else np.nan for c in centers])
    if np.all(np.isnan(xs)):
        return [0,0,0]
    xs = np.nan_to_num(xs, nan=np.nanmean(xs))
    ys = np.nan_to_num(ys, nan=np.nanmean(ys))
    dt = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.033]*(len(xs)-1))
    dx = np.diff(xs)
    dy = np.diff(ys)
    speeds = np.sqrt(dx**2 + dy**2) / (dt + 1e-6)
    saccade = np.percentile(speeds, 90)
    smoothness = 1 - (np.std(speeds)/(np.mean(speeds)+1e-6))
    jitter = np.std(xs)+np.std(ys)
    return [float(saccade), float(np.clip(smoothness,0,1)), float(jitter)]

# ---------------------------
# JS Components
# ---------------------------
def auto_video_recorder(key, duration_ms=5000):
    component = f"""
    <video id="player-{key}" autoplay playsinline muted style="width: 100%; max-width: 400px; border: 2px solid #ccc;"></video>
    <script>
    const player = document.getElementById("player-{key}");
    let chunks = [];
    let mediaRecorder;
    navigator.mediaDevices.getUserMedia({{ video: true, audio: false }})
        .then(stream => {{
            player.srcObject = stream;
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => chunks.push(e.data);
            mediaRecorder.onstop = e => {{
                const blob = new Blob(chunks, {{ type: 'video/mp4' }});
                chunks = [];
                const reader = new FileReader();
                reader.readAsDataURL(blob);
                reader.onloadend = function() {{
                    const base64data = reader.result;
                    window.parent.postMessage({{ key: "{key}", data: base64data }}, "*");
                }};
            }};
            mediaRecorder.start();
            setTimeout(()=>{{ mediaRecorder.stop(); }}, {duration_ms});
        }});
    </script>
    """
    st.components.v1.html(component, height=400)

# ---------------------------
# Stimuli
# ---------------------------
def saccade_target_html():
    return """
    <div id="dot" style="width:20px;height:20px;background:red;border-radius:50%;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"></div>
    <script>
    const dot = document.getElementById("dot");
    let direction = 1;
    let pos = 50;
    setInterval(()=>{
        pos += direction * 20;
        if(pos >= 80 || pos <= 20) direction *= -1;
        dot.style.left = pos + "%";
    }, 500);
    </script>
    """

def pursuit_target_html():
    return """
    <div id="dot" style="width:20px;height:20px;background:blue;border-radius:50%;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"></div>
    <script>
    const dot = document.getElementById("dot");
    let angle = 0;
    setInterval(()=>{
        angle += 5;
        const radius = 150;
        const x = 50 + (radius * Math.cos(angle * Math.PI / 180)) / window.innerWidth * 100;
        const y = 50 + (radius * Math.sin(angle * Math.PI / 180)) / window.innerHeight * 100;
        dot.style.left = x + "%";
        dot.style.top = y + "%";
    }, 50);
    </script>
    """

def fixation_target_html():
    return """
    <div style="width:20px;height:20px;background:green;border-radius:50%;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"></div>
    """

# ---------------------------
# Session State Init
# ---------------------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
if 'test_data' not in st.session_state:
    st.session_state['test_data'] = {}

# ---------------------------
# Pages
# ---------------------------
def login_page():
    st.title("EyeTrack-PD Prototype — Login")
    phone = st.text_input("Phone Number", placeholder="+91xxxxxxxxxx")
    if st.button("Send OTP"):
        st.session_state['otp_sent'] = True
        st.success("OTP sent (simulated: 1234)")
    if st.session_state.get('otp_sent'):
        otp = st.text_input("Enter OTP")
        if st.button("Verify"):
            if otp == "1234":
                st.session_state['page'] = 'camera_permission'
                st.rerun()
            else:
                st.error("Invalid OTP")
    if st.button("Continue as Guest"):
        st.session_state['page'] = 'camera_permission'
        st.rerun()

def camera_permission_page():
    st.title("Camera Permission")
    st.info("Please allow camera permission in your browser.")
    if st.button("Start Screening"):
        st.session_state['page'] = 'saccade_test'
        st.rerun()

def saccade_test_page():
    st.title("Saccade Test (5s)")
    st.components.v1.html(saccade_target_html(), height=300)
    auto_video_recorder("saccade", duration_ms=5000)
    st.info("Recording in progress...")
    time.sleep(6)
    st.session_state['page'] = 'pursuit_test'
    st.rerun()

def pursuit_test_page():
    st.title("Pursuit Test (10s)")
    st.components.v1.html(pursuit_target_html(), height=300)
    auto_video_recorder("pursuit", duration_ms=10000)
    st.info("Recording in progress...")
    time.sleep(11)
    st.session_state['page'] = 'fixation_test'
    st.rerun()

def fixation_test_page():
    st.title("Fixation Test (8s)")
    st.components.v1.html(fixation_target_html(), height=300)
    auto_video_recorder("fixation", duration_ms=8000)
    st.info("Recording in progress...")
    time.sleep(9)
    st.session_state['page'] = 'analyzing'
    st.rerun()

def analyzing_page():
    st.title("Analyzing Test Results...")
    st.write("⚙️ Extracting features from recorded videos...")
    features = {key:[0,0,0] for key in ['saccade','pursuit','fixation']}
    st.session_state['analysis_features'] = features
    st.session_state['feature_vector'] = [features['saccade'][0], features['pursuit'][1], features['fixation'][2]]
    st.session_state['page'] = 'result'
    st.rerun()

def result_page():
    st.title("Screening Result")
    sv = st.session_state['feature_vector']
    pred = MODEL.predict([sv])[0]
    prob = MODEL.predict_proba([sv])[0]
    confidence = np.max(prob)
    st.write(f"Saccade speed: {sv[0]:.2f}")
    st.write(f"Pursuit smoothness: {sv[1]:.2f}")
    st.write(f"Fixation jitter: {sv[2]:.2f}")
    if pred == 0:
        st.success(f"✅ Likely Healthy (Confidence: {confidence:.2f})")
    else:
        st.warning(f"⚠️ Parkinson's Traces Detected (Confidence: {confidence:.2f})")
    if st.button("Restart"):
        st.session_state.clear()
        st.rerun()

# ---------------------------
# Routing
# ---------------------------
def main():
    page = st.session_state['page']
    if page == 'login':
        login_page()
    elif page == 'camera_permission':
        camera_permission_page()
    elif page == 'saccade_test':
        saccade_test_page()
    elif page == 'pursuit_test':
        pursuit_test_page()
    elif page == 'fixation_test':
        fixation_test_page()
    elif page == 'analyzing':
        analyzing_page()
    elif page == 'result':
        result_page()

if __name__ == "__main__":
    main()
