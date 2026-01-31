import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import csv
from datetime import datetime
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Fitness Tracker PRO", layout="wide")

# ----------------- Setup -----------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ----------------- Counters and Stage -----------------
counters = {"squat":0, "pushup":0, "bicep":0}
stage = {"squat":None, "pushup":None, "bicep":None}

# ----------------- Smoothing Angles -----------------
angle_history = {"squat":deque(maxlen=5), "pushup":deque(maxlen=5), "bicep":deque(maxlen=5)}

# ----------------- Streamlit Layout -----------------
st.title("🏋️ AI Fitness Tracker PRO")
st.sidebar.header("Controls")

# Exercise selection
exercise_options = st.sidebar.multiselect(
    "Select Exercises to Track", ["squat","pushup","bicep"], default=["squat","pushup","bicep"]
)

start_session = st.sidebar.button("Start Workout")
reset_session = st.sidebar.button("Reset Counters")

if reset_session:
    counters = {"squat":0, "pushup":0, "bicep":0}
    stage = {"squat":None, "pushup":None, "bicep":None}
    st.sidebar.success("Counters reset!")

# Webcam & graphs placeholders
frame_placeholder = st.empty()
counter_placeholder = st.empty()
graph_placeholder = st.empty()
timer_placeholder = st.empty()

# ----------------- Angle Calculation -----------------
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# ----------------- Historical Data -----------------
def load_history():
    try:
        df = pd.read_csv("workout_log.csv", names=["Datetime","Squats","Pushups","BicepCurls","Duration"])
        return df
    except:
        return pd.DataFrame(columns=["Datetime","Squats","Pushups","BicepCurls","Duration"])

history_df = load_history()

# ----------------- Start Video Capture -----------------
cap = cv2.VideoCapture(0)
start_time = time.time()

# Reps history for live graph
time_history = []
squat_history = []
pushup_history = []
bicep_history = []

while start_session:
    ret, frame = cap.read()
    if not ret:
        st.warning("Cannot access webcam")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- Squat ---
        if "squat" in exercise_options:
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            knee_angle = calculate_angle(hip,knee,ankle)
            angle_history["squat"].append(knee_angle)
            knee_angle_avg = np.mean(angle_history["squat"])

            if knee_angle_avg < 95: stage["squat"] = "down"
            if knee_angle_avg > 160 and stage["squat"] == "down":
                stage["squat"] = "up"
                counters["squat"] += 1

            cv2.putText(frame,f'Squat Angle: {int(knee_angle_avg)}', 
                        tuple(np.multiply(knee,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2,cv2.LINE_AA)

        # --- Push-up ---
        if "pushup" in exercise_options:
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            elbow_angle = calculate_angle(shoulder,elbow,wrist)
            angle_history["pushup"].append(elbow_angle)
            elbow_angle_avg = np.mean(angle_history["pushup"])

            if elbow_angle_avg < 95: stage["pushup"] = "down"
            if elbow_angle_avg > 165 and stage["pushup"] == "down":
                stage["pushup"] = "up"
                counters["pushup"] += 1

            cv2.putText(frame,f'Elbow Angle: {int(elbow_angle_avg)}',
                        tuple(np.multiply(elbow,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)

        # --- Bicep Curl ---
        if "bicep" in exercise_options:
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            curl_angle = calculate_angle(shoulder_l,elbow_l,wrist_l)
            angle_history["bicep"].append(curl_angle)
            curl_angle_avg = np.mean(angle_history["bicep"])

            if curl_angle_avg > 160: stage["bicep"] = "down"
            if curl_angle_avg < 50 and stage["bicep"] == "down":
                stage["bicep"] = "up"
                counters["bicep"] += 1

            cv2.putText(frame,f'Curl Angle: {int(curl_angle_avg)}',
                        tuple(np.multiply(elbow_l,[640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)

    # --- Update histories for live graph ---
    current_time = round(time.time() - start_time,1)
    time_history.append(current_time)
    squat_history.append(counters["squat"])
    pushup_history.append(counters["pushup"])
    bicep_history.append(counters["bicep"])

    # --- Display counters & timer ---
    counter_placeholder.markdown(f"""
    **Time:** {int(current_time)} sec  
    **Squats:** {counters['squat']}  
    **Push-ups:** {counters['pushup']}  
    **Bicep Curls:** {counters['bicep']}  
    """)

    # --- Display webcam frame ---
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # --- Plot live graphs ---
    fig, ax = plt.subplots(figsize=(8,3))
    if "squat" in exercise_options: ax.plot(time_history,squat_history,label="Squats",color="blue")
    if "pushup" in exercise_options: ax.plot(time_history,pushup_history,label="Push-ups",color="green")
    if "bicep" in exercise_options: ax.plot(time_history,bicep_history,label="Bicep Curls",color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reps")
    ax.legend()
    ax.grid(True)
    graph_placeholder.pyplot(fig)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- Save session data ---
end_time = time.time()
duration = end_time - start_time
date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open("workout_log.csv","a",newline="") as f:
    writer = csv.writer(f)
    writer.writerow([date_time, counters["squat"], counters["pushup"], counters["bicep"], round(duration,2)])

cap.release()
cv2.destroyAllWindows()
st.success("Workout session ended! Data saved to workout_log.csv")

# --- Display historical stats ---
st.subheader("📊 Historical Workout Summary")
if not history_df.empty:
    st.dataframe(history_df.tail(10))
    st.line_chart(history_df.set_index("Datetime")[["Squats","Pushups","BicepCurls"]])
else:
    st.info("No historical workout data found yet.")