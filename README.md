# AI-Fitness-Tracker
Computer Vision–Based Smart Workout Monitoring System

AI Fitness Tracker PRO is a computer vision–based fitness monitoring system that acts like a virtual personal trainer.
It uses a webcam to detect body posture, count exercise repetitions, provide real-time feedback, and track workout progress over time — without requiring any wearable devices.

The system leverages MediaPipe Pose Estimation and OpenCV to analyze human body movements and delivers live analytics through an interactive Streamlit dashboard.

--> Key Features

- Real-Time Pose Detection using MediaPipe Pose
-Automatic Rep Counting for exercises like:
Squats
Push-ups
Bicep curls
 Angle-Based Exercise Analysis
Joint angle calculation
Rolling average smoothing for accuracy
 Live Feedback on Posture & Form
 Real-Time Performance Graphs

--> Workout History Logging
Timestamp
Repetitions
Duration
--> Progress Tracking & Trend Analysis
 CSV-Based Persistent Storage

--> Tech Stack
Category	Tools
Language	Python
Computer Vision	OpenCV
Pose Estimation	MediaPipe Pose
UI / Dashboard	Streamlit
Data Handling	Pandas
Visualization	Matplotlib, Streamlit Charts
Storage	CSV Files
Utilities	NumPy, deque (rolling average), datetime
--> Project Structure
AI-Fitness-Tracker/
│
├── main.py
├── workout_log.csv
├── code/
├── docs/
└── README.md

⚙️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Rajsinha7/AI-Fitness-Tracker.git
cd AI-Fitness-Tracker

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run main.py


--> Requirements

Python 3.8+
Webcam access
Good lighting for better pose detection

--> How It Works (System Flow)

Webcam captures live video feed
MediaPipe Pose detects body landmarks
Joint angles are calculated per frame
Rolling averages smooth noisy measurements
Exercise reps are counted based on angle thresholds

Live metrics & graphs are displayed on dashboard
