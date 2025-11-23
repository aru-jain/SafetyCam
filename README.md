# SafetyCam
Built an AI-powered security system that detects weapons (YOLOv8) and hand distress signals (MediaPipe) in real time, sending instant SOS alerts with live location via Telegram. Integrated a Flask backend, Streamlit dashboard, and frontend UI to ensure rapid emergency response.  

📌 Features
1. Real-Time Detection — app.py

Accesses the webcam.

Detects distress gestures.

Detects weapons using helper modules.

Saves captured images and slow-motion videos.

Sends alerts via Telegram API.

All detection logic is inside app.py — no other scripts must be run.


<img width="151" height="160" alt="image" src="https://github.com/user-attachments/assets/9eec1e69-7a27-4168-982a-3a02d299dafc" />



2. Evidence Viewer — web.py / web1.py

Displays stored photos of suspects.

Shows slow-motion videos recorded during alerts.

3. Location Fetcher — web2.py

Fetches and displays geolocation where the incident was detected.

🗂️ Project Structure
model/                       → Model files for gesture & weapon detection
templates/                  → HTML templates for the web interfaces
utils/                      → Helper utility functions
wep_img/                    → Stored weapon detection images
app.py                      → Main application (camera + gesture & weapon detection)
telegram_alerts.py          → Sends Telegram alerts (optional)
web.py (or web1.py)         → Web interface to view photos & videos
web2.py                     → Shows incident location
weapon_detection.py         → Helper module imported inside app.py (NOT run separately)
testt.py                    → Test script
README.md                   → Project documentation

🚀 How It Works
Step 1 — Run Detection

Only run:

python app.py


This performs:

Webcam access

Distress gesture detection

Weapon detection (using imported helper modules)

Evidence capture (images & videos)

Alerts

You never run weapon_detection.py separately.
It is only imported into app.py as a helper.

Step 2 — View Evidence

Run:

python web.py


This opens a web interface showing:

Collected photos

Slow-motion videos

Step 3 — View Location

Run:

python web2.py


This shows the approximate location of the incident.


🔹 Tech Stack

Computer Vision: Python, OpenCV

AI Models: YOLOv8 (Weapon Detection), MediaPipe (Hand Gestures)

Alert & Location: Telegram Bot API, Geolocation API

Backend & Frontend: Flask, Streamlit
