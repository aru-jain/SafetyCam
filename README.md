# SafetyCam
Built an AI-powered security system that detects weapons (YOLOv8) and hand distress signals (MediaPipe) in real time, sending instant SOS alerts with live location via Telegram. Integrated a Flask backend, Streamlit dashboard, and frontend UI to ensure rapid emergency response.  

🚨 Features
1. Real-time Distress Gesture Detection — app.py

Accesses the system camera.

Detects distress hand gestures using a trained model.

Triggers alerts automatically when a gesture is identified.

Captures images/video as evidence.

2. Weapon Detection (Knife, Pistol, Grenade) — weapon_detection.py

Uses object-detection models to identify dangerous weapons.

Supports knives, pistols, and grenades.

Automatically logs detection results.

Saves evidence in the wep_img/ folder.

3. Evidence Viewer — web.py (or web1.py)

Displays captured photos and slow-motion videos of suspects/cases.

Helps security personnel quickly review incidents.

Simple and clean web interface.

4. Location Fetching — web2.py

Retrieves the geolocation (e.g., address / coordinates).

Used for identifying where the incident occurred.

Can integrate with alerts, logs, or live monitoring.


🗂 Project Structure

model/                       → YOLO model for detection
templates/                  → HTML templates for web interface
utils/                      → Helper scripts
wep_img/                    → Captured weapon detection images
app.py                      → Distress gesture detection + camera access
weapon_detection.py         → Weapon detection module
web.py (web1.py)            → Shows photos & slow-motion videos
web2.py                     → Location fetching service
telegram_alerts.py          → Sends Telegram notifications to authorities
testt.py                    → Test script
README.md                   → Project documentation




🚀 How It Works
1. Run ONLY app.py

This single file manages everything:

Opens webcam

Detects distress gestures

Detects weapons (via imported modules)

Saves images/videos

Sends alerts (optional)

You do not run weapon_detection.py, eye_detection.py, or other modules manually — they work as imported helper functions.

2. View Evidence

Run:

python web.py

This shows:

Captured photos

Slow-motion videos

3. Fetch Location

Run:

python web2.py

This returns the incident’s approximate location.


🔹 Tech Stack

Computer Vision: Python, OpenCV

AI Models: YOLOv8 (Weapon Detection), MediaPipe (Hand Gestures)

Alert & Location: Telegram Bot API, Geolocation API

Backend & Frontend: Flask, Streamlit
