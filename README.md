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



2. Evidence Viewer — web.py

Displays stored photos of suspects.

Shows slow-motion videos recorded during alerts.


<img width="800" height="296" alt="image" src="https://github.com/user-attachments/assets/7f2003f6-20be-475a-aaa5-30707c827e15" />


3. Location Fetcher — web2.py

Fetches and displays geolocation where the incident was detected.


<img width="800" height="386" alt="image" src="https://github.com/user-attachments/assets/a9852ac3-37ea-4091-94fb-032108badeb7" />


<img width="738" height="1600" alt="image" src="https://github.com/user-attachments/assets/cfcf18a6-c23f-495f-b5a1-522ad405e098" />



🗂️ Project Structure


<img width="876" height="355" alt="image" src="https://github.com/user-attachments/assets/76d81fb4-9f25-4cd8-8c29-084961c6cc39" />


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
