# alerts.py
import requests
import os


TELEGRAM_TOKEN = '8178986331:AAEv1tCmVI3jz8cTNys52BS4xP548DpJdyg'
CHAT_ID = '1116435276'

def send_telegram_alert(message, image_path=None):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, data=data)

        if image_path and os.path.exists(image_path):
            url_img = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            with open(image_path, 'rb') as photo:
                requests.post(url_img, data={"chat_id": CHAT_ID}, files={"photo": photo})
    except Exception as e:
        print(f"Telegram alert failed: {e}")
