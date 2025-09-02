from ultralytics import YOLO
import cv2
import os
import glob
from telegram_alerts import send_telegram_alert

class WeaponDetector:
    def __init__(self):
        self.model = YOLO("weights/best.pt")
        if not os.path.exists("wep_img"):
            os.makedirs("wep_img")
        self.image_counter = self.get_next_image_number()

    def get_next_image_number(self):
        files = glob.glob("wep_img/wep_*.jpg")
        numbers = [int(f.split("_")[-1].split(".")[0]) for f in files if f.split("_")[-1].split(".")[0].isdigit()]
        return max(numbers, default=0) + 1

    def detect_and_save(self, img):
        results = self.model(img)[0]
        detected = False

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            if conf > 0.5:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                detected = True

        if detected:
            filename = f"wep_img/wep_{self.image_counter}.jpg"
            cv2.imwrite(filename, img)
            print(f"[+] Weapon snapshot saved: {filename}")
            try:
                send_telegram_alert("Weapon detected!", image_path=filename)
            except Exception as e:
                print(f"Telegram alert failed: {e}")
            self.image_counter += 1

        return img


