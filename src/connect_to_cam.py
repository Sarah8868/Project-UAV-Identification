import cv2
import requests
import time
import numpy as np
from kalman_utils import init_kalman
from yolo_utils import detect_uav
from config import CAMERA_URL, ESP32_ALERT_URL



# חיבור למצלמה
cap = cv2.VideoCapture(CAMERA_URL)

# קלמן
kalman = init_kalman()
tracking = False
missed_frames = 0
max_missed = 10

# משתנים להגבלת התראות
last_alert_time = 0
ALERT_INTERVAL = 60  # שניות

while True:
    ret, frame = cap.read()
    if not ret:
        print("בעיה בקריאת וידאו")
        break

    detections, yolo_result = detect_uav(frame)
    detected = False

    for (x1, y1, x2, y2), conf in detections:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        tracking = True
        missed_frames = 0
        detected = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"UAV: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if tracking:
        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        cv2.circle(frame, (pred_x, pred_y), 8, (0, 0, 255), -1)
        cv2.putText(frame, "Tracking", (pred_x + 10, pred_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if not detected:
            missed_frames += 1
            if missed_frames > max_missed:
                tracking = False

    # שליחת התראה רק אם עברו 60 שניות מהפעם הקודמת
    current_time = time.time()
    if detected and (current_time - last_alert_time > ALERT_INTERVAL):
        print("זוהה כטב\"ם! שולחת התראה ל-ESP32...")
        try:
            response = requests.get(ESP32_ALERT_URL, timeout=3)
            if response.status_code == 200:
                print("התראה התקבלה בהצלחה ב־ESP32")
            else:
                print(f"שגיאה בתגובה מה־ESP32: {response.status_code}")
        except Exception as e:
            print("בעיה בשליחת ההתראה:", e)

        last_alert_time = current_time  # עדכון זמן ההתראה האחרונה

    # תצוגה
    cv2.imshow("UAV Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
