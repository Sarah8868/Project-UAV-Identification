
import cv2
import numpy as np
from ultralytics import YOLO
import os
from config import DETECTION_VIDEOS_DIR,BEST_MODEL_PATH  # 👈 ייבוא הנתיב לשמירת סרטונים

# טען את המודל
model = YOLO(BEST_MODEL_PATH)

# פתח את הווידאו
video_path = "../data/videos/20250402-2000-36.3702551.mp4"
vid = cv2.VideoCapture(video_path)

# נתוני וידאו
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv2.CAP_PROP_FPS)

# יצירת שם חדש לקובץ הפלט על בסיס שם הקובץ המקורי
video_name = os.path.basename(video_path).replace(" ", "_")
output_video_path = os.path.join(DETECTION_VIDEOS_DIR, f"tracked_{video_name}")

# ודא שהתיקייה קיימת
os.makedirs(DETECTION_VIDEOS_DIR, exist_ok=True)

# הגדרת וידאו פלט
out = cv2.VideoWriter(output_video_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

# הגדר Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

tracking = False
missed_frames = 0
max_missed = 10

while True:
    ret, frame = vid.read()
    if not ret:
        break

    results = model(frame)
    detected = False

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = box.conf[0]

            if confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                tracking = True
                missed_frames = 0
                detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls_id]}: {confidence:.2f}"
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

    cv2.imshow("Tracking", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()
print(f"הווידאו נשמר בנתיב: {output_video_path}")
