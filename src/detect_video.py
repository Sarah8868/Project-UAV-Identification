from config import BEST_MODEL_PATH, DETECTION_VIDEOS_DIR
import cv2
import os
from ultralytics import YOLO

# טען את המודל
model = YOLO(BEST_MODEL_PATH)

# פתח את קובץ הוידאו
video_path = "../data/videos/20250402-2000-36.3702551.mp4"
vidObj = cv2.VideoCapture(video_path)

# קבל פרטים על הוידאו
frame_width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vidObj.get(cv2.CAP_PROP_FPS)

# יצירת שם קובץ פלט לפי שם קובץ המקור
video_name = os.path.basename(video_path).replace(" ", "_")
output_path = os.path.join(DETECTION_VIDEOS_DIR, f"detection_{video_name}")

# ודא שהתיקייה קיימת
os.makedirs(DETECTION_VIDEOS_DIR, exist_ok=True)

# הגדר קובץ פלט
out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

while True:
    success, frame = vidObj.read()
    if not success:
        break

    # רץ זיהוי על הפריים
    results = model(frame)

    # עבור כל זיהוי בפריים
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            confidence = box.conf[0]

            if confidence > 0.6:  # סף ביטחון
                # צייר תיבה
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # הראה את הפריים
    cv2.imshow("Detections", frame)

    # כתוב את הפריים לקובץ הפלט
    out.write(frame)

    # יציאה אם לוחצים על 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# סגור הכל
vidObj.release()
out.release()
cv2.destroyAllWindows()
print(f"הסרטון נשמר בהצלחה: {output_path}")
